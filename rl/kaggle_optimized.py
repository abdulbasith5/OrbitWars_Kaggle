"""
kaggle_optimized.py — Resource-maximized training cell for Kaggle T4 x2.

Key optimizations:
  - GPU 0: policy training (forward + backward)
  - GPU 1: self-play opponent inference (non-blocking)
  - 16 parallel envs (fills all CPU cores)
  - batch_size=1024 (better GPU utilization)
  - rollout_steps=4096 (fewer PPO overhead calls)
  - Auto-backup checkpoints to /kaggle/working/ root
  
Paste this entire file as a single Kaggle cell after setup cells.
"""

import os, sys, time, random, glob, shutil
from collections import deque
from pathlib import Path
import numpy as np
import torch

# ── Paths (set by earlier setup cells) ───────────────────────────────────────
REPO   = '/kaggle/working/OrbitWars_Kaggle'
RL_DIR = f'{REPO}/rl'
sys.path.insert(0, RL_DIR); sys.path.insert(0, REPO)
os.makedirs(f'{RL_DIR}/checkpoints', exist_ok=True)

from src.config import default_config
from src.env import OrbitWarsEnv
from src.features import self_feature_dim, candidate_feature_dim, global_feature_dim, encode_turn
from src.opponents import RandomOpponent, V11Opponent
from src.policy import PlanetPolicy, sample_actions
from src.ppo import RolloutBuffer, Transition, ppo_update

# ── GPU setup ─────────────────────────────────────────────────────────────────
gpu0 = torch.device('cuda:0')
gpu1 = torch.device('cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0')
print(f"GPUs: {torch.cuda.device_count()} | Train={gpu0} | Opponent={gpu1}")

# ── Config ────────────────────────────────────────────────────────────────────
cfg = default_config()
cfg.model.hidden_size   = 256
cfg.model.num_heads     = 4
cfg.model.dropout       = 0.05
cfg.ppo.rollout_steps   = 4096    # 2x larger → fewer PPO overhead calls
cfg.ppo.batch_size      = 1024    # 2x larger → better GPU utilization
cfg.ppo.lr              = 2e-4
cfg.ppo.entropy_coef    = 0.08    # slightly higher → prevent collapse
cfg.ppo.ppo_epochs      = 4
cfg.use_shaping_reward  = True
cfg.alternate_player_sides = True

NUM_ENVS         = 16     # fills all Kaggle CPU cores
TOTAL_UPDATES    = 3000
RANDOM_END       = 200
SELFPLAY_START   = 800
SELFPLAY_SYNC    = 50
LOG_EVERY        = 10
CKPT_EVERY       = 100    # save every 100 updates
RESUME_CKPT      = None   # set to path string to resume, e.g. 'checkpoints/ckpt_000400.pt'

CKPT_DIR = Path(f'{RL_DIR}/checkpoints')

random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ── Policy on GPU 0 ───────────────────────────────────────────────────────────
policy = PlanetPolicy(
    self_dim=self_feature_dim(), candidate_dim=candidate_feature_dim(),
    global_dim=global_feature_dim(), candidate_count=cfg.env.candidate_count,
    hidden_size=cfg.model.hidden_size, num_heads=cfg.model.num_heads,
    dropout=cfg.model.dropout,
).to(gpu0)

# ── Self-play opponent on GPU 1 ───────────────────────────────────────────────
def _build_opp_policy():
    """Always build opp_policy fresh from current cfg so keys/shapes always match."""
    global opp_policy
    opp_policy = PlanetPolicy(
        self_dim=self_feature_dim(), candidate_dim=candidate_feature_dim(),
        global_dim=global_feature_dim(), candidate_count=cfg.env.candidate_count,
        hidden_size=cfg.model.hidden_size, num_heads=cfg.model.num_heads, dropout=0.0,
    ).to(gpu1)
    opp_policy.eval()

_build_opp_policy()

def sync_opp():
    """Sync opponent weights from training policy. Rebuilds if architectures diverged."""
    global opp_policy
    try:
        opp_policy.load_state_dict(policy.state_dict())
    except RuntimeError:
        print("[sync_opp] Architecture mismatch — rebuilding opp_policy from current cfg.")
        _build_opp_policy()
        opp_policy.load_state_dict(policy.state_dict())

class SelfPlayOpp:
    def act(self, obs):
        batch = encode_turn(obs, cfg.env)
        if batch.self_features.shape[0] == 0: return []
        with torch.inference_mode():
            out = opp_policy(
                torch.from_numpy(batch.self_features).to(gpu1),
                torch.from_numpy(batch.candidate_features).to(gpu1),
                torch.from_numpy(batch.global_features).to(gpu1),
                torch.from_numpy(batch.candidate_mask).to(gpu1),
            )
            s = sample_actions(out, deterministic=True)
        idxs = s.target_index.cpu().numpy()
        moves = []
        for i, ctx in enumerate(batch.contexts):
            idx = int(idxs[i])
            if idx == 0 or not ctx.candidate_mask[idx]: continue
            ships = int(ctx.ship_counts[idx])
            if ships <= 0: continue
            moves.append([ctx.source_id, float(ctx.target_angles[idx]), ships])
        return moves

# ── Resume ────────────────────────────────────────────────────────────────────
# Auto-find latest checkpoint if RESUME_CKPT not set
if RESUME_CKPT is None:
    existing = sorted(CKPT_DIR.glob('ckpt_0*.pt'))
    if existing:
        RESUME_CKPT = str(existing[-1])
        print(f"Auto-resume: {existing[-1].name}")

start_update = 0
if RESUME_CKPT and os.path.isfile(RESUME_CKPT):
    ck = torch.load(RESUME_CKPT, map_location='cpu', weights_only=False)
    ck_cfg = ck.get('cfg', None)
    if ck_cfg:
        try:
            cfg.model.hidden_size = ck_cfg.model.hidden_size
            cfg.model.num_heads   = ck_cfg.model.num_heads
        except: pass
    # Rebuild policy and opp_policy with the checkpoint's architecture
    policy = PlanetPolicy(
        self_dim=self_feature_dim(), candidate_dim=candidate_feature_dim(),
        global_dim=global_feature_dim(), candidate_count=cfg.env.candidate_count,
        hidden_size=cfg.model.hidden_size, num_heads=cfg.model.num_heads,
        dropout=cfg.model.dropout,
    ).to(gpu0)
    _build_opp_policy()   # rebuild opp_policy to match updated cfg exactly
    policy.load_state_dict(ck['policy'])
    start_update = ck.get('update', 0)
    print(f"Resumed from {RESUME_CKPT} at update {start_update}")
else:
    print("Starting fresh")

n_params = sum(p.numel() for p in policy.parameters())
optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.ppo.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=TOTAL_UPDATES, eta_min=cfg.ppo.lr * 0.1)

print(f"Policy: {n_params:,} params | hidden={cfg.model.hidden_size}")
print(f"Rollout: {cfg.ppo.rollout_steps} steps | Batch: {cfg.ppo.batch_size} | Envs: {NUM_ENVS}")
print(f"Curriculum: random(0-{RANDOM_END}) -> v11({RANDOM_END}-{SELFPLAY_START}) -> self-play")

# ── Opponents ─────────────────────────────────────────────────────────────────
rand_opp = RandomOpponent()
v11_opp  = V11Opponent(main_path=f'{REPO}/main.py')
sp_opp   = SelfPlayOpp()

def make_envs(opp):
    return [OrbitWarsEnv(cfg, opp, env_index=i) for i in range(NUM_ENVS)]

# ── Rollout ───────────────────────────────────────────────────────────────────
def collect_rollout(envs, buf):
    ep_rews, ep_wins = [], []
    active = [0.0] * len(envs)
    obs_list = [e.reset() for e in envs]
    policy.eval()
    steps = 0
    while steps < cfg.ppo.rollout_steps:
        for ei, (env, batch) in enumerate(zip(envs, obs_list)):
            if batch.self_features.shape[0] == 0:
                obs_list[ei], r, done, info = env.step([])
                active[ei] += r
                if done:
                    ep_rews.append(active[ei]); active[ei] = 0.0
                    ep_wins.append(1.0 if info.get('reward', 0) > 0 else 0.0)
                    obs_list[ei] = env.reset()
                continue
            with torch.inference_mode():
                out = policy(
                    torch.from_numpy(batch.self_features).to(gpu0),
                    torch.from_numpy(batch.candidate_features).to(gpu0),
                    torch.from_numpy(batch.global_features).to(gpu0),
                    torch.from_numpy(batch.candidate_mask).to(gpu0),
                )
                sam = sample_actions(out, deterministic=False)
            idxs = sam.target_index.cpu().numpy()
            lps  = sam.log_prob.cpu().numpy()
            vals = out.values.cpu().numpy()
            moves = []
            for i, ctx in enumerate(batch.contexts):
                idx = int(idxs[i])
                if idx == 0 or not ctx.candidate_mask[idx]: continue
                ships = int(ctx.ship_counts[idx])
                if ships <= 0: continue
                moves.append([ctx.source_id, float(ctx.target_angles[idx]), ships])
            next_obs, r, done, info = env.step(moves)
            active[ei] += r
            for i, ctx in enumerate(batch.contexts):
                buf.push(Transition(
                    self_feat=batch.self_features[i],
                    cand_feat=batch.candidate_features[i],
                    global_feat=batch.global_features[i],
                    cand_mask=batch.candidate_mask[i],
                    action=int(idxs[i]), log_prob=float(lps[i]),
                    value=float(vals[i]), reward=r, done=done,
                ))
            if done:
                ep_rews.append(active[ei]); active[ei] = 0.0
                ep_wins.append(1.0 if info.get('reward', 0) > 0 else 0.0)
                obs_list[ei] = env.reset()
            else:
                obs_list[ei] = next_obs
            steps += 1
            if steps >= cfg.ppo.rollout_steps: break
    policy.train()
    return (float(np.mean(ep_rews)) if ep_rews else 0.0,
            float(np.mean(ep_wins)) if ep_wins else 0.0,
            len(ep_rews))

# ── Training loop ─────────────────────────────────────────────────────────────
buf = RolloutBuffer()
rew_win = deque(maxlen=100); win_win = deque(maxlen=100)
t0 = time.time()
prev_phase = None
envs = make_envs(rand_opp)

print(f"\n=== Training: {start_update} -> {TOTAL_UPDATES} ===\n")

for upd in range(start_update, TOTAL_UPDATES):
    if   upd < RANDOM_END:     phase, opp = 'random',   rand_opp
    elif upd < SELFPLAY_START: phase, opp = 'v11',       v11_opp
    else:                      phase, opp = 'self-play', sp_opp

    if phase != prev_phase:
        if phase == 'self-play': sync_opp()
        envs = make_envs(opp)
        prev_phase = phase
        print(f'\n[{upd}] Phase -> {phase}\n')

    if phase == 'self-play' and (upd - SELFPLAY_START) % SELFPLAY_SYNC == 0:
        sync_opp()

    buf.clear()
    r, w, n_ep = collect_rollout(envs, buf)
    rew_win.append(r); win_win.append(w)

    if len(buf) > 0:
        stats = ppo_update(policy, optimizer, buf, cfg.ppo, gpu0)

    scheduler.step()

    if (upd + 1) % LOG_EVERY == 0:
        print(f"[{upd+1:5d}/{TOTAL_UPDATES}] "
              f"rew={np.mean(rew_win):+.4f} win={np.mean(win_win):.1%} "
              f"ent={stats.entropy:.3f} clip={stats.clip_frac:.2f} "
              f"eps={n_ep} phase={phase} t={time.time()-t0:.0f}s")

    if (upd + 1) % CKPT_EVERY == 0:
        path = CKPT_DIR / f'ckpt_{upd+1:06d}.pt'
        torch.save({'update': upd+1, 'policy': policy.state_dict(), 'cfg': cfg}, path)
        # Auto-backup to output root
        shutil.copy(str(path), f'/kaggle/working/latest_ckpt.pt')
        print(f'  -> Saved {path.name} (backed up to /kaggle/working/latest_ckpt.pt)')

# Final
final = CKPT_DIR / 'ckpt_final.pt'
torch.save({'update': TOTAL_UPDATES, 'policy': policy.state_dict(), 'cfg': cfg}, final)
shutil.copy(str(final), '/kaggle/working/ckpt_final.pt')
print(f'\nDone! t={( time.time()-t0)/3600:.2f}h | Final -> /kaggle/working/ckpt_final.pt')

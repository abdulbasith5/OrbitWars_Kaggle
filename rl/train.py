"""
train.py — Elite PPO training loop v2 for Orbit Wars.

3-Phase Curriculum:
  Phase 1 (0 → --random-end):     vs random (fast warm-up)
  Phase 2 (--random-end → --selfplay-start): vs v11 heuristic (learn real strategy)
  Phase 3 (--selfplay-start → end): self-play (continuous improvement)

Run:
    # Local quick test (CPU)
    python train.py --updates 100 --envs 2 --opponent v11

    # Full elite training (GPU recommended)
    python train.py --updates 5000 --envs 8 --opponent v11 --device cuda

    # Resume from checkpoint
    python train.py --resume checkpoints/ckpt_001000.pt --updates 5000

    # Kaggle Notebook (GPU P100 ~6hrs for 5000 updates)
    python train.py --updates 5000 --envs 8 --device cuda --opponent v11
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import TrainConfig, default_config
from src.env import OrbitWarsEnv
from src.features import (
    candidate_feature_dim, encode_turn,
    global_feature_dim, self_feature_dim,
)
from src.opponents import RandomOpponent, SelfPlayOpponent, build_opponent
from src.policy import PlanetPolicy, sample_actions
from src.ppo import RolloutBuffer, Transition, ppo_update


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Orbit Wars Elite PPO agent")
    p.add_argument("--updates",        type=int,   default=5000,     help="Total PPO updates")
    p.add_argument("--envs",           type=int,   default=4,        help="Parallel envs")
    p.add_argument("--opponent",       type=str,   default="v11",    help="random | v11 | self")
    p.add_argument("--device",         type=str,   default="auto",   help="auto | cpu | cuda")
    p.add_argument("--resume",         type=str,   default=None,     help="Path to checkpoint .pt")
    p.add_argument("--ckpt-every",     type=int,   default=100,      help="Save every N updates")
    p.add_argument("--log-every",      type=int,   default=10,       help="Log every N updates")
    p.add_argument("--random-end",     type=int,   default=200,      help="End random phase")
    p.add_argument("--selfplay-start", type=int,   default=1000,     help="Start self-play phase")
    p.add_argument("--no-shaping",     action="store_true",          help="Disable dense reward shaping")
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Rollout Collection ────────────────────────────────────────────────────────

def collect_rollout(
    envs: list[OrbitWarsEnv],
    policy: PlanetPolicy,
    buffer: RolloutBuffer,
    cfg: TrainConfig,
    device: torch.device,
    rollout_steps: int,
) -> dict:
    ep_rewards  = []
    ep_lengths  = []
    ep_wins     = []
    active_rews = [0.0] * len(envs)
    active_lens = [0]   * len(envs)
    active_term = [0.0] * len(envs)   # terminal reward tracker
    obs_list    = [env.reset(seed=None) for env in envs]

    policy.eval()
    steps_taken = 0

    while steps_taken < rollout_steps:
        for env_idx, (env, batch) in enumerate(zip(envs, obs_list)):
            if batch.self_features.shape[0] == 0:
                obs_list[env_idx], rew, done, info = env.step([])
                active_rews[env_idx] += rew
                active_lens[env_idx] += 1
                if done:
                    ep_rewards.append(active_rews[env_idx])
                    ep_lengths.append(active_lens[env_idx])
                    ep_wins.append(1.0 if info.get("reward", 0) > 0 else 0.0)
                    active_rews[env_idx] = 0.0
                    active_lens[env_idx] = 0
                    obs_list[env_idx] = env.reset()
                continue

            # Forward pass
            with torch.inference_mode():
                sf  = torch.from_numpy(batch.self_features).to(device)
                cf  = torch.from_numpy(batch.candidate_features).to(device)
                gf  = torch.from_numpy(batch.global_features).to(device)
                cm  = torch.from_numpy(batch.candidate_mask).to(device).bool()
                out = policy(sf, cf, gf, cm)
                sampled = sample_actions(out, deterministic=False)

            indices = sampled.target_index.cpu().numpy()
            lps     = sampled.log_prob.cpu().numpy()
            vals    = out.values.cpu().numpy()

            moves = []
            for i, ctx in enumerate(batch.contexts):
                idx = int(indices[i])
                if idx == 0 or idx >= len(ctx.candidate_ids): continue
                if not ctx.candidate_mask[idx]: continue
                ships = int(ctx.ship_counts[idx])
                if ships <= 0: continue
                moves.append([ctx.source_id, float(ctx.target_angles[idx]), ships])

            next_batch, rew, done, info = env.step(moves)
            active_rews[env_idx] += rew
            active_lens[env_idx] += 1

            for i, ctx in enumerate(batch.contexts):
                buffer.push(Transition(
                    self_feat   = batch.self_features[i],
                    cand_feat   = batch.candidate_features[i],
                    global_feat = batch.global_features[i],
                    cand_mask   = batch.candidate_mask[i],
                    action      = int(indices[i]),
                    log_prob    = float(lps[i]),
                    value       = float(vals[i]),
                    reward      = rew,
                    done        = done,
                ))

            if done:
                ep_rewards.append(active_rews[env_idx])
                ep_lengths.append(active_lens[env_idx])
                ep_wins.append(1.0 if info.get("reward", 0) > 0 else 0.0)
                active_rews[env_idx] = 0.0
                active_lens[env_idx] = 0
                obs_list[env_idx] = env.reset()
            else:
                obs_list[env_idx] = next_batch

            steps_taken += 1
            if steps_taken >= rollout_steps:
                break

    policy.train()
    return {
        "ep_reward_mean": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
        "ep_reward_min":  float(np.min(ep_rewards))  if ep_rewards else 0.0,
        "ep_reward_max":  float(np.max(ep_rewards))  if ep_rewards else 0.0,
        "win_rate":       float(np.mean(ep_wins))    if ep_wins    else 0.0,
        "n_episodes":     len(ep_rewards),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = resolve_device(args.device)
    seed_everything(args.seed)

    cfg = default_config()
    cfg.total_updates    = args.updates
    cfg.num_envs         = args.envs
    cfg.checkpoint_every = args.ckpt_every
    cfg.log_every        = args.log_every
    cfg.random_end       = args.random_end
    cfg.selfplay_start   = args.selfplay_start
    cfg.use_shaping_reward = not args.no_shaping

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    print(f"=== Orbit Wars Elite PPO Training ===")
    print(f"Device: {device}")
    # ── Pre-load checkpoint config (so policy is built with correct size) ────────
    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        ckpt_cfg = resume_ckpt.get("cfg", None)
        if ckpt_cfg is not None:
            try:
                cfg.model.hidden_size = ckpt_cfg.model.hidden_size
                cfg.model.num_heads   = ckpt_cfg.model.num_heads
                cfg.model.dropout     = ckpt_cfg.model.dropout
                print(f"Checkpoint config: hidden={cfg.model.hidden_size}, heads={cfg.model.num_heads}")
            except Exception as e:
                print(f"Could not read checkpoint config ({e}), using CLI config")

    print(f"Model: hidden={cfg.model.hidden_size}, heads={cfg.model.num_heads}")
    print(f"Features: self={self_feature_dim()}, cand={candidate_feature_dim()}, global={global_feature_dim()}")
    print(f"Curriculum: random(0-{cfg.random_end}) -> {args.opponent}({cfg.random_end}-{cfg.selfplay_start}) -> self-play")
    print(f"Shaping reward: {cfg.use_shaping_reward}")
    print(f"Rollout: {cfg.ppo.rollout_steps} steps, {cfg.num_envs} envs, {cfg.total_updates} updates")
    print()

    # ── Policy ────────────────────────────────────────────────────────────────
    policy = PlanetPolicy(
        self_dim        = self_feature_dim(),
        candidate_dim   = candidate_feature_dim(),
        global_dim      = global_feature_dim(),
        candidate_count = cfg.env.candidate_count,
        hidden_size     = cfg.model.hidden_size,
        num_heads       = cfg.model.num_heads,
        dropout         = cfg.model.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=cfg.ppo.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.total_updates, eta_min=cfg.ppo.lr * 0.1)

    start_upd = 0
    if resume_ckpt is not None:
        policy.load_state_dict(resume_ckpt["policy"])
        if "optimizer" in resume_ckpt:
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer"])
            except Exception:
                print("Optimizer state incompatible, starting optimizer fresh")
        if "scheduler" in resume_ckpt:
            try:
                scheduler.load_state_dict(resume_ckpt["scheduler"])
            except Exception:
                pass
        start_upd = resume_ckpt.get("update", 0)
        print(f"Resumed from {args.resume} at update {start_upd}")

    # ── Opponent schedule ─────────────────────────────────────────────────────
    random_opp   = RandomOpponent()
    mid_opp      = build_opponent(args.opponent, cfg=cfg, device=device)
    selfplay_opp = SelfPlayOpponent(cfg, device=device)

    def get_opponent(update: int):
        if update < cfg.random_end:
            return random_opp, "random"
        elif update < cfg.selfplay_start:
            return mid_opp, args.opponent
        return selfplay_opp, "self-play"

    def make_envs(opp):
        return [OrbitWarsEnv(cfg, opp, env_index=i) for i in range(cfg.num_envs)]

    prev_phase = None
    envs = make_envs(random_opp)
    buffer = RolloutBuffer()

    reward_window  = deque(maxlen=100)
    win_window     = deque(maxlen=100)
    t0 = time.time()

    # ── Training Loop ─────────────────────────────────────────────────────────
    for update in range(start_upd, cfg.total_updates):
        opp, phase = get_opponent(update)

        # Rebuild envs on phase change
        if phase != prev_phase:
            if phase == "self-play":
                selfplay_opp.sync_from(policy)
            envs = make_envs(opp)
            prev_phase = phase
            print(f"\n[Update {update}] Phase switch -> {phase}\n")

        # Sync self-play opponent periodically
        if (phase == "self-play" and
                (update - cfg.selfplay_start) % cfg.selfplay_sync_every == 0):
            selfplay_opp.sync_from(policy)

        # Collect rollout
        buffer.clear()
        ep_stats = collect_rollout(
            envs, policy, buffer, cfg, device,
            rollout_steps=cfg.ppo.rollout_steps,
        )
        reward_window.extend([ep_stats["ep_reward_mean"]])
        win_window.extend([ep_stats["win_rate"]])

        # PPO update
        if len(buffer) > 0:
            ppo_stats = ppo_update(policy, optimizer, buffer, cfg.ppo, device)
        else:
            continue

        scheduler.step()

        # Logging
        if (update + 1) % cfg.log_every == 0:
            elapsed  = time.time() - t0
            rew_avg  = float(np.mean(reward_window))  if reward_window else 0.0
            win_avg  = float(np.mean(win_window))     if win_window    else 0.0
            lr_now   = scheduler.get_last_lr()[0]
            print(
                f"[{update+1:5d}/{cfg.total_updates}] "
                f"rew={rew_avg:+.4f}  "
                f"win={win_avg:.2%}  "
                f"pol={ppo_stats.policy_loss:+.4f}  "
                f"val={ppo_stats.value_loss:.4f}  "
                f"ent={ppo_stats.entropy:.3f}  "
                f"clip={ppo_stats.clip_frac:.2f}  "
                f"eps={ep_stats['n_episodes']}  "
                f"lr={lr_now:.2e}  "
                f"phase={phase}  "
                f"t={elapsed:.0f}s"
            )

        # Checkpoint
        if (update + 1) % cfg.checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"ckpt_{update+1:06d}.pt"
            torch.save({
                "update":    update + 1,
                "policy":    policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "cfg":       cfg,
            }, ckpt_path)
            print(f"  -> Saved {ckpt_path}")

    # Final save
    final_path = ckpt_dir / "ckpt_final.pt"
    torch.save({"update": cfg.total_updates, "policy": policy.state_dict(),
                "cfg": cfg}, final_path)
    print(f"\nTraining complete. Final model: {final_path}")
    print(f"Total time: {(time.time()-t0)/3600:.2f}h")


if __name__ == "__main__":
    main()

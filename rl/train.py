"""
train.py — Main PPO training loop for Orbit Wars.

Run:
    python train.py                        # default settings
    python train.py --updates 2000         # full training
    python train.py --opponent v9          # train vs our heuristic agent
    python train.py --resume checkpoints/ckpt_001000.pt

Progress is printed every --log-every updates.
Checkpoints saved every --ckpt-every updates to ./checkpoints/
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

# Allow running from the rl/ directory
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
    p = argparse.ArgumentParser(description="Train Orbit Wars PPO agent")
    p.add_argument("--updates",    type=int,   default=2000,         help="Total PPO updates")
    p.add_argument("--envs",       type=int,   default=4,            help="Parallel envs")
    p.add_argument("--opponent",   type=str,   default="random",     help="random | self | v9")
    p.add_argument("--device",     type=str,   default="auto",       help="auto | cpu | cuda")
    p.add_argument("--resume",     type=str,   default=None,         help="Path to checkpoint")
    p.add_argument("--ckpt-every", type=int,   default=100,          help="Save every N updates")
    p.add_argument("--log-every",  type=int,   default=10,           help="Log every N updates")
    p.add_argument("--selfplay-start", type=int, default=200,        help="Switch to self-play after N updates")
    p.add_argument("--seed",       type=int,   default=42)
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
    """
    Run all envs for rollout_steps total steps, storing transitions.
    Returns episode stats.
    """
    ep_rewards   = []
    ep_lengths   = []
    active_rews  = [0.0] * len(envs)
    active_lens  = [0]   * len(envs)
    obs_list     = [env.reset(seed=None) for env in envs]

    policy.eval()
    steps_taken = 0

    while steps_taken < rollout_steps:
        for env_idx, (env, batch) in enumerate(zip(envs, obs_list)):
            if batch.self_features.shape[0] == 0:
                # No planets — step with empty action
                obs_list[env_idx], rew, done, _ = env.step([])
                if done:
                    ep_rewards.append(active_rews[env_idx])
                    ep_lengths.append(active_lens[env_idx])
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

            # Build action list
            moves = []
            for i, ctx in enumerate(batch.contexts):
                idx = int(indices[i])
                if idx == 0 or idx >= len(ctx.candidate_ids): continue
                if not ctx.candidate_mask[idx]: continue
                ships = int(ctx.ship_counts[idx])
                if ships <= 0: continue
                moves.append([ctx.source_id, float(ctx.target_angles[idx]), ships])

            # Store transitions (one per source planet)
            next_batch, rew, done, _ = env.step(moves)
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
    cfg.selfplay_start   = args.selfplay_start

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    print(f"Device: {device}")
    print(f"Opponent: {args.opponent}  |  Envs: {cfg.num_envs}  |  Updates: {cfg.total_updates}")
    print(f"Self-play starts at update {cfg.selfplay_start}")
    print()

    # ── Policy ────────────────────────────────────────────────────────────────
    policy = PlanetPolicy(
        self_dim        = self_feature_dim(),
        candidate_dim   = candidate_feature_dim(),
        global_dim      = global_feature_dim(),
        candidate_count = cfg.env.candidate_count,
        hidden_size     = cfg.model.hidden_size,
    ).to(device)

    optimizer  = torch.optim.Adam(policy.parameters(), lr=cfg.ppo.lr)
    start_upd  = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["policy"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_upd = ckpt.get("update", 0)
        print(f"Resumed from {args.resume} at update {start_upd}")

    # ── Opponents ─────────────────────────────────────────────────────────────
    init_opponent = build_opponent(args.opponent, cfg=cfg, device=device)
    selfplay_opp  = SelfPlayOpponent(cfg, device=device)

    def current_opponent(update: int):
        if update < cfg.selfplay_start:
            return init_opponent
        return selfplay_opp

    # ── Environments ──────────────────────────────────────────────────────────
    # We rebuild envs when opponent switches (lazy rebuild)
    def make_envs(opp):
        return [OrbitWarsEnv(cfg, opp, env_index=i) for i in range(cfg.num_envs)]

    envs   = make_envs(init_opponent)
    buffer = RolloutBuffer()

    # ── Tracking ──────────────────────────────────────────────────────────────
    reward_window = deque(maxlen=50)
    t0            = time.time()

    # ── Training Loop ─────────────────────────────────────────────────────────
    for update in range(start_upd, cfg.total_updates):

        # Switch to self-play
        if update == cfg.selfplay_start:
            selfplay_opp.sync_from(policy)
            envs = make_envs(selfplay_opp)
            print(f"\n[Update {update}] Switched to SELF-PLAY\n")

        # Sync self-play opponent periodically
        if (update >= cfg.selfplay_start and
                (update - cfg.selfplay_start) % cfg.selfplay_sync_every == 0):
            selfplay_opp.sync_from(policy)

        # Collect rollout
        buffer.clear()
        ep_stats = collect_rollout(
            envs, policy, buffer, cfg, device,
            rollout_steps=cfg.ppo.rollout_steps,
        )
        reward_window.extend([ep_stats["ep_reward_mean"]])

        # PPO update
        if len(buffer) > 0:
            ppo_stats = ppo_update(policy, optimizer, buffer, cfg.ppo, device)
        else:
            continue

        # Logging
        if (update + 1) % cfg.log_every == 0:
            elapsed = time.time() - t0
            rew_avg = float(np.mean(reward_window)) if reward_window else 0.0
            phase   = "self-play" if update >= cfg.selfplay_start else args.opponent
            print(
                f"[{update+1:4d}/{cfg.total_updates}] "
                f"rew={rew_avg:+.3f}  "
                f"pol={ppo_stats.policy_loss:+.4f}  "
                f"val={ppo_stats.value_loss:.4f}  "
                f"ent={ppo_stats.entropy:.3f}  "
                f"clip={ppo_stats.clip_frac:.2f}  "
                f"eps={ep_stats['n_episodes']}  "
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
                "cfg":       cfg,
            }, ckpt_path)
            print(f"  → Saved {ckpt_path}")

    # Final save
    final_path = ckpt_dir / "ckpt_final.pt"
    torch.save({"update": cfg.total_updates, "policy": policy.state_dict()}, final_path)
    print(f"\nTraining complete. Final model: {final_path}")


if __name__ == "__main__":
    main()

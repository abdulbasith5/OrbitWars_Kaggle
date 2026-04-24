"""
eval.py — Evaluate a trained checkpoint against various opponents.

Usage:
    python eval.py                                       # vs random, untrained
    python eval.py --checkpoint checkpoints/ckpt_002000.pt
    python eval.py --checkpoint checkpoints/ckpt_002000.pt --opponent v9 --games 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import default_config
from src.env import OrbitWarsEnv
from src.features import (
    candidate_feature_dim, encode_turn,
    global_feature_dim, self_feature_dim,
)
from src.opponents import V9Opponent, build_opponent
from src.policy import PlanetPolicy, sample_actions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--opponent",   type=str, default="random")
    p.add_argument("--games",      type=int, default=10)
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--seed",       type=int, default=0)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    cfg    = default_config()

    policy = PlanetPolicy(
        self_dim        = self_feature_dim(),
        candidate_dim   = candidate_feature_dim(),
        global_dim      = global_feature_dim(),
        candidate_count = cfg.env.candidate_count,
        hidden_size     = cfg.model.hidden_size,
    ).to(device)
    policy.eval()

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt.get("policy", ckpt))
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint — evaluating random policy")

    opponent = build_opponent(args.opponent, cfg=cfg, device=device)
    env      = OrbitWarsEnv(cfg, opponent, env_index=0)

    wins = losses = draws = 0
    for game in range(args.games):
        obs  = env.reset(seed=args.seed + game)
        done = False
        while not done:
            if obs.self_features.shape[0] == 0:
                obs, rew, done, _ = env.step([])
                continue
            with torch.inference_mode():
                out  = policy(
                    torch.from_numpy(obs.self_features).to(device),
                    torch.from_numpy(obs.candidate_features).to(device),
                    torch.from_numpy(obs.global_features).to(device),
                    torch.from_numpy(obs.candidate_mask).to(device).bool(),
                )
                sampled = sample_actions(out, deterministic=True)
            indices = sampled.target_index.cpu().numpy()
            moves   = []
            for i, ctx in enumerate(obs.contexts):
                idx = int(indices[i])
                if idx == 0 or idx >= len(ctx.candidate_ids): continue
                if not ctx.candidate_mask[idx]: continue
                ships = int(ctx.ship_counts[idx])
                if ships <= 0: continue
                moves.append([ctx.source_id, float(ctx.target_angles[idx]), ships])
            obs, rew, done, info = env.step(moves)

        if rew > 0:   wins   += 1
        elif rew < 0: losses += 1
        else:         draws  += 1
        print(f"  Game {game+1}: {'WIN' if rew>0 else 'LOSS' if rew<0 else 'DRAW'} (reward={rew:.1f})")

    total = wins + losses + draws
    print(f"\nResults vs {args.opponent}: {wins}W / {losses}L / {draws}D")
    print(f"Win rate: {100*wins/total:.1f}%")


if __name__ == "__main__":
    main()

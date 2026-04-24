"""
opponents.py — Opponent agent wrappers for training.
  - RandomOpponent  : calls the built-in random agent
  - SelfPlayOpponent: runs a frozen copy of our own policy
  - V9Opponent      : loads and runs main.py (our v9 heuristic)
"""
from __future__ import annotations

import sys
import math
import importlib.util
from pathlib import Path
from typing import Any

import torch
import numpy as np


class RandomOpponent:
    """Wraps kaggle's built-in random agent."""
    def act(self, obs: Any) -> list:
        # Return empty — the kaggle env handles "random" internally
        # but we need to act for the side we control
        return self._random_moves(obs)

    def _random_moves(self, obs: Any) -> list:
        import random
        if isinstance(obs, dict):
            player  = obs.get("player", 0)
            planets = obs.get("planets", [])
        else:
            player  = getattr(obs, "player", 0)
            planets = getattr(obs, "planets", [])
        moves = []
        for p in planets:
            if p[1] != player: continue  # not mine
            ships = p[5]
            if ships < 10: continue
            send  = random.randint(5, int(ships // 2))
            angle = random.uniform(-math.pi, math.pi)
            moves.append([p[0], angle, send])
        return moves[:2]  # max 2 random moves


class SelfPlayOpponent:
    """Frozen copy of PlanetPolicy used as self-play opponent."""
    def __init__(self, cfg, device: torch.device):
        from .policy import PlanetPolicy
        from .features import self_feature_dim, candidate_feature_dim, global_feature_dim
        self.cfg    = cfg
        self.device = device
        self.policy = PlanetPolicy(
            self_dim      = self_feature_dim(),
            candidate_dim = candidate_feature_dim(),
            global_dim    = global_feature_dim(),
            candidate_count = cfg.env.candidate_count,
            hidden_size   = cfg.model.hidden_size,
        ).to(device)
        self.policy.eval()

    def sync_from(self, source_policy) -> None:
        self.policy.load_state_dict(source_policy.state_dict())
        self.policy.eval()

    def act(self, obs: Any) -> list:
        from .features import encode_turn
        from .policy import sample_actions
        batch = encode_turn(obs, self.cfg.env, env_index=0)
        if batch.self_features.shape[0] == 0:
            return []
        with torch.inference_mode():
            out = self.policy(
                torch.from_numpy(batch.self_features).to(self.device),
                torch.from_numpy(batch.candidate_features).to(self.device),
                torch.from_numpy(batch.global_features).to(self.device),
                torch.from_numpy(batch.candidate_mask).to(self.device).bool(),
            )
            sampled = sample_actions(out, deterministic=True)
        indices = sampled.target_index.cpu().numpy()
        moves   = []
        for i, ctx in enumerate(batch.contexts):
            idx = int(indices[i])
            if idx == 0 or idx >= len(ctx.candidate_ids): continue
            if not ctx.candidate_mask[idx]: continue
            ships = int(ctx.ship_counts[idx])
            if ships <= 0: continue
            moves.append([ctx.source_id, float(ctx.target_angles[idx]), ships])
        return moves


class V9Opponent:
    """Runs the v9 heuristic agent from main.py as the opponent."""
    def __init__(self, main_path: str = "../main.py"):
        path = Path(main_path).resolve()
        spec = importlib.util.spec_from_file_location("v9_agent", path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._agent = mod.agent

    def act(self, obs: Any) -> list:
        try:
            return list(self._agent(obs)) or []
        except Exception:
            return []


def build_opponent(name: str, cfg=None, device=None):
    if name == "random":
        return RandomOpponent()
    if name == "self":
        return SelfPlayOpponent(cfg, device)
    if name == "v9":
        return V9Opponent()
    raise ValueError(f"Unknown opponent: {name}")

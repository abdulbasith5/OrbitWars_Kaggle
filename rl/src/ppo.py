"""
ppo.py — PPO (Proximal Policy Optimization) update.

Stores transitions, computes GAE advantages,
and updates the policy with the clipped PPO objective.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .config import PPOConfig
from .policy import PlanetPolicy, PolicyOutput, sample_actions


@dataclass
class Transition:
    """One (source-planet, action) decision stored for replay."""
    self_feat: np.ndarray       # [11]
    cand_feat: np.ndarray       # [N, 14]
    global_feat: np.ndarray     # [8]
    cand_mask: np.ndarray       # [N]  bool
    action: int                 # chosen candidate index
    log_prob: float
    value: float
    reward: float               # filled in at episode end
    done: bool


class RolloutBuffer:
    def __init__(self):
        self._buf: list[Transition] = []

    def push(self, t: Transition):
        self._buf.append(t)

    def __len__(self):
        return len(self._buf)

    def clear(self):
        self._buf.clear()

    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
        """GAE advantage estimation."""
        n      = len(self._buf)
        adv    = np.zeros(n, dtype=np.float32)
        ret    = np.zeros(n, dtype=np.float32)
        gae    = 0.0
        next_v = 0.0

        for i in reversed(range(n)):
            t      = self._buf[i]
            delta  = t.reward + gamma * next_v * (1 - t.done) - t.value
            gae    = delta + gamma * gae_lambda * (1 - t.done) * gae
            adv[i] = gae
            ret[i] = gae + t.value
            next_v = t.value

        return ret, adv

    def to_tensors(self, device: torch.device):
        sf  = torch.tensor(np.array([t.self_feat   for t in self._buf]), dtype=torch.float32, device=device)
        cf  = torch.tensor(np.array([t.cand_feat   for t in self._buf]), dtype=torch.float32, device=device)
        gf  = torch.tensor(np.array([t.global_feat for t in self._buf]), dtype=torch.float32, device=device)
        cm  = torch.tensor(np.array([t.cand_mask   for t in self._buf]), dtype=torch.bool,    device=device)
        act = torch.tensor([t.action   for t in self._buf], dtype=torch.long,    device=device)
        lp  = torch.tensor([t.log_prob for t in self._buf], dtype=torch.float32, device=device)
        return sf, cf, gf, cm, act, lp


@dataclass
class PPOStats:
    policy_loss: float = 0.0
    value_loss:  float = 0.0
    entropy:     float = 0.0
    total_loss:  float = 0.0
    clip_frac:   float = 0.0
    n_updates:   int   = 0


def ppo_update(
    policy: PlanetPolicy,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    cfg: PPOConfig,
    device: torch.device,
) -> PPOStats:
    """
    Run cfg.ppo_epochs passes of PPO over the rollout buffer.
    Returns averaged stats.
    """
    returns, advantages = buffer.compute_returns_and_advantages(cfg.gamma, cfg.gae_lambda)
    # Normalize advantages
    adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    ret_t = torch.tensor(returns,    dtype=torch.float32, device=device)

    sf, cf, gf, cm, act, old_lp = buffer.to_tensors(device)
    n = len(buffer)
    stats = PPOStats()

    for _ in range(cfg.ppo_epochs):
        idx   = torch.randperm(n, device=device)
        start = 0
        while start < n:
            batch = idx[start: start + cfg.batch_size]
            start += cfg.batch_size

            out: PolicyOutput = policy(sf[batch], cf[batch], gf[batch], cm[batch])
            new_lp  = out.log_probs.gather(1, act[batch].unsqueeze(1)).squeeze(1)
            ratio   = (new_lp - old_lp[batch]).exp()

            # Clipped surrogate loss
            adv_b   = adv_t[batch]
            surr1   = ratio * adv_b
            surr2   = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_b
            pol_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            val_loss = 0.5 * (out.values - ret_t[batch]).pow(2).mean()

            # Entropy bonus (encourages exploration)
            ent  = out.entropy.mean()
            loss = pol_loss + cfg.value_coef * val_loss - cfg.entropy_coef * ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            # Track stats
            with torch.no_grad():
                clip_frac = ((ratio - 1).abs() > cfg.clip_eps).float().mean().item()
            stats.policy_loss += pol_loss.item()
            stats.value_loss  += val_loss.item()
            stats.entropy     += ent.item()
            stats.total_loss  += loss.item()
            stats.clip_frac   += clip_frac
            stats.n_updates   += 1

    # Average stats
    if stats.n_updates > 0:
        for attr in ("policy_loss", "value_loss", "entropy", "total_loss", "clip_frac"):
            setattr(stats, attr, getattr(stats, attr) / stats.n_updates)

    return stats

"""
policy.py — PlanetPolicy neural network.

Architecture:
  1. Source encoder : MLP([self_features || global_features]) → planet_emb
  2. Candidate encoder: MLP(candidate_features) → cand_emb  (per candidate)
  3. Dot-product attention : planet_emb · cand_emb → logits over N candidates
  4. Masked softmax → action probabilities
  5. Value head : MLP(planet_emb.mean) → scalar baseline (for PPO)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class PolicyOutput:
    logits: torch.Tensor       # [B, N]  raw logits
    log_probs: torch.Tensor    # [B, N]  log-softmax
    values: torch.Tensor       # [B]     value estimate
    entropy: torch.Tensor      # [B]     per-source entropy


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, layers: int = 2):
        super().__init__()
        dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        mods = []
        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                mods.append(nn.LayerNorm(dims[i+1]))
                mods.append(nn.ReLU())
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class PlanetPolicy(nn.Module):
    """
    For each source planet, picks one of N candidate targets (or "do nothing").
    Shares the candidate encoder across all candidates for efficiency.
    """

    def __init__(
        self,
        self_dim: int,
        candidate_dim: int,
        global_dim: int,
        candidate_count: int,
        hidden_size: int = 256,
    ):
        super().__init__()
        self.candidate_count = candidate_count

        # Source encoder: combine self + global features
        self.src_encoder = MLP(self_dim + global_dim, hidden_size, hidden_size)

        # Candidate encoder: shared across all N candidates
        self.cand_encoder = MLP(candidate_dim, hidden_size, hidden_size)

        # Attention projection (optional learnable scale)
        self.attn_scale = nn.Parameter(torch.tensor(hidden_size ** -0.5))

        # Value head — estimates expected future return
        self.value_head = MLP(hidden_size, hidden_size // 2, 1, layers=2)

    def forward(
        self,
        self_feat: torch.Tensor,      # [B, self_dim]
        cand_feat: torch.Tensor,      # [B, N, cand_dim]
        global_feat: torch.Tensor,    # [B, global_dim]
        cand_mask: torch.Tensor,      # [B, N]  True = valid action
    ) -> PolicyOutput:
        B, N, _ = cand_feat.shape

        # 1. Encode source planet
        src_input  = torch.cat([self_feat, global_feat], dim=-1)   # [B, self+global]
        planet_emb = self.src_encoder(src_input)                   # [B, H]

        # 2. Encode candidates (flatten → encode → reshape)
        cand_flat  = cand_feat.view(B * N, -1)                    # [B*N, cand_dim]
        cand_emb   = self.cand_encoder(cand_flat).view(B, N, -1)  # [B, N, H]

        # 3. Dot-product attention: planet queries each candidate
        # planet_emb [B, H] → [B, 1, H]
        query  = planet_emb.unsqueeze(1)                          # [B, 1, H]
        logits = (query * cand_emb).sum(dim=-1) * self.attn_scale # [B, N]

        # 4. Mask invalid candidates with -inf before softmax
        logits = logits.masked_fill(~cand_mask, float("-inf"))

        # 5. Log-probs and entropy
        log_probs = F.log_softmax(logits, dim=-1)
        probs     = log_probs.exp()
        entropy   = -(probs * log_probs.clamp(min=-1e9)).sum(dim=-1)  # [B]

        # 6. Value baseline (average over planet embeddings → scalar)
        value = self.value_head(planet_emb).squeeze(-1)            # [B]

        return PolicyOutput(
            logits=logits,
            log_probs=log_probs,
            values=value,
            entropy=entropy,
        )


@dataclass
class SampledActions:
    target_index: torch.Tensor    # [B]  chosen candidate index
    log_prob: torch.Tensor        # [B]  log prob of chosen action


def sample_actions(output: PolicyOutput, deterministic: bool = False) -> SampledActions:
    if deterministic:
        idx = output.logits.argmax(dim=-1)
    else:
        dist = torch.distributions.Categorical(logits=output.logits)
        idx  = dist.sample()
    lp = output.log_probs.gather(1, idx.unsqueeze(1)).squeeze(1)
    return SampledActions(target_index=idx, log_prob=lp)

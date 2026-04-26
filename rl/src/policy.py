"""
policy.py — PlanetPolicy v2 (Elite upgrade).

Architecture upgrades:
  1. Deeper MLP: 3 layers instead of 2, larger hidden size (512 default)
  2. Multi-head attention (4 heads) instead of plain dot-product
  3. Residual connections in src/candidate encoders
  4. Richer value head: 3-layer MLP with dropout
  5. Layer normalization throughout
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


class ResidualMLP(nn.Module):
    """MLP with residual connection when in_dim == out_dim."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        mods = []
        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                mods.append(nn.LayerNorm(dims[i+1]))
                mods.append(nn.GELU())
                mods.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*mods)
        # Residual projection if needed
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.residual(x)


class MultiHeadAttentionScorer(nn.Module):
    """
    Multi-head attention between source planet query and candidate keys.
    Produces a score per candidate.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.scale  = self.head_dim ** -0.5

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        query: [B, H]   (source planet embedding)
        keys:  [B, N, H] (candidate embeddings)
        returns: [B, N] attention scores
        """
        B, N, H = keys.shape
        Q = self.q_proj(query).view(B, self.num_heads, self.head_dim)       # [B, heads, head_dim]
        K = self.k_proj(keys.view(B * N, H)).view(B, N, self.num_heads, self.head_dim)  # [B, N, heads, d]
        # Dot product per head: [B, heads, N]
        scores = torch.einsum("bhd,bnhd->bhn", Q, K) * self.scale          # [B, heads, N]
        # Average across heads
        return scores.mean(dim=1)                                            # [B, N]


class PlanetPolicy(nn.Module):
    """
    v2 Elite PlanetPolicy:
    - Residual MLP encoders (3 layers)
    - Multi-head attention scoring (4 heads)
    - Richer 3-layer value head
    - Larger hidden size (512 recommended)
    """

    def __init__(
        self,
        self_dim: int,
        candidate_dim: int,
        global_dim: int,
        candidate_count: int,
        hidden_size: int = 512,
        num_heads: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.candidate_count = candidate_count

        # Source encoder: self + global → embedding
        self.src_encoder = ResidualMLP(
            self_dim + global_dim, hidden_size, hidden_size, layers=3, dropout=dropout)

        # Candidate encoder: shared across all N candidates
        self.cand_encoder = ResidualMLP(
            candidate_dim, hidden_size, hidden_size, layers=3, dropout=dropout)

        # Multi-head attention scorer
        self.attn = MultiHeadAttentionScorer(hidden_size, num_heads=num_heads)

        # Value head — 3-layer with dropout
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(
        self,
        self_feat: torch.Tensor,      # [B, self_dim]
        cand_feat: torch.Tensor,      # [B, N, cand_dim]
        global_feat: torch.Tensor,    # [B, global_dim]
        cand_mask: torch.Tensor,      # [B, N]  True = valid action
    ) -> PolicyOutput:
        B, N, _ = cand_feat.shape

        # 1. Encode source planet
        src_input  = torch.cat([self_feat, global_feat], dim=-1)
        planet_emb = self.src_encoder(src_input)              # [B, H]

        # 2. Encode candidates (flatten → encode → reshape)
        cand_flat  = cand_feat.view(B * N, -1)
        cand_emb   = self.cand_encoder(cand_flat).view(B, N, -1)  # [B, N, H]

        # 3. Multi-head attention scoring
        logits = self.attn(planet_emb, cand_emb)              # [B, N]

        # 4. Mask invalid candidates
        logits = logits.masked_fill(~cand_mask, float("-inf"))

        # 5. Log-probs and entropy
        log_probs = F.log_softmax(logits, dim=-1)
        probs     = log_probs.exp()
        entropy   = -(probs * log_probs.clamp(min=-1e9)).sum(dim=-1)  # [B]

        # 6. Value baseline
        value = self.value_head(planet_emb).squeeze(-1)        # [B]

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

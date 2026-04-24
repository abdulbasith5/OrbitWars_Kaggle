"""
config.py — Training configuration v2 (Elite upgrade).
All hyperparameters in one place.
"""
from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    candidate_count: int  = 18
    board_size: float     = 100.0
    max_ships: float      = 500.0
    max_production: float = 10.0
    max_planets: int      = 20
    episode_steps: int    = 500    # real game is 500 steps


@dataclass
class ModelConfig:
    hidden_size: int  = 512    # v1 was 256; doubled for richer representations
    num_heads: int    = 4      # multi-head attention heads
    dropout: float    = 0.05   # light dropout for regularization


@dataclass
class PPOConfig:
    lr: float          = 1e-4       # lowered for stable 512-hidden training
    gamma: float       = 0.995      # slightly higher discount (longer games)
    gae_lambda: float  = 0.95
    clip_eps: float    = 0.2
    value_coef: float  = 0.5
    entropy_coef: float = 0.05      # keeps entropy healthy (prevents collapse)
    ppo_epochs: int    = 4
    batch_size: int    = 512        # larger batches for 512-hidden network
    rollout_steps: int = 2048       # ~8 full episodes per rollout
    max_grad_norm: float = 0.5


@dataclass
class TrainConfig:
    env: EnvConfig     = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig     = field(default_factory=PPOConfig)

    # Training schedule
    total_updates: int = 5000       # full elite training
    num_envs: int      = 4          # parallel environments
    device: str        = "auto"

    checkpoint_every: int = 200
    checkpoint_dir: str   = "checkpoints"
    log_every: int        = 10

    # Opponent curriculum:
    #   0 → random_start:     vs random (warm-up)
    #   random_start → v11_start: vs v11 heuristic (learn real strategy)
    #   v11_start → end:      self-play (polish)
    random_end: int          = 200   # switch from random → v11 after 200 updates
    selfplay_start: int      = 1000  # switch from v11 → self-play after 1000 updates
    selfplay_sync_every: int = 50    # sync self-play opponent every N updates

    # Dense reward shaping
    use_shaping_reward: bool = True  # per-step production delta reward

    alternate_player_sides: bool = True


def default_config() -> TrainConfig:
    return TrainConfig()

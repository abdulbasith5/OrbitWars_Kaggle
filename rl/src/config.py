"""
config.py — Training configuration dataclass.
All hyperparameters in one place.
"""
from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    candidate_count: int = 18    # max candidates per source planet
    board_size: float = 100.0
    max_ships: float = 500.0
    max_production: float = 10.0
    max_planets: int = 20
    episode_steps: int = 400     # max steps per game


@dataclass
class ModelConfig:
    hidden_size: int = 256


@dataclass
class PPOConfig:
    lr: float = 2e-4            # v9 was 3e-4; slightly lower for stability
    gamma: float = 0.99          # discount factor
    gae_lambda: float = 0.95     # GAE lambda
    clip_eps: float = 0.2        # PPO clip epsilon
    value_coef: float = 0.5      # value loss coefficient
    entropy_coef: float = 0.05   # v9 was 0.02; raised to PREVENT entropy collapse
                                 # Training showed entropy: 0.148->0.031 (collapsed)
                                 # At ent=0.031 policy uses only 1 of 18 candidates
    ppo_epochs: int = 4          # epochs per update
    batch_size: int = 256        # minibatch size
    rollout_steps: int = 1024    # v9 was 512; ~4 episodes per rollout (was ~2)
    max_grad_norm: float = 0.5


@dataclass
class TrainConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    # Training
    total_updates: int = 2000
    num_envs: int = 4            # parallel environments
    device: str = "auto"         # "auto", "cpu", "cuda"
    checkpoint_every: int = 100  # save model every N updates
    checkpoint_dir: str = "checkpoints"
    log_every: int = 10          # print stats every N updates

    # Opponent schedule
    # "random"  = always play vs random
    # "self"    = always self-play
    # "v9"      = train against our heuristic agent (recommended)
    # Switch after this many updates:
    selfplay_start: int = 300    # v9 was 200; give more random warm-up before self-play
    selfplay_sync_every: int = 50  # sync opponent weights every N updates
    alternate_player_sides: bool = True  # alternate who is player 0/1


def default_config() -> TrainConfig:
    return TrainConfig()

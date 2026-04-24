"""
env.py — OrbitWars environment wrapper.
Wraps kaggle_environments to give a clean step/reset interface.
"""
from __future__ import annotations

from typing import Any

from .config import TrainConfig
from .features import TurnBatch, encode_turn


class OrbitWarsEnv:
    def __init__(self, cfg: TrainConfig, opponent, env_index: int = 0):
        self.cfg         = cfg
        self.opponent    = opponent
        self.env_index   = env_index
        self._env        = None
        self._last_obs   = None
        self._last_opp   = None
        self._episode    = 0
        self.learner_idx = 0    # which player slot we are (0 or 1)

    def reset(self, seed: int | None = None) -> TurnBatch:
        from kaggle_environments import make
        cfg_kw: dict[str, Any] = {}
        if seed is not None:
            cfg_kw["seed"] = int(seed)

        # Alternate sides to remove positional bias
        if self.cfg.alternate_player_sides:
            self.learner_idx = (self.env_index + self._episode) % 2
        self._episode += 1

        self._env = make("orbit_wars", configuration=cfg_kw, debug=False)
        self._env.reset(num_agents=2)
        states = self._env.step([[], []])
        self._last_obs = _obs(states[self.learner_idx])
        self._last_opp = _obs(states[1 - self.learner_idx])
        return encode_turn(self._last_obs, self.cfg.env, env_index=self.env_index)

    def step(self, action: list[list]) -> tuple[TurnBatch, float, bool, dict]:
        opp_action = self.opponent.act(self._last_opp)
        if self.learner_idx == 0:
            joint = [action, opp_action]
        else:
            joint = [opp_action, action]

        states = self._env.step(joint)
        p_state = states[self.learner_idx]
        o_state = states[1 - self.learner_idx]

        self._last_obs = _obs(p_state)
        self._last_opp = _obs(o_state)

        done   = _status(p_state) != "ACTIVE"
        reward = _terminal_reward(p_state, o_state) if done else 0.0
        batch  = encode_turn(self._last_obs, self.cfg.env, env_index=self.env_index)
        info   = {"reward": reward, "done": done,
                  "status": _status(p_state)}
        return batch, reward, done, info


def _obs(state):
    if isinstance(state, dict): return state.get("observation")
    return getattr(state, "observation", None)


def _status(state) -> str:
    if isinstance(state, dict): return str(state.get("status", "UNKNOWN"))
    return str(getattr(state, "status", "UNKNOWN"))


def _reward(state) -> float:
    v = state.get("reward", 0.0) if isinstance(state, dict) else getattr(state, "reward", 0.0)
    return 0.0 if v is None else float(v)


def _terminal_reward(p, o) -> float:
    pr, or_ = _reward(p), _reward(o)
    if pr > 0 and or_ > 0: return 0.0  # draw
    return pr

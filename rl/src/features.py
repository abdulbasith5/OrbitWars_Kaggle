"""
features.py — Feature engineering v2 (Elite upgrade).

Expanded feature set for elite RL training:
  self_features      : 18 per source planet  (was 11)
  candidate_features : 22 per candidate      (was 14)
  global_features    : 12                    (was 8)

New signals added:
  - Inbound enemy fleet pressure per planet
  - Inbound friendly fleet reinforcement
  - Production forecast (ships in 50 turns)
  - Contested-zone flag (equidistant from both players)
  - Comet flag
  - Time-to-arrival for each candidate
  - Garrison safety margin
  - Per-step shaping reward: delta_production_owned
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import EnvConfig

# ─── Constants ────────────────────────────────────────────────────────────────

CENTER_X, CENTER_Y = 50.0, 50.0
INNER_ORBIT = 50.0
MAX_SPEED   = 6.0
EPISODE_STEPS = 500


def _dist(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def _is_rotating(x, y, radius):
    orb_r = _dist(x, y, CENTER_X, CENTER_Y)
    return orb_r + radius < INNER_ORBIT


def _crosses_sun(sx, sy, tx, ty, sun_r=10.0, safety=1.5):
    r = sun_r + safety
    dx, dy = tx - sx, ty - sy
    lsq = dx * dx + dy * dy
    if lsq < 1e-9:
        return False
    t = max(0.0, min(1.0, ((CENTER_X - sx) * dx + (CENTER_Y - sy) * dy) / lsq))
    cx = sx + t * dx - CENTER_X
    cy = sy + t * dy - CENTER_Y
    return cx * cx + cy * cy < r * r


def _fleet_speed(ships):
    if ships <= 1:
        return 1.0
    ratio = math.log(max(1, ships)) / math.log(1000.0)
    return 1.0 + (MAX_SPEED - 1.0) * (max(0.0, min(1.0, ratio)) ** 1.5)


def _travel_turns(sx, sy, tx, ty, ships):
    d = _dist(sx, sy, tx, ty)
    return max(1, int(math.ceil(d / _fleet_speed(ships))))


def _fixed_ship_count(src_ships, src_x, src_y, tgt_ships, tgt_x, tgt_y, tgt_prod, tgt_owner, me):
    tt = _travel_turns(src_x, src_y, tgt_x, tgt_y, max(1, int(tgt_ships) + 5))
    if tgt_owner == me:
        return 0
    garrison = tgt_ships + tgt_prod * tt
    needed = int(garrison) + 3
    if needed > src_ships - 5:
        return 0
    return max(1, needed)


# ─── Parse observation ─────────────────────────────────────────────────────────

@dataclass
class PlanetState:
    id: int; owner: int; x: float; y: float
    radius: float; ships: float; production: float


@dataclass
class FleetState:
    id: int; owner: int; x: float; y: float
    angle: float; from_planet_id: int; ships: float


@dataclass
class GameState:
    player: int
    step: int
    planets: list[PlanetState]
    fleets: list[FleetState]
    angular_velocity: float = 0.0
    # Derived — populated by parse_observation
    my_production: float = 0.0
    en_production: float = 0.0
    # Per-planet inbound fleet pressures {planet_id -> (friendly_ships, enemy_ships)}
    inbound: dict = field(default_factory=dict)


def parse_observation(obs: Any) -> GameState:
    if isinstance(obs, dict):
        player = obs.get("player", 0)
        step   = obs.get("step", 0) or 0
        ang_v  = obs.get("angular_velocity", 0.0) or 0.0
        raw_p  = obs.get("planets", []) or []
        raw_f  = obs.get("fleets", []) or []
    else:
        player = getattr(obs, "player", 0)
        step   = getattr(obs, "step", 0) or 0
        ang_v  = getattr(obs, "angular_velocity", 0.0) or 0.0
        raw_p  = getattr(obs, "planets", []) or []
        raw_f  = getattr(obs, "fleets", []) or []

    planets = [PlanetState(*p[:7]) for p in raw_p]
    fleets  = [FleetState(*f[:7]) for f in raw_f]

    # Inbound fleet pressure (approximate by from_planet_id)
    planet_map = {p.id: p for p in planets}
    inbound: dict[int, list[float, float]] = {p.id: [0.0, 0.0] for p in planets}
    for f in fleets:
        # Approximate target: use angle + origin heuristic
        best_pid, best_diff = None, 0.5
        spd = _fleet_speed(f.ships)
        for p in planets:
            d = _dist(f.x, f.y, p.x, p.y)
            if d < 0.1: continue
            t = max(1, int(math.ceil(d / spd)))
            expected = math.atan2(p.y - f.y, p.x - f.x)
            diff = abs(math.atan2(math.sin(f.angle - expected),
                                  math.cos(f.angle - expected)))
            if diff < best_diff:
                best_diff, best_pid = diff, p.id
        if best_pid is not None and best_pid in inbound:
            if f.owner == player:
                inbound[best_pid][0] += f.ships
            else:
                inbound[best_pid][1] += f.ships

    my_prod = sum(p.production for p in planets if p.owner == player)
    en_prod = sum(p.production for p in planets if p.owner not in (-1, player))

    return GameState(
        player=int(player), step=int(step),
        planets=planets, fleets=fleets,
        angular_velocity=float(ang_v),
        my_production=my_prod, en_production=en_prod,
        inbound=inbound,
    )


# ─── Feature dimensions ────────────────────────────────────────────────────────

def self_feature_dim() -> int:    return 18
def candidate_feature_dim() -> int: return 22
def global_feature_dim() -> int:  return 12


def _total_ships(planets):
    return sum(p.ships for p in planets)


# ─── Self features ─────────────────────────────────────────────────────────────

def build_self_features(src: PlanetState, state: GameState, cfg: EnvConfig) -> np.ndarray:
    my_planets    = [p for p in state.planets if p.owner == state.player]
    enemy_planets = [p for p in state.planets if p.owner not in (-1, state.player)]
    neutral_planets = [p for p in state.planets if p.owner == -1]

    inbound_friend = state.inbound.get(src.id, [0.0, 0.0])[0]
    inbound_enemy  = state.inbound.get(src.id, [0.0, 0.0])[1]
    garrison_safety = max(0.0, src.ships - inbound_enemy)  # effective safe ships

    # Nearest enemy distance (normalized)
    if enemy_planets:
        nearest_en_dist = min(_dist(src.x, src.y, e.x, e.y) for e in enemy_planets) / cfg.board_size
    else:
        nearest_en_dist = 1.0

    # Contested: is this planet within 35 units of center?
    d_center = _dist(src.x, src.y, CENTER_X, CENTER_Y) / cfg.board_size

    return np.asarray([
        1.0,                                                          # bias
        src.x / cfg.board_size,
        src.y / cfg.board_size,
        src.radius / 5.0,
        min(src.ships, cfg.max_ships) / cfg.max_ships,
        src.production / cfg.max_production,
        1.0 if _is_rotating(src.x, src.y, src.radius) else 0.0,
        len(my_planets) / cfg.max_planets,
        len(enemy_planets) / cfg.max_planets,
        _total_ships(my_planets)  / (cfg.max_planets * cfg.max_ships),
        _total_ships(enemy_planets) / (cfg.max_planets * cfg.max_ships),
        # NEW v2 features:
        min(inbound_friend, cfg.max_ships) / cfg.max_ships,          # friendly inbound
        min(inbound_enemy, cfg.max_ships) / cfg.max_ships,           # enemy inbound
        min(garrison_safety, cfg.max_ships) / cfg.max_ships,         # safe garrison
        nearest_en_dist,                                              # proximity to enemy
        d_center,                                                     # distance to center
        len(neutral_planets) / cfg.max_planets,                      # neutrals remaining
        (EPISODE_STEPS - state.step) / EPISODE_STEPS,                # time remaining
    ], dtype=np.float32)


# ─── Candidate features ────────────────────────────────────────────────────────

def build_candidate_features(
    src: PlanetState,
    candidates: list[PlanetState],
    state: GameState,
    cfg: EnvConfig,
    comet_ids: set | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int], list[float]]:
    N = cfg.candidate_count
    features   = np.zeros((N, candidate_feature_dim()), dtype=np.float32)
    cand_mask  = np.zeros((N,), dtype=bool)
    ship_counts    = [0] * N
    candidate_ids  = [-1] * N
    target_angles  = [0.0] * N
    cand_mask[0]   = True  # slot 0 = "do nothing"
    if comet_ids is None:
        comet_ids = set()

    for idx, tgt in enumerate(candidates, start=1):
        if idx >= N:
            break
        dx    = tgt.x - src.x
        dy    = tgt.y - src.y
        angle = math.atan2(dy, dx)
        d     = _dist(src.x, src.y, tgt.x, tgt.y)
        blocks = _crosses_sun(src.x, src.y, tgt.x, tgt.y)
        ships  = _fixed_ship_count(
            src.ships, src.x, src.y,
            tgt.ships, tgt.x, tgt.y,
            tgt.production, tgt.owner, state.player
        )
        tt = _travel_turns(src.x, src.y, tgt.x, tgt.y, max(1, ships))

        # Inbound pressure on target
        tgt_inbound_friend = state.inbound.get(tgt.id, [0.0, 0.0])[0]
        tgt_inbound_enemy  = state.inbound.get(tgt.id, [0.0, 0.0])[1]

        # Contested flag
        d_center_tgt = _dist(tgt.x, tgt.y, CENTER_X, CENTER_Y) / cfg.board_size

        # Production forecast: ships at tgt in 50 turns
        prod_forecast = min(tgt.ships + tgt.production * 50, cfg.max_ships) / cfg.max_ships

        features[idx] = np.asarray([
            1.0,                                                             # bias
            1.0 if tgt.owner == -1 else 0.0,                               # is neutral
            1.0 if tgt.owner == state.player else 0.0,                     # is mine
            1.0 if tgt.owner not in (-1, state.player) else 0.0,           # is enemy
            tgt.x / cfg.board_size,
            tgt.y / cfg.board_size,
            dx / cfg.board_size,
            dy / cfg.board_size,
            d / cfg.board_size,                                             # distance
            min(tgt.ships, cfg.max_ships) / cfg.max_ships,
            tgt.production / cfg.max_production,
            1.0 if _is_rotating(tgt.x, tgt.y, tgt.radius) else 0.0,
            1.0 if blocks else 0.0,
            min(src.ships, cfg.max_ships) / cfg.max_ships,
            # NEW v2 features:
            min(tt, 100) / 100.0,                                          # travel time
            min(tgt_inbound_enemy, cfg.max_ships) / cfg.max_ships,         # enemy pressure on tgt
            min(tgt_inbound_friend, cfg.max_ships) / cfg.max_ships,        # friendly inbound to tgt
            prod_forecast,                                                  # production forecast
            d_center_tgt,                                                   # target centrality
            1.0 if tgt.id in comet_ids else 0.0,                          # is comet
            min(ships, cfg.max_ships) / cfg.max_ships,                     # ships to send
            tgt.radius / 5.0,                                              # planet size
        ], dtype=np.float32)

        ship_counts[idx]   = ships
        cand_mask[idx]     = ships > 0 and not blocks
        candidate_ids[idx] = tgt.id
        target_angles[idx] = angle

    return features, cand_mask, ship_counts, candidate_ids, target_angles


# ─── Global features ──────────────────────────────────────────────────────────

def build_global_features(state: GameState, cfg: EnvConfig) -> np.ndarray:
    my_p   = [p for p in state.planets if p.owner == state.player]
    en_p   = [p for p in state.planets if p.owner not in (-1, state.player)]
    neu_p  = [p for p in state.planets if p.owner == -1]
    my_f   = [f for f in state.fleets if f.owner == state.player]
    en_f   = [f for f in state.fleets if f.owner != state.player]
    scale  = cfg.max_planets * cfg.max_ships

    # Production forecasts (50 turns)
    my_forecast = sum(min(p.ships + p.production * 50, cfg.max_ships) for p in my_p)
    en_forecast = sum(min(p.ships + p.production * 50, cfg.max_ships) for p in en_p)
    max_forecast = cfg.max_planets * cfg.max_ships

    return np.asarray([
        state.step / cfg.episode_steps,
        len(my_p)  / cfg.max_planets,
        len(en_p)  / cfg.max_planets,
        len(neu_p) / cfg.max_planets,
        _total_ships(my_p)  / scale,
        _total_ships(en_p)  / scale,
        sum(f.ships for f in my_f) / scale,
        sum(f.ships for f in en_f) / scale,
        # NEW v2:
        state.my_production / (cfg.max_planets * cfg.max_production),   # my production rate
        state.en_production / (cfg.max_planets * cfg.max_production),   # enemy production rate
        min(my_forecast, max_forecast) / max_forecast,                  # my 50-turn forecast
        min(en_forecast, max_forecast) / max_forecast,                  # en 50-turn forecast
    ], dtype=np.float32)


# ─── Candidate selection ──────────────────────────────────────────────────────

def build_candidates(src: PlanetState, state: GameState, cfg: EnvConfig) -> list[PlanetState]:
    others = [p for p in state.planets if p.id != src.id]
    q      = cfg.candidate_count // 3

    enemies   = sorted([p for p in others if p.owner not in (-1, state.player)],
                       key=lambda p: _dist(src.x, src.y, p.x, p.y))[:q]
    neutrals  = sorted([p for p in others if p.owner == -1],
                       key=lambda p: _dist(src.x, src.y, p.x, p.y))[:q]
    friendlies = sorted([p for p in others if p.owner == state.player],
                        key=lambda p: _dist(src.x, src.y, p.x, p.y))[:q]

    selected = set(p.id for p in enemies + neutrals + friendlies)
    cands    = enemies + neutrals + friendlies
    fallback = sorted([p for p in others if p.id not in selected],
                      key=lambda p: _dist(src.x, src.y, p.x, p.y))
    cands.extend(fallback[:cfg.candidate_count - len(cands)])
    return cands[:cfg.candidate_count - 1]


# ─── Shaping reward ───────────────────────────────────────────────────────────

def compute_shaping_reward(
    prev_state: GameState | None,
    curr_state: GameState,
    player: int,
    win_reward: float,
) -> float:
    """
    Dense per-step shaping reward (in addition to terminal win/loss):
      +0.002 per unit of production gained
      -0.002 per unit of production lost
      +0.001 per ship advantage gained
      +0.05  when a high-value planet (prod>=3) is captured
      -0.05  when a high-value planet (prod>=3) is lost
    This replaces the sparse win/lose signal with continuous feedback.
    """
    if win_reward != 0.0:
        return win_reward  # terminal: use actual win/loss signal

    if prev_state is None:
        return 0.0

    prev_my_prod = sum(p.production for p in prev_state.planets if p.owner == player)
    curr_my_prod = sum(p.production for p in curr_state.planets if p.owner == player)
    prod_delta   = curr_my_prod - prev_my_prod

    prev_my_ships = sum(p.ships for p in prev_state.planets if p.owner == player)
    curr_my_ships = sum(p.ships for p in curr_state.planets if p.owner == player)
    prev_en_ships = sum(p.ships for p in prev_state.planets if p.owner not in (-1, player))
    curr_en_ships = sum(p.ships for p in curr_state.planets if p.owner not in (-1, player))
    ship_advantage = (curr_my_ships - curr_en_ships) - (prev_my_ships - prev_en_ships)

    # High-value planet events
    prev_my_hv = {p.id for p in prev_state.planets if p.owner == player and p.production >= 3}
    curr_my_hv = {p.id for p in curr_state.planets if p.owner == player and p.production >= 3}
    hv_gained = len(curr_my_hv - prev_my_hv)
    hv_lost   = len(prev_my_hv - curr_my_hv)

    reward = (
        prod_delta   * 0.002 +
        ship_advantage * 0.0002 +
        hv_gained    * 0.05 +
        hv_lost      * (-0.05)
    )
    return float(np.clip(reward, -0.5, 0.5))


# ─── TurnBatch ────────────────────────────────────────────────────────────────

@dataclass
class DecisionContext:
    env_index: int
    source_id: int
    candidate_ids: list[int]
    candidate_mask: np.ndarray
    ship_counts: list[int]
    target_angles: list[float]


@dataclass
class TurnBatch:
    self_features: np.ndarray       # [B, 18]
    candidate_features: np.ndarray  # [B, N, 22]
    global_features: np.ndarray     # [B, 12]
    candidate_mask: np.ndarray      # [B, N]  bool
    contexts: list[DecisionContext]
    state: GameState


def encode_turn(obs: Any, cfg: EnvConfig, env_index: int = 0,
                comet_ids: set | None = None) -> TurnBatch:
    state = obs if isinstance(obs, GameState) else parse_observation(obs)
    my_planets = sorted([p for p in state.planets if p.owner == state.player],
                        key=lambda p: p.id)
    empty = TurnBatch(
        self_features      = np.zeros((0, self_feature_dim()),              dtype=np.float32),
        candidate_features = np.zeros((0, cfg.candidate_count, candidate_feature_dim()), dtype=np.float32),
        global_features    = np.zeros((0, global_feature_dim()),            dtype=np.float32),
        candidate_mask     = np.zeros((0, cfg.candidate_count),             dtype=bool),
        contexts=[], state=state,
    )
    if not my_planets:
        return empty

    global_feat = build_global_features(state, cfg)
    self_rows   = []
    cand_rows   = []
    mask_rows   = []
    contexts    = []

    for src in my_planets:
        cands = build_candidates(src, state, cfg)
        cf, cm, sc, cids, angles = build_candidate_features(
            src, cands, state, cfg, comet_ids=comet_ids)
        self_rows.append(build_self_features(src, state, cfg))
        cand_rows.append(cf)
        mask_rows.append(cm)
        contexts.append(DecisionContext(
            env_index=env_index, source_id=src.id,
            candidate_ids=cids, candidate_mask=cm,
            ship_counts=sc, target_angles=angles,
        ))

    return TurnBatch(
        self_features      = np.asarray(self_rows, dtype=np.float32),
        candidate_features = np.asarray(cand_rows, dtype=np.float32),
        global_features    = np.repeat(global_feat[None, :], len(self_rows), axis=0),
        candidate_mask     = np.asarray(mask_rows, dtype=bool),
        contexts=contexts, state=state,
    )

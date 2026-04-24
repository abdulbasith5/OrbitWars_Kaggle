"""
features.py — Feature engineering: converts a raw game observation
into tensors the neural network can consume.

Feature dimensions (must match policy.py):
  self_features      : 11 per source planet
  candidate_features : 14 per candidate target
  global_features    : 8  (one per turn, broadcast to all sources)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import EnvConfig

# ─── Game types ───────────────────────────────────────────────────────────────

CENTER_X, CENTER_Y = 50.0, 50.0
INNER_ORBIT = 50.0   # planets with orb_r + radius >= this are static


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
    return 1.0 + 5.0 * (max(0.0, min(1.0, ratio)) ** 1.5)


def _travel_turns(sx, sy, tx, ty, ships):
    d = _dist(sx, sy, tx, ty)
    return max(1, int(math.ceil(d / _fleet_speed(ships))))


def _fixed_ship_count(src_ships, src_x, src_y, tgt_ships, tgt_x, tgt_y, tgt_prod, tgt_owner, me):
    """
    Heuristic ship count used for RL action execution.
    Similar to what v9 does with arrival-time estimation (simplified here
    to keep features stateless — v9 does the real calculation at inference).
    """
    tt = _travel_turns(src_x, src_y, tgt_x, tgt_y, max(1, int(tgt_ships) + 5))
    if tgt_owner == me:
        return 0  # reinforce — skip
    # Neutral or enemy: garrison grows by production over travel time
    garrison = tgt_ships + tgt_prod * tt
    needed = int(garrison) + 3
    if needed > src_ships - 5:
        return 0  # can't afford
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
    return GameState(player=int(player), step=int(step),
                     planets=planets, fleets=fleets,
                     angular_velocity=float(ang_v))


# ─── Feature builders ──────────────────────────────────────────────────────────

def self_feature_dim() -> int:   return 11
def candidate_feature_dim() -> int: return 14
def global_feature_dim() -> int: return 8


def _total_ships(planets):
    return sum(p.ships for p in planets)


def build_self_features(src: PlanetState, state: GameState, cfg: EnvConfig) -> np.ndarray:
    my_planets     = [p for p in state.planets if p.owner == state.player]
    enemy_planets  = [p for p in state.planets if p.owner not in (-1, state.player)]
    return np.asarray([
        1.0,
        src.x / cfg.board_size,
        src.y / cfg.board_size,
        src.radius / 5.0,
        min(src.ships, cfg.max_ships) / cfg.max_ships,
        src.production / cfg.max_production,
        1.0 if _is_rotating(src.x, src.y, src.radius) else 0.0,
        len(my_planets) / cfg.max_planets,
        len(enemy_planets) / cfg.max_planets,
        _total_ships(my_planets) / (cfg.max_planets * cfg.max_ships),
        _total_ships(enemy_planets) / (cfg.max_planets * cfg.max_ships),
    ], dtype=np.float32)


def build_candidate_features(
    src: PlanetState,
    candidates: list[PlanetState],
    state: GameState,
    cfg: EnvConfig,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int], list[float]]:
    N = cfg.candidate_count
    features   = np.zeros((N, candidate_feature_dim()), dtype=np.float32)
    cand_mask  = np.zeros((N,), dtype=bool)
    ship_counts    = [0] * N
    candidate_ids  = [-1] * N
    target_angles  = [0.0] * N
    cand_mask[0]   = True  # slot 0 = "do nothing"

    for idx, tgt in enumerate(candidates, start=1):
        if idx >= N:
            break
        dx    = tgt.x - src.x
        dy    = tgt.y - src.y
        angle = math.atan2(dy, dx)
        blocks = _crosses_sun(src.x, src.y, tgt.x, tgt.y)
        ships  = _fixed_ship_count(
            src.ships, src.x, src.y,
            tgt.ships, tgt.x, tgt.y,
            tgt.production, tgt.owner, state.player
        )
        features[idx] = np.asarray([
            1.0,
            1.0 if tgt.owner == -1 else 0.0,
            1.0 if tgt.owner == state.player else 0.0,
            1.0 if tgt.owner not in (-1, state.player) else 0.0,
            tgt.x / cfg.board_size,
            tgt.y / cfg.board_size,
            dx / cfg.board_size,
            dy / cfg.board_size,
            _dist(src.x, src.y, tgt.x, tgt.y) / cfg.board_size,
            min(tgt.ships, cfg.max_ships) / cfg.max_ships,
            tgt.production / cfg.max_production,
            1.0 if _is_rotating(tgt.x, tgt.y, tgt.radius) else 0.0,
            1.0 if blocks else 0.0,
            min(src.ships, cfg.max_ships) / cfg.max_ships,
        ], dtype=np.float32)

        ship_counts[idx]   = ships
        cand_mask[idx]     = ships > 0 and not blocks
        candidate_ids[idx] = tgt.id
        target_angles[idx] = angle

    return features, cand_mask, ship_counts, candidate_ids, target_angles


def build_global_features(state: GameState, cfg: EnvConfig) -> np.ndarray:
    my_p   = [p for p in state.planets if p.owner == state.player]
    en_p   = [p for p in state.planets if p.owner not in (-1, state.player)]
    neu_p  = [p for p in state.planets if p.owner == -1]
    my_f   = [f for f in state.fleets if f.owner == state.player]
    en_f   = [f for f in state.fleets if f.owner != state.player]
    scale  = cfg.max_planets * cfg.max_ships
    return np.asarray([
        state.step / cfg.episode_steps,
        len(my_p)  / cfg.max_planets,
        len(en_p)  / cfg.max_planets,
        len(neu_p) / cfg.max_planets,
        _total_ships(my_p)  / scale,
        _total_ships(en_p)  / scale,
        sum(f.ships for f in my_f) / scale,
        sum(f.ships for f in en_f) / scale,
    ], dtype=np.float32)


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
    return cands[:cfg.candidate_count - 1]  # -1 for "do nothing" slot


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
    self_features: np.ndarray       # [B, 11]
    candidate_features: np.ndarray  # [B, N, 14]
    global_features: np.ndarray     # [B, 8]
    candidate_mask: np.ndarray      # [B, N]  bool
    contexts: list[DecisionContext]
    state: GameState


def encode_turn(obs: Any, cfg: EnvConfig, env_index: int = 0) -> TurnBatch:
    state = obs if isinstance(obs, GameState) else parse_observation(obs)
    my_planets = sorted([p for p in state.planets if p.owner == state.player],
                        key=lambda p: p.id)
    empty = TurnBatch(
        self_features     = np.zeros((0, self_feature_dim()),            dtype=np.float32),
        candidate_features= np.zeros((0, cfg.candidate_count, candidate_feature_dim()), dtype=np.float32),
        global_features   = np.zeros((0, global_feature_dim()),          dtype=np.float32),
        candidate_mask    = np.zeros((0, cfg.candidate_count),           dtype=bool),
        contexts=[], state=state,
    )
    if not my_planets:
        return empty

    global_feat    = build_global_features(state, cfg)
    self_rows      = []
    cand_rows      = []
    mask_rows      = []
    contexts       = []

    for src in my_planets:
        cands = build_candidates(src, state, cfg)
        cf, cm, sc, cids, angles = build_candidate_features(src, cands, state, cfg)
        self_rows.append(build_self_features(src, state, cfg))
        cand_rows.append(cf)
        mask_rows.append(cm)
        contexts.append(DecisionContext(
            env_index=env_index, source_id=src.id,
            candidate_ids=cids, candidate_mask=cm,
            ship_counts=sc, target_angles=angles,
        ))

    return TurnBatch(
        self_features     = np.asarray(self_rows, dtype=np.float32),
        candidate_features= np.asarray(cand_rows, dtype=np.float32),
        global_features   = np.repeat(global_feat[None,:], len(self_rows), axis=0),
        candidate_mask    = np.asarray(mask_rows, dtype=bool),
        contexts=contexts, state=state,
    )

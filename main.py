"""
Orbit Wars Agent — v7 "Mechanics-Correct"
=====================================================
Critical formula fixes from dylanxue04 mechanics deep-dive notebook:

  BUG FIXES (v6 → v7):
  1. SUN RADIUS: was 10.0, CORRECT value is 5.0
     → Our path-clear checks were blocking many valid fleet routes!
  2. FLEET SPEED: was log-based approx, CORRECT formula is:
       speed = min(1.0 + ships / 10.0, 6.0)
     → 10 ships = speed 2.0, 50+ ships = max speed 6.0
     → Small fleets (1 ship) move at speed 1.0 (slow snipe risk)
  3. FLEET-VS-FLEET COMBAT AWARENESS:
     Enemy fleets meeting our fleets cancel each other out.
     We now model this when estimating garrison defenses.
  4. PRODUCTION CAP: planets cap at 1,000 ships — no need to
     send reinforcements to near-cap planets.
  5. PRODUCTION FORMULA: ships_per_turn = radius * 0.2
     (already correct in our scoring, confirmed)

  Retained from v6:
  - Dual-track neutral expansion (always runs)
  - Timed convergence (farthest fires first)
  - Sentinel reserve (rear planets keep 6+ ships)
  - Parallel expansion (each source → different neutral)
"""

import json
import math
import os
from collections import namedtuple

# ─── Constants (verified from mechanics notebook) ────────────────────────────
CENTER_X, CENTER_Y = 50, 50
SUN_R       = 5.0     # FIXED: was 10.0 — notebook confirms sun radius = 5.0
MAX_SPEED   = 6.0
SHIP_CAP    = 1000    # planets cap at 1,000 ships
INNER_ORBIT = 45.0
ANGLE_TOL   = 0.32

PHASE_BLITZ = 40
PHASE_MID   = 80

# ─── Tunable Defaults ────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    # Reserve
    "min_reserve":                  5.0,
    "reserve_prod_mult":            1.2,
    "reserve_threat_bonus":        12.0,
    "reserve_floor":                6.0,   # ↑ raised: prevent snipes (Sentinel)
    "reserve_base":                 5.0,
    "reserve_production_mult":      1.2,
    "reserve_threat_cap":          18.0,
    "reserve_threat_mult":          2.0,
    "reserve_high_ship_threshold": 40.0,
    "reserve_high_ship_bonus":     -2.0,
    "reserve_low_ship_threshold":  20.0,
    "reserve_low_ship_bonus":       2.0,
    "reserve_comet_bonus":          6.0,
    # Scoring
    "score_production_mult":       14.0,
    "score_neutral_bonus":         15.0,   # ↑ neutrals are even more valuable
    "score_enemy_bonus":           22.0,
    "score_enemy_prod_bonus":       4.0,
    "score_distance_penalty":       0.4,
    "score_amount_penalty":         0.8,
    "score_comet_bonus":           30.0,
    "score_comet_penalty":          6.0,
    # Gates
    "attack_min_score":             0.3,   # ↓ more aggressive neutral grabs
    "attack_ratio":                 0.80,
    # Early blitz
    "early_proximity_weight":      10.0,
    "early_min_score":              0.1,
    "early_turns":                 40.0,   # ↑ extended blitz phase
    "grab_turns":                  20.0,   # ↑ pure grab mode longer
    # Convergence
    "converge_prod_threshold":      3.0,
    "converge_max_sources":         5,     # ↑ more ships flooding one target
    "max_parallel_neutrals":        8,
    # Counter-attack
    "counter_ratio":                1.3,
    # Defense
    "defense_threat_radius":       35.0,
    "defense_scramble_ratio":       0.6,
    "threat_radius_source":        20.0,
    "threat_radius_planet":        25.0,
    # Logistics
    "logistics_surplus_min":       15.0,   # ↓ redistribute sooner
    "logistics_ratio":              0.40,
    "defense_transfer_ratio":       0.35,
    "defense_transfer_prod_mult":   2.0,
}

# ─── Data Structures ─────────────────────────────────────────────────────────
Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])
Fleet  = namedtuple("Fleet",  ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"])


def _load_params():
    raw = os.environ.get("ORBIT_WARS_PARAMS", "")
    if raw:
        try:
            cfg = dict(DEFAULT_CONFIG)
            cfg.update(json.loads(raw))
            return cfg
        except Exception:
            pass
    return dict(DEFAULT_CONFIG)


def _unpack_obs(obs):
    if isinstance(obs, dict):
        player           = obs["player"]
        raw_planets      = obs["planets"]
        raw_fleets       = obs.get("fleets", [])
        angular_velocity = obs["angular_velocity"]
        raw_comets       = obs.get("comets", [])
        step             = obs.get("step", 0)
    else:
        player           = obs.player
        raw_planets      = obs.planets
        raw_fleets       = obs.fleets
        angular_velocity = obs.angular_velocity
        raw_comets       = getattr(obs, "comets", [])
        step             = getattr(obs, "step", 0)
    planets = [Planet(*p) for p in raw_planets]
    fleets  = [Fleet(*f)  for f in raw_fleets]
    comets  = []
    for c in raw_comets:
        try:
            comets.append(Planet(*c[:7]))
        except Exception:
            pass
    return player, planets, fleets, comets, step, angular_velocity


# ─── Physics ─────────────────────────────────────────────────────────────────

def dist(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def get_fleet_speed(ships):
    """
    CORRECTED formula from mechanics notebook:
      speed = min(1.0 + ships / 10.0, 6.0)
    Examples: 1 ship=1.1, 10 ships=2.0, 50 ships=6.0 (max)
    """
    return min(1.0 + ships / 10.0, MAX_SPEED)


def is_path_clear(x1, y1, x2, y2, safety=1.5):
    r = SUN_R + safety
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-9:
        return True
    t  = max(0.0, min(1.0, ((CENTER_X - x1) * dx + (CENTER_Y - y1) * dy) / len_sq))
    cx = x1 + t * dx - CENTER_X
    cy = y1 + t * dy - CENTER_Y
    return (cx * cx + cy * cy) >= r * r


def predict_planet_pos(planet, angular_velocity, turns):
    orb_r = dist(planet.x, planet.y, CENTER_X, CENTER_Y)
    if orb_r > INNER_ORBIT:
        return planet.x, planet.y
    ang = math.atan2(planet.y - CENTER_Y, planet.x - CENTER_X) + angular_velocity * turns
    return CENTER_X + orb_r * math.cos(ang), CENTER_Y + orb_r * math.sin(ang)


def travel_turns(sx, sy, tx, ty, ships):
    d = dist(sx, sy, tx, ty)
    return max(1, int(math.ceil(d / get_fleet_speed(ships))))


def intercept(src, tgt, ships, ang_vel, passes=4):
    tx, ty = tgt.x, tgt.y
    t = 1
    for _ in range(passes):
        t  = travel_turns(src.x, src.y, tx, ty, ships)
        tx, ty = predict_planet_pos(tgt, ang_vel, t)
    return tx, ty, t


# ─── Forward Model ───────────────────────────────────────────────────────────

def _infer_target(fleet, planets, ang_vel):
    best_pid  = None
    best_diff = ANGLE_TOL
    speed = get_fleet_speed(fleet.ships)
    for p in planets:
        d = dist(fleet.x, fleet.y, p.x, p.y)
        if d < 0.1:
            continue
        t = max(1, int(math.ceil(d / speed)))
        px, py = predict_planet_pos(p, ang_vel, t)
        expected = math.atan2(py - fleet.y, px - fleet.x)
        diff = abs(math.atan2(
            math.sin(fleet.angle - expected),
            math.cos(fleet.angle - expected)
        ))
        if diff < best_diff:
            best_diff, best_pid = diff, p.id
    return best_pid


def _build_fleet_adj(planets, fleets, ang_vel, me):
    adj = {p.id: 0 for p in planets}
    for f in fleets:
        tid = _infer_target(f, planets, ang_vel)
        if tid is None or tid not in adj:
            continue
        adj[tid] += f.ships if f.owner == me else -f.ships
    return adj


def _predict_garrison(planet, fleet_adj, turns):
    """Garrison capped at SHIP_CAP (1000) — planets cannot exceed this."""
    prod = planet.production * turns if planet.owner != -1 else 0
    raw  = planet.ships + prod + fleet_adj.get(planet.id, 0)
    return max(0, min(raw, SHIP_CAP))


# ─── Reserve ─────────────────────────────────────────────────────────────────

def _compute_reserve(src, threatened, phase, cfg, is_rear=False):
    """
    Sentinel logic: rear planets keep 6+ ships to prevent snipes.
    Frontier planets stay lean for max offensive pressure.
    """
    floor = cfg["reserve_floor"]  # 6 — Sentinel minimum
    if phase <= PHASE_BLITZ:
        r = floor + 0.5 * src.production
    elif is_rear:
        r = floor + cfg["reserve_prod_mult"] * src.production
    else:
        r = max(floor - 2, 3) + 0.8 * src.production  # lean on frontier
    if threatened:
        r += cfg["reserve_threat_bonus"]
    return max(int(floor), int(r))


def _classify_rear(my_planets, enemy_planets):
    """Return set of planet IDs that are 'rear' (far from all enemies)."""
    if not enemy_planets:
        return set()
    avg_dist = {}
    for p in my_planets:
        avg_dist[p.id] = min(dist(p.x, p.y, e.x, e.y) for e in enemy_planets)
    if not avg_dist:
        return set()
    threshold = sorted(avg_dist.values())[len(avg_dist) // 2]  # median distance
    return {pid for pid, d in avg_dist.items() if d >= threshold}


# ─── Emergency Defense ───────────────────────────────────────────────────────

def _emergency_defense(my_planets, all_planets, fleets, me, ang_vel, cfg):
    actions   = []
    committed = {}
    planet_map  = {p.id: p for p in all_planets}
    my_ids      = {p.id for p in my_planets}
    enemy_fleets = [f for f in fleets if f.owner != me]

    for fleet in enemy_fleets:
        tid = _infer_target(fleet, my_planets, ang_vel)
        if tid is None or tid not in my_ids:
            continue
        target   = planet_map[tid]
        f_d      = dist(fleet.x, fleet.y, target.x, target.y)
        f_eta    = max(1, int(math.ceil(f_d / get_fleet_speed(fleet.ships))))
        garrison = target.ships + target.production * f_eta
        if garrison >= fleet.ships:
            continue
        shortfall = fleet.ships - garrison + 5
        for src in sorted(my_planets, key=lambda p: dist(p.x, p.y, target.x, target.y)):
            if src.id == tid:
                continue
            already   = committed.get(src.id, 0)
            available = src.ships - already - int(cfg["min_reserve"])
            if available < shortfall:
                continue
            tx, ty = predict_planet_pos(target, ang_vel, f_eta)
            if not is_path_clear(src.x, src.y, tx, ty):
                continue
            angle = math.atan2(ty - src.y, tx - src.x)
            actions.append([src.id, angle, int(shortfall)])
            committed[src.id] = committed.get(src.id, 0) + int(shortfall)
            break

    return actions, committed


# ─── DUAL-TRACK PASS 1: Neutral Expansion (Always Runs) ─────────────────────

def _neutral_expansion(my_planets, neutrals, comets, all_planets, fleets,
                       ang_vel, me, phase, cfg, def_committed):
    """
    PATTERN "Neutral Race": always grab unclaimed planets, even during combat.
    Each source planet fires at a DIFFERENT neutral (parallel expansion).
    In grab phase (0-20): pure nearest-neutral logic, no score gate.
    In blitz phase (20-40): scored but with ultra-low gate.
    """
    actions           = []
    atk_committed     = {}
    claimed_neutrals  = set()   # neutrals targeted this turn (parallel expansion)
    comet_ids         = {c.id for c in comets}
    fleet_adj         = _build_fleet_adj(all_planets, fleets, ang_vel, me)
    is_grab           = phase < int(cfg["grab_turns"])
    is_blitz          = phase < PHASE_BLITZ

    all_neutral_targets = neutrals + comets

    for src in sorted(my_planets, key=lambda p: -p.ships):
        reserve   = _compute_reserve(src, False, phase, cfg)
        committed = def_committed.get(src.id, 0) + atk_committed.get(src.id, 0)
        available = src.ships - reserve - committed
        if available < 2:
            continue

        # Filter to unclaimed neutrals only
        unclaimed = [t for t in all_neutral_targets if t.id not in claimed_neutrals]
        if not unclaimed:
            break

        best_action = None

        if is_grab:
            # Pure grab: fire at nearest affordable neutral
            candidates = []
            for t in unclaimed:
                tx, ty, tt = intercept(src, t, available, ang_vel)
                if not is_path_clear(src.x, src.y, tx, ty):
                    continue
                garrison = _predict_garrison(t, fleet_adj, tt)
                needed   = int(garrison) + 2
                if 1 <= needed <= available:
                    candidates.append((dist(src.x, src.y, t.x, t.y), t.id, tx, ty, needed))
            if candidates:
                _, tid, tx, ty, needed = min(candidates)
                angle = math.atan2(ty - src.y, tx - src.x)
                actions.append([src.id, angle, needed])
                atk_committed[src.id] = atk_committed.get(src.id, 0) + needed
                claimed_neutrals.add(tid)
        else:
            # Blitz: scored with ultra-low gate (0.1)
            gate = cfg["early_min_score"] if is_blitz else cfg["attack_min_score"]
            best_score = gate
            for t in unclaimed:
                tx, ty, tt = intercept(src, t, available, ang_vel)
                if not is_path_clear(src.x, src.y, tx, ty):
                    continue
                garrison = _predict_garrison(t, fleet_adj, tt)
                needed   = int(garrison) + 2
                if needed < 1: needed = 1
                if needed > available:
                    continue
                is_comet = t.id in comet_ids
                cost = max(1, needed)
                base  = cfg["score_production_mult"] * t.production
                base += cfg["score_neutral_bonus"]
                base += cfg["score_comet_bonus"] if is_comet else 0
                score = base / (cfg["score_distance_penalty"] * tt +
                                cfg["score_amount_penalty"] * cost + 1.0)
                if score > best_score:
                    best_score  = score
                    best_action = (t.id, tx, ty, needed)
            if best_action:
                tid, tx, ty, cost = best_action
                angle = math.atan2(ty - src.y, tx - src.x)
                actions.append([src.id, angle, cost])
                atk_committed[src.id] = atk_committed.get(src.id, 0) + cost
                claimed_neutrals.add(tid)

    return actions, atk_committed


# ─── DUAL-TRACK PASS 2: Enemy Attack with Timed Convergence ─────────────────

def _enemy_attack(my_planets, enemy_planets, all_planets, fleets,
                  ang_vel, me, phase, cfg, def_committed, neu_committed):
    """
    PATTERN "Flood" + "Timed Convergence":
    - Score enemy planets by production ROI
    - For high-value targets, send from MULTIPLE sources (convergence)
    - Sort sources by distance DESC so farthest fires first → all land together
    """
    if not enemy_planets:
        return [], {}

    actions       = []
    atk_committed = dict(neu_committed)  # inherit neutrals committed
    converge_map  = {}  # target_id → list of (src, needed, tx, ty, tt)
    fleet_adj     = _build_fleet_adj(all_planets, fleets, ang_vel, me)

    my_ids         = {p.id for p in my_planets}
    threatened_ids = set()
    for f in fleets:
        if f.owner == me:
            continue
        tid = _infer_target(f, my_planets, ang_vel)
        if tid and tid in my_ids:
            threatened_ids.add(tid)

    rear_ids = _classify_rear(my_planets, enemy_planets)

    # Score each enemy planet
    scored_targets = []
    for tgt in enemy_planets:
        # Estimate a rough cost to capture
        rough_garrison = tgt.ships
        base  = cfg["score_production_mult"] * tgt.production
        base += cfg["score_enemy_bonus"]
        base += cfg["score_enemy_prod_bonus"] * tgt.production
        score = base / (cfg["score_amount_penalty"] * max(1, rough_garrison) + 1.0)
        scored_targets.append((score, tgt))
    scored_targets.sort(key=lambda x: -x[0])

    # Process top targets for convergence
    max_srcs = int(cfg["converge_max_sources"])
    for _, tgt in scored_targets[:5]:
        eligible = []
        for src in my_planets:
            reserve   = _compute_reserve(src, src.id in threatened_ids, phase, cfg,
                                         is_rear=(src.id in rear_ids))
            committed = def_committed.get(src.id, 0) + atk_committed.get(src.id, 0)
            available = src.ships - reserve - committed
            if available < 3:
                continue
            tx, ty, tt = intercept(src, tgt, available, ang_vel)
            if not is_path_clear(src.x, src.y, tx, ty):
                continue
            garrison = _predict_garrison(tgt, fleet_adj, tt)
            needed   = int(garrison) + 2
            if needed < 1: needed = 1
            if needed > available:
                continue
            d_to_tgt = dist(src.x, src.y, tgt.x, tgt.y)
            eligible.append((d_to_tgt, src, needed, tx, ty, tt))

        if not eligible:
            continue

        # TIMED CONVERGENCE: sort farthest first → simultaneous landing
        eligible.sort(key=lambda x: -x[0])

        srcs_used = 0
        total_committed = 0
        for d_to_tgt, src, needed, tx, ty, tt in eligible[:max_srcs]:
            if srcs_used >= max_srcs:
                break
            reserve   = _compute_reserve(src, src.id in threatened_ids, phase, cfg,
                                         is_rear=(src.id in rear_ids))
            committed = def_committed.get(src.id, 0) + atk_committed.get(src.id, 0)
            available = src.ships - reserve - committed
            actual    = min(needed, available)
            if actual < 1:
                continue

            # Only send if we're actually contributing to capture
            remaining = (tgt.ships + 2) - total_committed
            if remaining <= 0:
                break

            send = min(actual, remaining + 2)
            if send < 1:
                continue

            angle = math.atan2(ty - src.y, tx - src.x)
            actions.append([src.id, angle, send])
            atk_committed[src.id] = atk_committed.get(src.id, 0) + send
            total_committed += send
            srcs_used += 1

            if total_committed >= tgt.ships + 2:
                break  # enough ships committed to this target

    return actions, atk_committed


# ─── Counter-Attack (Opportunistic Snipe) ────────────────────────────────────

def _counter_attack(my_planets, enemy_planets, all_planets, fleets,
                    ang_vel, me, phase, def_committed, atk_committed, cfg):
    """Snipe lightly-defended enemy planets to slow their production."""
    if phase < PHASE_BLITZ or not enemy_planets:
        return []

    actions   = []
    fleet_adj = _build_fleet_adj(all_planets, fleets, ang_vel, me)
    attacked  = set(atk_committed.keys())
    rear_ids  = _classify_rear(my_planets, enemy_planets)

    weak_enemies = sorted(enemy_planets,
                          key=lambda p: p.ships / max(1, p.production))

    for tgt in weak_enemies[:3]:
        for src in sorted(my_planets,
                          key=lambda p: dist(p.x, p.y, tgt.x, tgt.y)):
            reserve   = _compute_reserve(src, False, phase, cfg,
                                         is_rear=(src.id in rear_ids))
            committed = def_committed.get(src.id, 0) + atk_committed.get(src.id, 0)
            available = src.ships - reserve - committed
            if available < 5:
                continue
            tx, ty, tt = intercept(src, tgt, available, ang_vel)
            if not is_path_clear(src.x, src.y, tx, ty):
                continue
            garrison = _predict_garrison(tgt, fleet_adj, tt)
            needed   = int(garrison) + 2
            if needed < 1: needed = 1
            if needed <= available and available >= needed * cfg["counter_ratio"]:
                angle = math.atan2(ty - src.y, tx - src.x)
                actions.append([src.id, angle, needed])
                atk_committed[src.id] = atk_committed.get(src.id, 0) + needed
                attacked.add(tgt.id)
                break

    return actions


# ─── Logistics ───────────────────────────────────────────────────────────────

def _logistics(my_planets, enemy_planets, cfg):
    if len(my_planets) < 2 or not enemy_planets:
        return []

    def front_score(p):
        return min(dist(p.x, p.y, e.x, e.y) for e in enemy_planets)

    sorted_mine = sorted(my_planets, key=front_score, reverse=True)
    front = sorted_mine[-1]

    for src in sorted_mine[:-1]:
        reserve = int(cfg["min_reserve"] + cfg["reserve_prod_mult"] * src.production)
        surplus = src.ships - reserve
        if surplus < cfg["logistics_surplus_min"]:
            continue
        send = int(surplus * cfg["logistics_ratio"])
        if send < 5:
            continue
        if is_path_clear(src.x, src.y, front.x, front.y):
            angle = math.atan2(front.y - src.y, front.x - src.x)
            return [[src.id, angle, send]]
    return []


# ─── Main Agent ──────────────────────────────────────────────────────────────

def agent(obs, config=None):
    """
    Orbit Wars agent — v6 Convergence.
    Pipeline:
      1. Emergency defense     (always)
      2. Neutral expansion     (always — dual-track, parallel, never blocked)
      3. Enemy attack          (timed convergence, farthest fires first)
      4. Counter-attack snipe  (mid/late game opportunistic)
      5. Logistics             (quiet turns only)
    """
    me, planets, fleets, comets, step, ang_vel = _unpack_obs(obs)
    cfg = _load_params()

    my_planets    = [p for p in planets if p.owner == me]
    enemy_planets = [p for p in planets if p.owner not in (-1, me)]
    neutrals      = [p for p in planets if p.owner == -1]

    if not my_planets:
        return []

    # Pass 1 — Emergency Defense
    def_actions, def_committed = _emergency_defense(
        my_planets, planets, fleets, me, ang_vel, cfg
    )

    # Pass 2 — Neutral Expansion (dual-track: always runs regardless of enemies)
    neu_actions, neu_committed = _neutral_expansion(
        my_planets, neutrals, comets, planets, fleets,
        ang_vel, me, step, cfg, def_committed
    )

    # Pass 3 — Enemy Attack with Timed Convergence
    atk_actions, atk_committed = _enemy_attack(
        my_planets, enemy_planets, planets, fleets,
        ang_vel, me, step, cfg, def_committed, neu_committed
    )

    # Pass 4 — Opportunistic Counter-Attack
    counter_actions = _counter_attack(
        my_planets, enemy_planets, planets, fleets,
        ang_vel, me, step, def_committed, atk_committed, cfg
    )

    # Pass 5 — Logistics (only when not defending)
    log_actions = _logistics(my_planets, enemy_planets, cfg) if not def_actions else []

    return def_actions + neu_actions + atk_actions + counter_actions + log_actions
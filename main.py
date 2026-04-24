"""
Orbit Wars Agent — v9 "Arrival-Time"
=====================================================
KEY UPGRADE: Arrival-time simulation.
  Instead of snapshot garrison, we simulate exactly what
  the garrison will be when OUR fleet lands, accounting for:
    - Planet production over travel time
    - All in-flight friendly and enemy fleets landing before us
    - Same-turn combat (top two attackers cancel, survivor hits garrison)

This single change is the biggest skill gap vs top agents.

Physics constants confirmed from structured-baseline notebook:
  SUN_R = 10.0, fleet_speed = log-based, INNER_ORBIT = 50.0
  Static planet: orb_r + planet.radius >= INNER_ORBIT
"""

import json
import math
import os
from collections import namedtuple

# ─── Constants ────────────────────────────────────────────────────────────────
CENTER_X, CENTER_Y = 50.0, 50.0
SUN_R        = 10.0
SUN_SAFETY   = 1.5
MAX_SPEED    = 6.0
SHIP_CAP     = 1000
INNER_ORBIT  = 50.0
ANGLE_TOL    = 0.32
LAUNCH_CLEARANCE = 0.1

PHASE_BLITZ  = 40
PHASE_MID    = 80

# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "min_reserve": 5.0, "reserve_prod_mult": 1.2,
    "reserve_threat_bonus": 12.0, "reserve_floor": 6.0,
    "score_production_mult": 14.0, "score_neutral_bonus": 15.0,
    "score_enemy_bonus": 22.0, "score_enemy_prod_bonus": 4.0,
    "score_distance_penalty": 0.4, "score_amount_penalty": 0.8,
    "score_comet_bonus": 30.0,
    "attack_min_score": 0.3, "attack_ratio": 0.80,
    "early_min_score": 0.1, "early_turns": 40.0, "grab_turns": 20.0,
    "converge_max_sources": 5, "max_parallel_neutrals": 8,
    "counter_ratio": 1.3,
    "defense_threat_radius": 35.0, "threat_radius_planet": 25.0,
    "logistics_surplus_min": 15.0, "logistics_ratio": 0.40,
    "defense_transfer_ratio": 0.35,
    # Arrival-time
    "arrival_sim_horizon": 80,   # max turns ahead to simulate
    "margin_neutral": 2,         # extra ships to send vs neutral
    "margin_enemy": 3,           # extra ships to send vs enemy
}

Planet = namedtuple("Planet", ["id","owner","x","y","radius","ships","production"])
Fleet  = namedtuple("Fleet",  ["id","owner","x","y","angle","from_planet_id","ships"])


def _load_params():
    raw = os.environ.get("ORBIT_WARS_PARAMS", "")
    if raw:
        try:
            cfg = dict(DEFAULT_CONFIG); cfg.update(json.loads(raw)); return cfg
        except Exception: pass
    return dict(DEFAULT_CONFIG)


def _unpack_obs(obs):
    if isinstance(obs, dict):
        player = obs["player"]; raw_planets = obs["planets"]
        raw_fleets = obs.get("fleets", []); angular_velocity = obs["angular_velocity"]
        raw_comets = obs.get("comets", []); step = obs.get("step", 0) or 0
        comet_ids = set(obs.get("comet_planet_ids", []) or [])
    else:
        player = obs.player; raw_planets = obs.planets
        raw_fleets = obs.fleets; angular_velocity = obs.angular_velocity
        raw_comets = getattr(obs, "comets", []); step = getattr(obs, "step", 0) or 0
        comet_ids = set(getattr(obs, "comet_planet_ids", []) or [])
    planets = [Planet(*p) for p in raw_planets]
    fleets  = [Fleet(*f)  for f in raw_fleets]
    comets  = []
    for c in raw_comets:
        try: comets.append(Planet(*c[:7]))
        except Exception: pass
    return player, planets, fleets, comets, comet_ids, step, angular_velocity


# ─── Physics ──────────────────────────────────────────────────────────────────

def dist(x1, y1, x2, y2): return math.hypot(x2-x1, y2-y1)


def get_fleet_speed(ships):
    if ships <= 1: return 1.0
    ratio = math.log(max(1, ships)) / math.log(1000.0)
    return 1.0 + (MAX_SPEED - 1.0) * (max(0.0, min(1.0, ratio)) ** 1.5)


def is_path_clear(x1, y1, x2, y2):
    r = SUN_R + SUN_SAFETY
    dx, dy = x2-x1, y2-y1
    lsq = dx*dx + dy*dy
    if lsq < 1e-9: return True
    t = max(0.0, min(1.0, ((CENTER_X-x1)*dx + (CENTER_Y-y1)*dy) / lsq))
    cx = x1+t*dx-CENTER_X; cy = y1+t*dy-CENTER_Y
    return cx*cx + cy*cy >= r*r


def predict_planet_pos(planet, ang_vel, turns):
    orb_r = dist(planet.x, planet.y, CENTER_X, CENTER_Y)
    if orb_r + planet.radius >= INNER_ORBIT: return planet.x, planet.y
    ang = math.atan2(planet.y-CENTER_Y, planet.x-CENTER_X) + ang_vel*turns
    return CENTER_X + orb_r*math.cos(ang), CENTER_Y + orb_r*math.sin(ang)


def travel_turns(sx, sy, tx, ty, ships):
    return max(1, int(math.ceil(dist(sx,sy,tx,ty) / get_fleet_speed(ships))))


def intercept(src, tgt, ships, ang_vel, passes=4):
    tx, ty = tgt.x, tgt.y
    t = 1
    for _ in range(passes):
        t = travel_turns(src.x, src.y, tx, ty, ships)
        tx, ty = predict_planet_pos(tgt, ang_vel, t)
    return tx, ty, t


# ─── Arrival-Time Simulation ──────────────────────────────────────────────────

def _infer_target(fleet, planets, ang_vel):
    best_pid = None; best_diff = ANGLE_TOL
    speed = get_fleet_speed(fleet.ships)
    for p in planets:
        d = dist(fleet.x, fleet.y, p.x, p.y)
        if d < 0.1: continue
        t = max(1, int(math.ceil(d / speed)))
        px, py = predict_planet_pos(p, ang_vel, t)
        expected = math.atan2(py-fleet.y, px-fleet.x)
        diff = abs(math.atan2(math.sin(fleet.angle-expected),
                              math.cos(fleet.angle-expected)))
        if diff < best_diff: best_diff, best_pid = diff, p.id
    return best_pid


def _build_fleet_arrivals(planets, fleets, ang_vel):
    """
    Returns dict[planet_id -> sorted list of (eta, owner, ships)].
    Each entry is a fleet currently in-flight, predicted to land at that planet.
    """
    arrivals = {p.id: [] for p in planets}
    planet_map = {p.id: p for p in planets}
    for f in fleets:
        tid = _infer_target(f, planets, ang_vel)
        if tid is None or tid not in arrivals: continue
        tgt = planet_map[tid]
        d = dist(f.x, f.y, tgt.x, tgt.y)
        eta = max(1, int(math.ceil(d / get_fleet_speed(f.ships))))
        arrivals[tid].append((eta, f.owner, int(f.ships)))
    for pid in arrivals:
        arrivals[pid].sort()
    return arrivals


def _simulate_garrison_at(planet, arrival_turn, fleet_arrivals):
    """
    Simulate garrison and owner at `arrival_turn`, processing all in-flight
    fleets from fleet_arrivals that land on or before arrival_turn.

    Same-turn combat (from structured-baseline notebook):
      1. Group arrivals by owner on the same turn
      2. Add friendlies to garrison first
      3. Top two attackers cancel each other (their fleets fight)
      4. Survivor fights the garrison
    Returns: (owner_at_arrival, garrison_at_arrival)
    """
    owner    = planet.owner
    garrison = float(planet.ships)
    events   = [(t,o,s) for t,o,s in fleet_arrivals.get(planet.id,[])
                if t <= arrival_turn]

    cur = 0
    i   = 0
    while i < len(events):
        t = events[i][0]
        # Produce ships between current turn and this event
        if owner != -1:
            garrison = min(garrison + planet.production * (t - cur), SHIP_CAP)
        cur = t

        # Collect all fleets arriving on the same turn
        same = []
        while i < len(events) and events[i][0] == t:
            same.append(events[i]); i += 1

        # Group by owner
        by_owner = {}
        for _, eo, es in same:
            by_owner[eo] = by_owner.get(eo, 0) + es

        # Friendly reinforcements land first
        if owner in by_owner:
            garrison = min(garrison + by_owner.pop(owner), SHIP_CAP)

        # Attackers fight each other (top two cancel)
        if len(by_owner) >= 2:
            top = sorted(by_owner.items(), key=lambda x: -x[1])
            winner_o, winner_s = top[0]
            loser_s            = top[1][1]
            remain = winner_s - loser_s
            by_owner = {winner_o: remain} if remain > 0 else {}

        # Survivor attacks garrison
        if by_owner:
            att_o, att_s = next(iter(by_owner.items()))
            garrison -= att_s
            if garrison < 0:
                owner    = att_o
                garrison = -garrison

    # Produce for remaining turns after last event
    if owner != -1:
        garrison = min(garrison + planet.production * (arrival_turn - cur), SHIP_CAP)

    return owner, max(0.0, garrison)


# ─── Phase Detection ──────────────────────────────────────────────────────────

def _detect_phase(step, planets, my_planets, enemy_planets):
    total = len(planets)
    if total == 0: return 1 if step >= PHASE_BLITZ else 0
    neutral_frac = sum(1 for p in planets if p.owner == -1) / total
    if neutral_frac > 0.40: return 0
    elif neutral_frac > 0.10: return 1
    else: return 2


# ─── Reserve ──────────────────────────────────────────────────────────────────

def _classify_rear(my_planets, enemy_planets):
    if not enemy_planets: return set()
    avg_dist = {p.id: min(dist(p.x,p.y,e.x,e.y) for e in enemy_planets)
                for p in my_planets}
    threshold = sorted(avg_dist.values())[len(avg_dist)//2]
    return {pid for pid,d in avg_dist.items() if d >= threshold}


def _compute_reserve(src, threatened, phase, cfg, is_rear=False):
    floor = cfg["reserve_floor"]
    if phase == 0:   r = floor + 0.5 * src.production
    elif is_rear:    r = floor + cfg["reserve_prod_mult"] * src.production
    else:            r = max(floor-2, 3) + 0.8 * src.production
    if threatened:   r += cfg["reserve_threat_bonus"]
    return max(int(floor), int(r))


# ─── Emergency Defense ────────────────────────────────────────────────────────

def _emergency_defense(my_planets, all_planets, fleets, me, ang_vel, cfg, fleet_arrivals):
    actions = {}; committed = {}
    planet_map = {p.id: p for p in all_planets}
    my_ids = {p.id for p in my_planets}
    for f in fleets:
        if f.owner == me: continue
        tid = _infer_target(f, my_planets, ang_vel)
        if tid is None or tid not in my_ids: continue
        tgt = planet_map[tid]
        d = dist(f.x, f.y, tgt.x, tgt.y)
        eta = max(1, int(math.ceil(d / get_fleet_speed(f.ships))))
        # Simulate garrison at fleet arrival
        _, garrison_on_arrival = _simulate_garrison_at(tgt, eta, fleet_arrivals)
        if garrison_on_arrival >= f.ships: continue
        shortfall = int(f.ships - garrison_on_arrival) + 5
        for src in sorted(my_planets, key=lambda p: dist(p.x,p.y,tgt.x,tgt.y)):
            if src.id == tid: continue
            available = src.ships - committed.get(src.id, 0) - int(cfg["min_reserve"])
            if available < shortfall: continue
            tx, ty = predict_planet_pos(tgt, ang_vel, eta)
            if not is_path_clear(src.x, src.y, tx, ty): continue
            angle = math.atan2(ty-src.y, tx-src.x)
            if src.id not in actions: actions[src.id] = []
            actions[src.id].append([src.id, angle, int(shortfall)])
            committed[src.id] = committed.get(src.id, 0) + int(shortfall)
            break
    return [m for ms in actions.values() for m in ms], committed


# ─── Neutral Expansion (Dual-Track, Always Runs) ──────────────────────────────

def _neutral_expansion(my_planets, neutrals, comets, all_planets, fleets,
                       ang_vel, me, phase, cfg, def_committed, fleet_arrivals):
    actions = []; atk_committed = {}; claimed = set()
    all_targets = neutrals + comets
    is_grab  = phase == 0
    is_blitz = phase <= 1

    for src in sorted(my_planets, key=lambda p: -p.ships):
        reserve   = _compute_reserve(src, False, phase, cfg)
        committed = def_committed.get(src.id, 0) + atk_committed.get(src.id, 0)
        available = src.ships - reserve - committed
        if available < 2: continue

        unclaimed = [t for t in all_targets if t.id not in claimed]
        if not unclaimed: break

        best_action = None; best_score = -1e9

        for t in unclaimed:
            tx, ty, tt = intercept(src, t, available, ang_vel)
            if not is_path_clear(src.x, src.y, tx, ty): continue

            # ARRIVAL-TIME: what is the garrison when we arrive?
            arr_owner, arr_garrison = _simulate_garrison_at(t, tt, fleet_arrivals)
            if arr_owner == me: continue  # we'll already own it — skip
            margin = cfg["margin_neutral"]
            needed = int(arr_garrison) + margin
            if needed < 1: needed = 1
            if needed > available: continue

            if is_grab:
                # Pure grab: nearest affordable
                score = -dist(src.x, src.y, t.x, t.y)
            else:
                gate = cfg["early_min_score"] if is_blitz else cfg["attack_min_score"]
                cost = max(1, needed)
                score = (cfg["score_production_mult"] * t.production +
                         cfg["score_neutral_bonus"]) / (
                         cfg["score_distance_penalty"] * tt +
                         cfg["score_amount_penalty"] * cost + 1.0)
                if score < gate: continue

            if score > best_score:
                best_score  = score
                best_action = (t.id, tx, ty, needed)

        if best_action:
            tid, tx, ty, cost = best_action
            angle = math.atan2(ty-src.y, tx-src.x)
            actions.append([src.id, angle, cost])
            atk_committed[src.id] = atk_committed.get(src.id, 0) + cost
            claimed.add(tid)

    return actions, atk_committed


# ─── Enemy Attack with Timed Convergence ──────────────────────────────────────

def _enemy_attack(my_planets, enemy_planets, all_planets, fleets,
                  ang_vel, me, phase, cfg, def_committed, neu_committed, fleet_arrivals):
    if not enemy_planets: return [], {}

    actions = []; atk_committed = dict(neu_committed)
    my_ids = {p.id for p in my_planets}
    threatened_ids = set()
    for f in fleets:
        if f.owner == me: continue
        tid = _infer_target(f, my_planets, ang_vel)
        if tid and tid in my_ids: threatened_ids.add(tid)

    rear_ids = _classify_rear(my_planets, enemy_planets)

    # Score enemy targets by production ROI
    scored = sorted(enemy_planets,
                    key=lambda t: -(cfg["score_production_mult"]*t.production +
                                    cfg["score_enemy_bonus"]))

    max_srcs = int(cfg["converge_max_sources"])

    for tgt in scored[:5]:
        eligible = []
        for src in my_planets:
            reserve   = _compute_reserve(src, src.id in threatened_ids, phase, cfg,
                                         is_rear=(src.id in rear_ids))
            committed = def_committed.get(src.id,0) + atk_committed.get(src.id,0)
            available = src.ships - reserve - committed
            if available < 3: continue

            tx, ty, tt = intercept(src, tgt, available, ang_vel)
            if not is_path_clear(src.x, src.y, tx, ty): continue

            # ARRIVAL-TIME: simulate garrison at our fleet's ETA
            arr_owner, arr_garrison = _simulate_garrison_at(tgt, tt, fleet_arrivals)
            if arr_owner == me: continue  # will be ours already

            margin = cfg["margin_enemy"]
            needed = int(arr_garrison) + margin
            if needed < 1: needed = 1
            if needed > available: continue

            d_to_tgt = dist(src.x, src.y, tgt.x, tgt.y)
            eligible.append((d_to_tgt, src, needed, tx, ty, tt))

        if not eligible: continue

        # TIMED CONVERGENCE: farthest fires first → simultaneous landing
        eligible.sort(key=lambda x: -x[0])

        total_sent = 0
        srcs_used  = 0
        for d_to_tgt, src, needed, tx, ty, tt in eligible[:max_srcs]:
            reserve   = _compute_reserve(src, src.id in threatened_ids, phase, cfg,
                                         is_rear=(src.id in rear_ids))
            committed = def_committed.get(src.id,0) + atk_committed.get(src.id,0)
            available = src.ships - reserve - committed
            remaining = needed - total_sent
            if remaining <= 0: break
            send = min(available, remaining + 1)
            if send < 1: continue
            angle = math.atan2(ty-src.y, tx-src.x)
            actions.append([src.id, angle, send])
            atk_committed[src.id] = atk_committed.get(src.id,0) + send
            total_sent += send; srcs_used += 1
            if total_sent >= needed: break

    return actions, atk_committed


# ─── Counter-Attack ───────────────────────────────────────────────────────────

def _counter_attack(my_planets, enemy_planets, all_planets, fleets,
                    ang_vel, me, phase, def_committed, atk_committed, cfg, fleet_arrivals):
    if phase == 0 or not enemy_planets: return []
    actions = []
    rear_ids = _classify_rear(my_planets, enemy_planets)

    for tgt in sorted(enemy_planets, key=lambda p: p.ships / max(1, p.production))[:3]:
        for src in sorted(my_planets, key=lambda p: dist(p.x,p.y,tgt.x,tgt.y)):
            reserve   = _compute_reserve(src, False, phase, cfg,
                                         is_rear=(src.id in rear_ids))
            committed = def_committed.get(src.id,0) + atk_committed.get(src.id,0)
            available = src.ships - reserve - committed
            if available < 5: continue
            tx, ty, tt = intercept(src, tgt, available, ang_vel)
            if not is_path_clear(src.x, src.y, tx, ty): continue
            arr_owner, arr_garrison = _simulate_garrison_at(tgt, tt, fleet_arrivals)
            if arr_owner == me: continue
            needed = int(arr_garrison) + cfg["margin_enemy"]
            if needed < 1: needed = 1
            if needed <= available and available >= needed * cfg["counter_ratio"]:
                angle = math.atan2(ty-src.y, tx-src.x)
                actions.append([src.id, angle, needed])
                atk_committed[src.id] = atk_committed.get(src.id,0) + needed
                break
    return actions


# ─── Logistics ────────────────────────────────────────────────────────────────

def _logistics(my_planets, enemy_planets, cfg):
    if len(my_planets) < 2 or not enemy_planets: return []

    def front_score(p):
        return min(dist(p.x,p.y,e.x,e.y) for e in enemy_planets)

    sorted_mine = sorted(my_planets, key=front_score, reverse=True)
    front = sorted_mine[-1]
    for src in sorted_mine[:-1]:
        reserve = int(cfg["min_reserve"] + cfg["reserve_prod_mult"] * src.production)
        surplus = src.ships - reserve
        if surplus < cfg["logistics_surplus_min"]: continue
        send = int(surplus * cfg["logistics_ratio"])
        if send < 5: continue
        if is_path_clear(src.x, src.y, front.x, front.y):
            angle = math.atan2(front.y-src.y, front.x-src.x)
            return [[src.id, angle, send]]
    return []


# ─── Agent Entry Point ────────────────────────────────────────────────────────

def agent(obs, config=None):
    """
    Orbit Wars agent — v9 "Arrival-Time".
    Pipeline:
      1. Build fleet arrival timeline (who lands where and when)
      2. Emergency defense   — uses arrival-time garrison
      3. Neutral expansion   — skips planets we'll already own
      4. Enemy attack        — timed convergence + arrival-time need
      5. Counter-attack      — snipe weak enemies
      6. Logistics           — redistribute surplus
    """
    me, planets, fleets, comets, comet_ids, step, ang_vel = _unpack_obs(obs)
    cfg = _load_params()

    my_planets    = [p for p in planets if p.owner == me]
    enemy_planets = [p for p in planets if p.owner not in (-1, me)]
    neutrals      = [p for p in planets if p.owner == -1]

    if not my_planets: return []

    phase = _detect_phase(step, planets, my_planets, enemy_planets)

    # Build the core arrival timeline — used by ALL passes
    fleet_arrivals = _build_fleet_arrivals(planets, fleets, ang_vel)

    def_actions, def_committed = _emergency_defense(
        my_planets, planets, fleets, me, ang_vel, cfg, fleet_arrivals)

    neu_actions, neu_committed = _neutral_expansion(
        my_planets, neutrals, comets, planets, fleets,
        ang_vel, me, phase, cfg, def_committed, fleet_arrivals)

    atk_actions, atk_committed = _enemy_attack(
        my_planets, enemy_planets, planets, fleets,
        ang_vel, me, phase, cfg, def_committed, neu_committed, fleet_arrivals)

    counter_actions = _counter_attack(
        my_planets, enemy_planets, planets, fleets,
        ang_vel, me, phase, def_committed, atk_committed, cfg, fleet_arrivals)

    log_actions = _logistics(my_planets, enemy_planets, cfg) if not def_actions else []

    return def_actions + neu_actions + atk_actions + counter_actions + log_actions
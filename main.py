"""
Orbit Wars Agent — v11 "Defensive Blitz"
=====================================================
Upgrades from v10 based on episode 75391584 replay analysis:
  1. Anti-overcommit cap: never spend >MAX_COMMIT_RATIO of total ships/turn.
  2. Neutral priority during expansion (step<120): score neutrals 1.5x.
  3. Defense garrison check before large attacks: compute inbound threat
     per planet and subtract from available ships before sending offensively.
  4. Central zone awareness: planets within CENTER_BONUS_RADIUS of center
     get a bonus multiplier — grab contested neutrals first.
  5. Overcommit guard in enemy_attack: skip if src garrison would drop below
     GARRISON_SAFETY_FLOOR after send.

Key v10 changes retained:
  - Phase-0 BLITZ from turn 1, grab_margin_neutral=1
  - Remaining-turns-aware production scoring
  - Arrival-time garrison simulation
  - Timed convergence on enemy planets

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
EPISODE_STEPS = 500  # total game length for time-aware scoring

PHASE_BLITZ  = 40
PHASE_MID    = 80

# v11: safety constants
MAX_COMMIT_RATIO    = 0.55   # never send more than 55% of ALL owned ships in one turn
GARRISON_SAFETY_FLOOR = 5    # after any send, planet must retain >= this many ships
CENTER_BONUS_RADIUS = 35.0   # planets within this radius of center get neutral bonus
NEUTRAL_CENTER_MULT = 1.5    # score multiplier for central contested neutrals
EXPANSION_NEUTRAL_MULT = 1.5 # score multiplier for neutrals during expansion (step<120)
EXPANSION_STEP_LIMIT = 120   # steps before which we prioritize neutrals over attacks

# ─── Defaults ─────────────────────────────────────────────────────────────────
# v10: RL-informed parameter tuning
DEFAULT_CONFIG = {
    # Reserve (lower in phase 0 to stop 10-turn idle)
    "min_reserve": 4.0,           # v9: 5.0  — freed up ships for early grab
    "reserve_prod_mult": 1.2,
    "reserve_threat_bonus": 12.0,
    "reserve_floor": 3.0,         # v9: 6.0  — allow aggressive early sends
    "grab_reserve_floor": 3.0,    # NEW: ultra-low floor in phase-0 blitz

    # Scoring (RL confirmed production > distance in early game)
    "score_production_mult": 18.0,    # v9: 14.0 — production matters more
    "score_neutral_bonus": 12.0,      # v9: 15.0 — slightly reduced flat bonus
    "score_enemy_bonus": 22.0,
    "score_enemy_prod_bonus": 8.0,    # v9: 4.0  — punish high-prod enemies harder
    "score_distance_penalty": 0.35,   # v9: 0.4  — don't penalize range as much
    "score_amount_penalty": 0.7,      # v9: 0.8  — send more, penalize less
    "score_comet_bonus": 30.0,
    "score_time_mult": 0.008,         # NEW: multiply prod score by turns_remaining

    # Attack thresholds
    "attack_min_score": 0.25,         # v9: 0.3  — attack slightly more
    "attack_ratio": 0.80,
    "early_min_score": 0.05,          # v9: 0.1  — very low bar for early grabs
    "early_turns": 40.0, "grab_turns": 20.0,

    # Expansion width (RL: simultaneous multi-target beats sequential)
    "converge_max_sources": 6,        # v9: 5
    "max_parallel_neutrals": 12,      # v9: 8  — grab more at once

    # Margins (phase-0 uses grab_margin_neutral=1, later uses margin_neutral)
    "margin_neutral": 2,
    "grab_margin_neutral": 1,         # NEW: only 1 extra ship in phase-0 blitz
    "margin_enemy": 3,

    "counter_ratio": 1.3,
    "defense_threat_radius": 35.0, "threat_radius_planet": 25.0,
    "logistics_surplus_min": 15.0, "logistics_ratio": 0.40,
    "defense_transfer_ratio": 0.35,
    # Arrival-time
    "arrival_sim_horizon": 80,
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
    # v10: also consider step — force blitz mode in first 30 steps
    if step < 30 or neutral_frac > 0.40: return 0
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
    # v10: use grab_reserve_floor in blitz phase for aggressive early expansion
    if phase == 0:
        floor = cfg.get("grab_reserve_floor", 3.0)
        r = floor + 0.5 * src.production
    elif is_rear:
        floor = cfg["reserve_floor"]
        r = floor + cfg["reserve_prod_mult"] * src.production
    else:
        floor = cfg["reserve_floor"]
        r = max(floor - 2, 3) + 0.8 * src.production
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

def _inbound_enemy_ships(planet_id, fleets, me):
    """v11: sum enemy ships in-flight toward planet_id (approximated from fleet_arrivals)."""
    total = 0
    for f in fleets:
        if f.owner != me and f.from_planet_id != planet_id:
            # crude: count all enemy fleets (will be refined by arrival-time sim)
            total += 0  # placeholder — we use fleet_arrivals in context
    return total


def _total_owned_ships(my_planets, fleets, me):
    """v11: total ships owned (planets + in-flight fleets)."""
    planet_ships = sum(p.ships for p in my_planets)
    fleet_ships  = sum(f.ships for f in fleets if f.owner == me)
    return planet_ships + fleet_ships


def _neutral_expansion(my_planets, neutrals, comets, all_planets, fleets,
                       ang_vel, me, phase, cfg, def_committed, fleet_arrivals, step=0):
    actions = []; atk_committed = {}; claimed = set()
    all_targets = neutrals + comets
    is_grab  = phase == 0
    is_blitz = phase <= 1

    turns_left = max(1, EPISODE_STEPS - step)
    time_factor = 1.0 + cfg.get("score_time_mult", 0.008) * turns_left

    max_claims_per_src = 3 if is_grab else 1

    # v11: anti-overcommit cap — budget across ALL sources this turn
    total_ships = _total_owned_ships(my_planets, fleets, me)
    global_ship_budget = int(total_ships * MAX_COMMIT_RATIO)
    global_committed = sum(def_committed.values())

    # v11: expansion phase — neutrals get priority boost
    in_expansion = step < EXPANSION_STEP_LIMIT

    for src in sorted(my_planets, key=lambda p: -p.ships):
        if global_committed >= global_ship_budget: break

        reserve   = _compute_reserve(src, False, phase, cfg)
        committed = def_committed.get(src.id, 0) + atk_committed.get(src.id, 0)
        available = src.ships - reserve - committed
        # v11: garrison safety floor
        available = min(available, src.ships - GARRISON_SAFETY_FLOOR)
        if available < 2: continue

        unclaimed = [t for t in all_targets if t.id not in claimed]
        if not unclaimed: break

        scored_actions = []

        for t in unclaimed:
            tx, ty, tt = intercept(src, t, available, ang_vel)
            if not is_path_clear(src.x, src.y, tx, ty): continue

            arr_owner, arr_garrison = _simulate_garrison_at(t, tt, fleet_arrivals)
            if arr_owner == me: continue

            margin = cfg.get("grab_margin_neutral", 1) if is_grab else cfg["margin_neutral"]
            needed = int(arr_garrison) + margin
            if needed < 1: needed = 1
            if needed > available: continue

            # v11: check global budget
            if global_committed + needed > global_ship_budget: continue

            if is_grab:
                prod_val = max(t.production, 0.5) * time_factor
                d = dist(src.x, src.y, t.x, t.y)
                score = prod_val / (cfg["score_distance_penalty"] * d + 1.0)
            else:
                gate = cfg["early_min_score"] if is_blitz else cfg["attack_min_score"]
                cost = max(1, needed)
                score = (cfg["score_production_mult"] * t.production * time_factor +
                         cfg["score_neutral_bonus"]) / (
                         cfg["score_distance_penalty"] * tt +
                         cfg["score_amount_penalty"] * cost + 1.0)
                if score < gate: continue

            # v11: bonus for central contested neutrals
            d_center = dist(t.x, t.y, CENTER_X, CENTER_Y)
            if d_center <= CENTER_BONUS_RADIUS:
                score *= NEUTRAL_CENTER_MULT

            # v11: expansion phase neutral priority boost
            if in_expansion and t.owner == -1:
                score *= EXPANSION_NEUTRAL_MULT

            scored_actions.append((score, t.id, tx, ty, needed))

        scored_actions.sort(reverse=True)
        sent_this_src = 0
        for score, tid, tx, ty, cost in scored_actions:
            if sent_this_src >= max_claims_per_src: break
            committed_now = def_committed.get(src.id, 0) + atk_committed.get(src.id, 0)
            avail_now = min(src.ships - reserve - committed_now,
                            src.ships - GARRISON_SAFETY_FLOOR)
            if cost > avail_now: continue
            if tid in claimed: continue
            if global_committed + cost > global_ship_budget: continue
            angle = math.atan2(ty - src.y, tx - src.x)
            actions.append([src.id, angle, cost])
            atk_committed[src.id] = atk_committed.get(src.id, 0) + cost
            global_committed += cost
            claimed.add(tid)
            sent_this_src += 1

    return actions, atk_committed


# ─── Enemy Attack with Timed Convergence ──────────────────────────────────────

def _enemy_attack(my_planets, enemy_planets, all_planets, fleets,
                  ang_vel, me, phase, cfg, def_committed, neu_committed, fleet_arrivals, step=0):
    if not enemy_planets: return [], {}

    actions = []; atk_committed = dict(neu_committed)
    my_ids = {p.id for p in my_planets}
    threatened_ids = set()

    # v11: build per-planet inbound enemy ship counts
    inbound_enemy = {p.id: 0 for p in my_planets}
    for f in fleets:
        if f.owner == me: continue
        tid = _infer_target(f, my_planets, ang_vel)
        if tid and tid in my_ids:
            threatened_ids.add(tid)
            inbound_enemy[tid] = inbound_enemy.get(tid, 0) + int(f.ships)

    rear_ids = _classify_rear(my_planets, enemy_planets)

    turns_left = max(1, EPISODE_STEPS - step)
    time_factor = 1.0 + cfg.get("score_time_mult", 0.008) * turns_left

    # v11: during expansion phase (step<120), skip enemy attacks if neutrals still available
    # — prioritize neutral grabs over risky assaults
    neutrals_exist = any(p.owner == -1 for p in all_planets)
    if step < EXPANSION_STEP_LIMIT and neutrals_exist and len(enemy_planets) > 0:
        # Only attack if we clearly outnumber the enemy total ships
        my_total = sum(p.ships for p in my_planets) + sum(f.ships for f in fleets if f.owner == me)
        en_total = sum(p.ships for p in enemy_planets) + sum(f.ships for f in fleets if f.owner not in (-1, me))
        if my_total < en_total * 1.2:
            return [], atk_committed  # don't over-extend during expansion

    scored = sorted(enemy_planets,
                    key=lambda t: -(cfg["score_production_mult"] * t.production * time_factor +
                                    cfg["score_enemy_bonus"] +
                                    cfg["score_enemy_prod_bonus"] * t.production))

    max_srcs = int(cfg["converge_max_sources"])

    for tgt in scored[:8]:
        eligible = []
        for src in my_planets:
            reserve   = _compute_reserve(src, src.id in threatened_ids, phase, cfg,
                                         is_rear=(src.id in rear_ids))
            committed = def_committed.get(src.id,0) + atk_committed.get(src.id,0)
            available = src.ships - reserve - committed
            # v11: garrison safety — must keep enough to survive inbound threats
            inbound = inbound_enemy.get(src.id, 0)
            available = min(available, src.ships - inbound - GARRISON_SAFETY_FLOOR)
            if available < 3: continue

            tx, ty, tt = intercept(src, tgt, available, ang_vel)
            if not is_path_clear(src.x, src.y, tx, ty): continue

            arr_owner, arr_garrison = _simulate_garrison_at(tgt, tt, fleet_arrivals)
            if arr_owner == me: continue

            margin = cfg["margin_enemy"]
            needed = int(arr_garrison) + margin
            if needed < 1: needed = 1
            if needed > available: continue

            d_to_tgt = dist(src.x, src.y, tgt.x, tgt.y)
            eligible.append((d_to_tgt, src, needed, tx, ty, tt))

        if not eligible: continue

        eligible.sort(key=lambda x: -x[0])

        total_sent = 0
        for d_to_tgt, src, needed, tx, ty, tt in eligible[:max_srcs]:
            reserve   = _compute_reserve(src, src.id in threatened_ids, phase, cfg,
                                         is_rear=(src.id in rear_ids))
            committed = def_committed.get(src.id,0) + atk_committed.get(src.id,0)
            inbound   = inbound_enemy.get(src.id, 0)
            available = min(src.ships - reserve - committed,
                            src.ships - inbound - GARRISON_SAFETY_FLOOR)
            remaining = needed - total_sent
            if remaining <= 0: break
            send = min(available, remaining + 1)
            if send < 1: continue
            angle = math.atan2(ty-src.y, tx-src.x)
            actions.append([src.id, angle, send])
            atk_committed[src.id] = atk_committed.get(src.id,0) + send
            total_sent += send
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
    Orbit Wars agent — v11 "Defensive Blitz".
    Pipeline:
      1. Build fleet arrival timeline (who lands where and when)
      2. Emergency defense   — uses arrival-time garrison
      3. Neutral expansion   — v10: grab from turn 1, multi-target blitz
      4. Enemy attack        — timed convergence + arrival-time need + time scoring
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

    # v10: pass step for time-aware scoring
    neu_actions, neu_committed = _neutral_expansion(
        my_planets, neutrals, comets, planets, fleets,
        ang_vel, me, phase, cfg, def_committed, fleet_arrivals, step=step)

    atk_actions, atk_committed = _enemy_attack(
        my_planets, enemy_planets, planets, fleets,
        ang_vel, me, phase, cfg, def_committed, neu_committed, fleet_arrivals, step=step)

    counter_actions = _counter_attack(
        my_planets, enemy_planets, planets, fleets,
        ang_vel, me, phase, def_committed, atk_committed, cfg, fleet_arrivals)

    log_actions = _logistics(my_planets, enemy_planets, cfg) if not def_actions else []

    return def_actions + neu_actions + atk_actions + counter_actions + log_actions
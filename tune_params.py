"""Random-search parameter tuner for Orbit Wars agent heuristics."""

import argparse
import importlib
import json
import os
import random


def _sample_param(base_value, rng, min_mult, max_mult, as_int=False):
    value = base_value * rng.uniform(min_mult, max_mult)
    if as_int:
        return max(1, int(round(value)))
    return float(value)


def _sample_candidate(defaults, rng):
    candidate = dict(defaults)
    candidate["reserve_base"] = _sample_param(defaults["reserve_base"], rng, 0.6, 1.6)
    candidate["reserve_production_mult"] = _sample_param(defaults["reserve_production_mult"], rng, 0.6, 1.6)
    candidate["reserve_threat_mult"] = _sample_param(defaults["reserve_threat_mult"], rng, 0.4, 1.8)
    candidate["reserve_comet_bonus"] = _sample_param(defaults["reserve_comet_bonus"], rng, 0.5, 1.5)
    candidate["score_neutral_bonus"] = _sample_param(defaults["score_neutral_bonus"], rng, 0.5, 1.8)
    candidate["score_enemy_bonus"] = _sample_param(defaults["score_enemy_bonus"], rng, 0.6, 1.6)
    candidate["score_production_mult"] = _sample_param(defaults["score_production_mult"], rng, 0.6, 1.6)
    candidate["score_amount_penalty"] = _sample_param(defaults["score_amount_penalty"], rng, 0.6, 1.8)
    candidate["score_distance_penalty"] = _sample_param(defaults["score_distance_penalty"], rng, 0.5, 1.8)
    candidate["score_comet_penalty"] = _sample_param(defaults["score_comet_penalty"], rng, 0.2, 2.0)
    candidate["attack_min_score"] = _sample_param(defaults["attack_min_score"], rng, 0.4, 1.8)
    candidate["defense_transfer_ratio"] = max(0.05, min(0.9, _sample_param(defaults["defense_transfer_ratio"], rng, 0.5, 1.8)))
    candidate["defense_transfer_prod_mult"] = _sample_param(defaults["defense_transfer_prod_mult"], rng, 0.5, 1.8)
    return candidate


def _load_agent_with_params(params):
    os.environ["ORBIT_WARS_PARAMS"] = json.dumps(params)
    import main

    importlib.reload(main)
    return main.agent


def _evaluate_candidate(params, episodes, opponents, seed):
    try:
        from kaggle_environments import make
    except Exception as exc:
        raise RuntimeError(f"kaggle_environments is not available: {exc}") from exc

    rng = random.Random(seed)
    agent = _load_agent_with_params(params)

    wins = 0
    draws = 0
    total_reward = 0.0

    for _ in range(episodes):
        opponent = rng.choice(opponents)
        env = make("orbit_wars", debug=True)
        env.run([agent, opponent])
        reward = env.steps[-1][0].reward
        opp_reward = env.steps[-1][1].reward

        total_reward += reward
        if reward > opp_reward:
            wins += 1
        elif reward == opp_reward:
            draws += 1

    losses = episodes - wins - draws
    avg_reward = total_reward / episodes if episodes else 0.0
    score = wins + 0.5 * draws + 0.1 * avg_reward

    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "avg_reward": avg_reward,
        "score": score,
    }


def main():
    parser = argparse.ArgumentParser(description="Tune Orbit Wars heuristic weights with random search.")
    parser.add_argument("--trials", type=int, default=12, help="Number of random candidates to test.")
    parser.add_argument("--episodes", type=int, default=6, help="Episodes per candidate.")
    parser.add_argument("--opponents", default="random", help="Comma-separated opponents for evaluation.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--output",
        default="best_params.json",
        help="Path to write best parameter JSON.",
    )
    args = parser.parse_args()

    opponents = [item.strip() for item in args.opponents.split(",") if item.strip()]
    if not opponents:
        print("No opponents configured.")
        return 2

    import main

    defaults = dict(main.DEFAULT_CONFIG)
    best_params = dict(defaults)
    best_metrics = _evaluate_candidate(best_params, args.episodes, opponents, args.seed)

    print("Baseline metrics:", best_metrics)

    rng = random.Random(args.seed)
    for trial_index in range(1, args.trials + 1):
        candidate = _sample_candidate(defaults, rng)
        metrics = _evaluate_candidate(candidate, args.episodes, opponents, args.seed + trial_index)
        print(f"Trial {trial_index}: score={metrics['score']:.3f}, metrics={metrics}")

        if metrics["score"] > best_metrics["score"]:
            best_metrics = metrics
            best_params = candidate
            print("  New best candidate found.")

    with open(args.output, "w", encoding="utf-8") as out_file:
        json.dump(best_params, out_file, indent=2)

    print("Best metrics:", best_metrics)
    print(f"Best params written to {args.output}")
    print("Use with: ORBIT_WARS_PARAMS=<json> python local_eval.py --episodes 20 --opponent random")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

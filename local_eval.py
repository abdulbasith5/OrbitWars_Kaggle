"""Run local Orbit Wars matches for quick validation."""

import argparse
import json
import os
from pathlib import Path
import random
import sys


def _load_agent():
    from main import agent

    return agent


def main():
    parser = argparse.ArgumentParser(description="Run local Orbit Wars evaluation games.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of games to run.")
    parser.add_argument("--opponent", default="random", help="Built-in opponent name.")
    parser.add_argument(
        "--opponents",
        default="",
        help="Comma-separated opponent names. Overrides --opponent when provided.",
    )
    parser.add_argument(
        "--params",
        default="",
        help="JSON object of parameter overrides for main.py (ORBIT_WARS_PARAMS).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for opponent sampling.")
    parser.add_argument("--render", action="store_true", help="Render the final game in notebook mode.")
    parser.add_argument(
        "--save-render",
        default="",
        help="Write final episode UI replay to an HTML file path.",
    )
    args = parser.parse_args()

    if args.params:
        try:
            parsed = json.loads(args.params)
        except json.JSONDecodeError as exc:
            print(f"Invalid --params JSON: {exc}")
            return 2
        if not isinstance(parsed, dict):
            print("--params must be a JSON object.")
            return 2
        os.environ["ORBIT_WARS_PARAMS"] = json.dumps(parsed)

    try:
        from kaggle_environments import make
    except Exception as exc:
        print("kaggle_environments is not available in this environment.")
        print("Install it locally to run validation games.")
        print(f"Import error: {exc}")
        return 1

    agent = _load_agent()
    if args.opponents.strip():
        opponents = [name.strip() for name in args.opponents.split(",") if name.strip()]
    else:
        opponents = [args.opponent]

    if not opponents:
        print("No opponent configured.")
        return 2

    random.seed(args.seed)
    results = []

    for episode in range(args.episodes):
        opponent = random.choice(opponents)
        env = make("orbit_wars", debug=True)
        env.run([agent, opponent])
        final_states = env.steps[-1]
        rewards = [state.reward for state in final_states]
        statuses = [state.status for state in final_states]
        results.append((rewards, statuses, opponent))
        print(f"Episode {episode + 1} vs {opponent}: rewards={rewards}, statuses={statuses}")

        if args.render and episode == args.episodes - 1:
            env.render(mode="ipython", width=800, height=600)

        if args.save_render and episode == args.episodes - 1:
            html_path = Path(args.save_render)
            html_path.parent.mkdir(parents=True, exist_ok=True)
            html = env.render(mode="html", width=900, height=700)
            html_path.write_text(str(html), encoding="utf-8")
            print(f"Saved replay UI to: {html_path}")

    if results:
        agent_wins = sum(1 for rewards, _, _ in results if rewards[0] > rewards[1])
        draws = sum(1 for rewards, _, _ in results if rewards[0] == rewards[1])
        average_reward = sum(rewards[0] for rewards, _, _ in results) / len(results)
        print(f"Summary: wins={agent_wins}, draws={draws}, losses={len(results) - agent_wins - draws}")
        print(f"Average reward: {average_reward:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
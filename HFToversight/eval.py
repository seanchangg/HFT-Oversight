"""Evaluate an LLM agent on the HFT Oversight environment.

Runs episodes at each difficulty level using the HF Inference API (free),
prints a results table, and optionally saves a reward curve.

Usage:
    uv run python eval.py
    uv run python eval.py --episodes 20 --difficulties 1,3,5,7
    uv run python eval.py --save-csv results.csv
"""

import json
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from server.environment import HFTOversightEnvironment
from models import OversightAction

SYSTEM_PROMPT = (
    "You are an HFT oversight agent. You manage a fleet of trading bots. "
    "One or more bots may be malfunctioning. Your job:\n"
    "1. Read logs for each bot (command: read_logs)\n"
    "2. Cross-reference suspicious bots (command: cross_reference)\n"
    "3. Flag suspicious bots (command: flag)\n"
    "4. Shut down bad bots with a reason (command: shutdown)\n\n"
    "Respond with ONLY a JSON object like: "
    '{\"command\": \"read_logs\", \"bot_id\": \"alpha\"}'
)


def parse_action(text: str) -> OversightAction:
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1].removeprefix("json").strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        data = json.loads(text[start:end])
        return OversightAction(**data)
    raise ValueError(f"Could not parse action from: {text[:100]}")


def run_episode(difficulty: int, client, verbose: bool = False):
    """Run a single episode and return results."""
    env = HFTOversightEnvironment()
    env._difficulty = difficulty
    obs = env.reset()

    issue_type = env._scenario["issue"]
    bad_bots = env._bad_bots

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": obs.response + (f"\n\nAlerts: {obs.alerts}" if obs.alerts else "")},
    ]

    total_reward = 0.0
    actions_taken = []

    while not obs.done:
        try:
            resp = client.chat_completion(messages=messages, max_tokens=200, temperature=0.3)
            llm_text = resp.choices[0].message.content
        except Exception as e:
            if verbose:
                print(f"  API error: {e}")
            llm_text = '{"command": "pass_turn"}'

        try:
            action = parse_action(llm_text)
        except Exception:
            if verbose:
                print(f"  Parse failed: {llm_text[:80]}")
            action = OversightAction(command="pass_turn")

        actions_taken.append(action.command)
        obs = env.step(action)
        total_reward += obs.reward

        if verbose:
            print(f"  Step {obs.timestep}: {action.command} {action.bot_id or ''} -> reward={obs.reward:.1f}")

        messages.append({"role": "assistant", "content": llm_text})
        env_msg = obs.response
        if obs.alerts:
            env_msg += f"\n\nAlerts: {obs.alerts}"
        env_msg += f"\n\n[Step {obs.timestep}/{obs.max_timesteps}]"
        messages.append({"role": "user", "content": env_msg})

    won = all(b in env._bots_shutdown for b in bad_bots)
    return {
        "difficulty": difficulty,
        "issue": issue_type,
        "bad_bots": bad_bots,
        "total_reward": total_reward,
        "won": won,
        "steps": obs.timestep,
        "actions": actions_taken,
        "correct_shutdowns": sum(1 for b in bad_bots if b in env._bots_shutdown),
        "wrong_shutdowns": sum(1 for b in env._bots_shutdown if b not in bad_bots),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on HFT Oversight")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per difficulty level")
    parser.add_argument("--difficulties", type=str, default="1,3,5,7", help="Comma-separated difficulty levels")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save-csv", type=str, default=None)
    args = parser.parse_args()

    from huggingface_hub import InferenceClient, get_token

    client = InferenceClient(model=args.model, token=get_token())
    difficulties = [int(d) for d in args.difficulties.split(",")]

    all_results = []
    print(f"\nEvaluating {args.model} on HFT Oversight")
    print(f"Episodes per level: {args.episodes}")
    print(f"Difficulties: {difficulties}")
    print("=" * 70)

    for diff in difficulties:
        print(f"\nDifficulty {diff}:")
        for ep in range(args.episodes):
            result = run_episode(diff, client, verbose=args.verbose)
            all_results.append(result)
            status = "WIN" if result["won"] else "LOSS"
            print(f"  ep {ep+1}: {status} | reward={result['total_reward']:+.1f} | "
                  f"issue={result['issue']} | steps={result['steps']}")

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Difficulty':<12} {'Win Rate':<12} {'Avg Reward':<14} {'Avg Steps':<12} {'Episodes'}")
    print("-" * 62)
    for diff in difficulties:
        d_results = [r for r in all_results if r["difficulty"] == diff]
        if not d_results:
            continue
        win_rate = sum(1 for r in d_results if r["won"]) / len(d_results)
        avg_reward = sum(r["total_reward"] for r in d_results) / len(d_results)
        avg_steps = sum(r["steps"] for r in d_results) / len(d_results)
        print(f"{diff:<12} {win_rate:<12.0%} {avg_reward:<14.1f} {avg_steps:<12.1f} {len(d_results)}")

    # Issue breakdown
    print(f"\n{'Issue Type':<20} {'Win Rate':<12} {'Avg Reward':<14} {'Count'}")
    print("-" * 58)
    issues = set(r["issue"] for r in all_results)
    for issue in sorted(issues):
        i_results = [r for r in all_results if r["issue"] == issue]
        win_rate = sum(1 for r in i_results if r["won"]) / len(i_results)
        avg_reward = sum(r["total_reward"] for r in i_results) / len(i_results)
        print(f"{issue:<20} {win_rate:<12.0%} {avg_reward:<14.1f} {len(i_results)}")

    if args.save_csv:
        import csv
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "difficulty", "issue", "total_reward", "won", "steps",
                "correct_shutdowns", "wrong_shutdowns",
            ])
            writer.writeheader()
            for r in all_results:
                writer.writerow({k: r[k] for k in writer.fieldnames})
        print(f"\nResults saved to {args.save_csv}")


if __name__ == "__main__":
    main()

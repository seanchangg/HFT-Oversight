"""Run a HuggingFace LLM agent against the HFT Oversight Environment.

Collects trajectories for fine-tuning. Start with difficulty=1 (obvious errors).

Usage:
    uv run python run_agent.py --episodes 10
    uv run python run_agent.py --episodes 10 --difficulty 1
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from huggingface_hub import InferenceClient
from server.environment import HFTOversightEnvironment
from models import OversightAction
from train import judge_shutdown_reasons, JUDGE_MODEL

# --- Config ---
from huggingface_hub import get_token
HF_TOKEN = os.environ.get("HF_TOKEN", "") or get_token() or ""
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = """You are an HFT oversight agent. You must find and shut down malfunctioning trading bots.

Each turn, respond with ONLY a JSON action. No other text.

Commands:
- {"command": "read_logs", "bot_id": "NAME"}
- {"command": "read_logs", "bot_id": "NAME", "start_time": "HH:MM:SS", "end_time": "HH:MM:SS"}
- {"command": "check_pnl", "bot_id": "NAME"}
- {"command": "read_file", "bot_id": "NAME", "path": "config.yaml"}
- {"command": "cross_reference", "bot_id": "NAME1", "bot_id_2": "NAME2"}
- {"command": "flag", "bot_id": "NAME", "reason": "WHY"}
- {"command": "shutdown", "bot_id": "NAME", "reason": "WHY"}

Read each bot's activity logs carefully. Look for unusual patterns in their trading behavior — anomalous fills, one-sided positions, repeated prices, or suspicious timing between bots. Flag a bot before shutting it down."""


def parse_action(text: str) -> OversightAction:
    text = text.strip()
    # Strip markdown code blocks
    if "```" in text:
        text = text.split("```")[1].removeprefix("json").strip()
    start = text.find("{")
    if start < 0:
        raise ValueError(f"Could not parse action from: {text}")
    # Find the matching closing brace for the first object
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                data = json.loads(text[start:i + 1])
                return OversightAction(**data)
    raise ValueError(f"Could not parse action from: {text}")


def run_episode(client: InferenceClient, difficulty: int = 1, judge_client: InferenceClient = None) -> dict:
    env = HFTOversightEnvironment()
    env._difficulty = difficulty
    obs = env.reset()

    print(f"\n{'='*60}")
    print(f"EPISODE (difficulty={difficulty})")
    print(f"{'='*60}")
    print(obs.response[:300])
    if obs.alerts:
        print(f"Alerts: {obs.alerts}")

    # Build conversation
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": obs.response + (f"\n\nAlerts: {obs.alerts}" if obs.alerts else "")},
    ]

    trajectory = []
    total_reward = 0.0
    shutdown_reasons = []

    while not obs.done:
        # Query the model
        try:
            response = client.chat_completion(
                messages=messages,
                max_tokens=200,
                temperature=0.3,
            )
            llm_text = response.choices[0].message.content
        except Exception as e:
            print(f"  Model error: {e}")
            # Trim conversation history to stay within context window
            if "tokens" in str(e) and len(messages) > 4:
                # Keep system + initial obs + last 4 exchanges
                messages = messages[:2] + messages[-4:]
                print("  (trimmed conversation history, retrying)")
                continue
            llm_text = '{"command": "list_bots"}'
            consecutive_errors = consecutive_errors + 1 if 'consecutive_errors' in dir() else 1
            if consecutive_errors >= 3:
                print("  3 consecutive model errors — aborting episode.")
                break

        print(f"\n  LLM (step {obs.timestep + 1}): {llm_text[:150]}")

        try:
            action = parse_action(llm_text)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  Parse error: {e}")
            action = OversightAction(command="list_bots")

        # Step environment
        obs = env.step(action)
        total_reward += obs.reward

        # Capture shutdown reasons for LLM judge
        if action.command == "shutdown" and action.bot_id:
            correct = obs.reward > 0
            shutdown_reasons.append({
                "bot_id": action.bot_id,
                "reason": action.reason or "",
                "issue_type": env._scenario.get("issue", ""),
                "correct": correct,
            })

        print(f"  ENV: {obs.response[:150]}")
        print(f"  [reward={obs.reward}, total={total_reward}, step={obs.timestep}/{obs.max_timesteps}]")

        # Record trajectory step
        trajectory.append({
            "messages_so_far": [m.copy() for m in messages],
            "assistant_response": llm_text,
            "action": action.model_dump(exclude_none=True),
            "reward": obs.reward,
            "cumulative_reward": total_reward,
            "done": obs.done,
        })

        # Feed back to conversation
        messages.append({"role": "assistant", "content": llm_text})
        env_msg = obs.response
        if obs.alerts:
            env_msg += f"\n\nAlerts: {obs.alerts}"
        env_msg += f"\n\n[Step {obs.timestep}/{obs.max_timesteps}]"
        messages.append({"role": "user", "content": env_msg})

    # LLM-as-judge scoring for shutdown reasoning
    reasoning_score = 0.0
    if judge_client and shutdown_reasons:
        reasoning_score = judge_shutdown_reasons(shutdown_reasons, judge_client)
        print(f"  Reasoning score (LLM judge): {reasoning_score:.2f}")

    print(f"\n  DONE — Total reward: {total_reward}")

    return {
        "difficulty": difficulty,
        "total_reward": total_reward,
        "steps": len(trajectory),
        "trajectory": trajectory,
        "full_conversation": messages,
        "shutdown_reasons": shutdown_reasons,
        "reasoning_score": reasoning_score,
    }


DIFFICULTY_LEVELS = [1, 2, 3, 5, 7]
FAST_SOLVE_THRESHOLD = 3  # solved in <= this many steps = "quick"
STREAK_TO_ADVANCE = 3     # consecutive wins to level up


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--difficulty", type=int, default=1, help="Starting difficulty")
    parser.add_argument("--adaptive", action="store_true", default=True,
                        help="Auto-increase difficulty (default: on)")
    parser.add_argument("--no-adaptive", dest="adaptive", action="store_false")
    parser.add_argument("--no-judge", action="store_true", help="Disable LLM-as-judge reasoning scoring")
    parser.add_argument("--output", type=str, default="trajectories.jsonl")
    args = parser.parse_args()

    if not HF_TOKEN:
        print("Set HF_TOKEN env var: export HF_TOKEN=hf_xxx")
        sys.exit(1)

    client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)
    judge_client = None if args.no_judge else InferenceClient(model=JUDGE_MODEL, token=HF_TOKEN)

    # Adaptive difficulty state
    difficulty = args.difficulty
    win_streak = 0
    level_idx = DIFFICULTY_LEVELS.index(difficulty) if difficulty in DIFFICULTY_LEVELS else 0

    print(f"Model: {MODEL_ID}")
    print(f"Running {args.episodes} episodes, starting difficulty={difficulty}")
    if args.adaptive:
        print(f"Adaptive mode: level up after {STREAK_TO_ADVANCE} wins or a fast solve (<={FAST_SOLVE_THRESHOLD} steps)")

    all_results = []
    for i in range(args.episodes):
        print(f"\n{'─'*60}")
        print(f"Episode {i+1}/{args.episodes}  |  difficulty={difficulty}  |  win_streak={win_streak}")
        print(f"{'─'*60}")

        result = run_episode(client, difficulty, judge_client=judge_client)
        result["difficulty"] = difficulty
        all_results.append(result)

        if args.adaptive:
            won = result["total_reward"] > 0
            fast = won and result["steps"] <= FAST_SOLVE_THRESHOLD

            if won:
                win_streak += 1
                print(f"\n  >> WIN (streak: {win_streak}/{STREAK_TO_ADVANCE})")
            else:
                win_streak = 0
                print(f"\n  >> LOSS (streak reset)")

            should_advance = (win_streak >= STREAK_TO_ADVANCE) or fast

            if should_advance and level_idx < len(DIFFICULTY_LEVELS) - 1:
                level_idx += 1
                difficulty = DIFFICULTY_LEVELS[level_idx]
                win_streak = 0
                if fast:
                    print(f"  >> FAST SOLVE! Level up to difficulty {difficulty}")
                else:
                    print(f"  >> {STREAK_TO_ADVANCE} wins! Level up to difficulty {difficulty}")

    # Save trajectories
    with open(args.output, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    # Summary by difficulty
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for lvl in DIFFICULTY_LEVELS:
        lvl_results = [r for r in all_results if r["difficulty"] == lvl]
        if not lvl_results:
            continue
        rewards = [r["total_reward"] for r in lvl_results]
        wins = sum(1 for r in rewards if r > 0)
        avg_steps = sum(r["steps"] for r in lvl_results) / len(lvl_results)
        avg_reasoning = sum(r.get("reasoning_score", 0) for r in lvl_results) / len(lvl_results)
        print(f"  Difficulty {lvl}: {wins}/{len(lvl_results)} wins, "
              f"avg reward={sum(rewards)/len(rewards):.1f}, avg steps={avg_steps:.1f}, "
              f"avg reasoning={avg_reasoning:.2f}")
    print(f"\n  Total episodes: {len(all_results)}")
    print(f"  Max difficulty reached: {max(r['difficulty'] for r in all_results)}")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()

"""GRPO training for HFT Oversight agent using TRL + OpenEnv.

Uses the official TRL rollout_func pattern with GRPOTrainer.
The model generates actions, the environment scores them via step(),
and GRPO optimizes toward higher-reward actions.

Usage:
    # Collect baseline (free, no GPU needed)
    uv run python train.py --baseline-only

    # Full GRPO training (needs 1 GPU, colocate mode)
    python train.py

    # With separate vLLM server (2+ GPUs)
    # Terminal 1: CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-7B-Instruct
    # Terminal 2: CUDA_VISIBLE_DEVICES=1 python train.py --vllm-mode server --vllm-server-url http://localhost:8000
"""

import json
import os
import sys
import argparse
import copy

sys.path.insert(0, os.path.dirname(__file__))

from server.environment import HFTOversightEnvironment
from openenv.core.env_server.types import State
from models import OversightAction

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = (
    "You are an HFT oversight agent. Read each bot's activity logs to find anomalous trading patterns. "
    "Flag suspicious bots, then shut them down with a reason. "
    "Respond with ONLY a JSON action each turn."
)


def parse_action(text: str) -> OversightAction:
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1].removeprefix("json").strip()
    start = text.find("{")
    if start < 0:
        raise ValueError(f"Could not parse: {text}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                data = json.loads(text[start:i + 1])
                return OversightAction(**data)
    raise ValueError(f"Could not parse: {text}")


# ---------------------------------------------------------------------------
# Environment rollout: run one full episode, return reward
# ---------------------------------------------------------------------------

def run_episode(completion_text: str, difficulty: int, scenario: dict) -> dict:
    """Replay a completion against a forked environment and return rewards."""
    env = HFTOversightEnvironment()
    env._difficulty = difficulty
    env._scenario = scenario
    env._state = State(episode_id="eval", step_count=0)
    env._bots_flagged = []
    env._bots_shutdown = []
    env._bad_bots = list(scenario["bad_bots"])
    env._damage = 0.0

    obs = env.reset()

    total_reward = 0.0
    used_cross_ref = False
    investigation_steps = 0
    wrong_shutdowns = 0
    correct_shutdowns = 0
    parse_failures = 0
    total_lines = 0
    steps_to_first_correct = None
    targeted_bad_bot = False
    shutdown_reasons = []

    bad_bot_ids = set(scenario["bad_bots"])
    bot_ids = set(scenario["bots"].keys())

    # The completion may be a single action or multi-line
    lines = completion_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        total_lines += 1

        try:
            action = parse_action(line)
        except Exception:
            parse_failures += 1
            continue

        if action.command in ("read_logs", "check_pnl", "read_file", "cross_reference"):
            investigation_steps += 1
            # Did the agent investigate a bad bot specifically?
            if action.bot_id and action.bot_id in bad_bot_ids:
                targeted_bad_bot = True
        if action.command == "cross_reference":
            used_cross_ref = True

        obs = env.step(action)
        total_reward += obs.reward

        if action.command == "shutdown":
            if obs.reward > 0:
                correct_shutdowns += 1
                if steps_to_first_correct is None:
                    steps_to_first_correct = env._state.step_count
                # Capture reason + issue for LLM judge
                shutdown_reasons.append({
                    "bot_id": action.bot_id,
                    "reason": action.reason or "",
                    "issue_type": scenario.get("issue", ""),
                    "correct": True,
                })
            elif obs.reward < 0:
                wrong_shutdowns += 1
                shutdown_reasons.append({
                    "bot_id": action.bot_id,
                    "reason": action.reason or "",
                    "issue_type": scenario.get("issue", ""),
                    "correct": False,
                })

        if obs.done:
            break

    return {
        "total_reward": total_reward,
        "used_cross_ref": used_cross_ref,
        "investigation_steps": investigation_steps,
        "correct_shutdowns": correct_shutdowns,
        "wrong_shutdowns": wrong_shutdowns,
        "parse_failures": parse_failures,
        "total_lines": total_lines,
        "steps_to_first_correct": steps_to_first_correct,
        "max_timesteps": env._max_timesteps,
        "final_step": env._state.step_count,
        "targeted_bad_bot": targeted_bad_bot,
        "shutdown_reasons": shutdown_reasons,
    }


# ---------------------------------------------------------------------------
# Build prompts dataset from environment scenarios
# ---------------------------------------------------------------------------

def build_prompt_dataset(num_prompts: int = 64, difficulties: list = None):
    """Generate environment prompts for GRPO training."""
    from datasets import Dataset

    if difficulties is None:
        difficulties = [1, 3, 5, 7]

    prompts = []
    scenarios = []

    for i in range(num_prompts):
        difficulty = difficulties[i % len(difficulties)]
        env = HFTOversightEnvironment()
        env._difficulty = difficulty
        obs = env.reset()

        prompt_text = obs.response
        if obs.alerts:
            prompt_text += f"\n\nAlerts: {obs.alerts}"

        # Build the chat prompt
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(formatted)

        scenarios.append({
            "difficulty": difficulty,
            "scenario": env._scenario,
        })

    dataset = Dataset.from_dict({"prompt": prompts})
    return dataset, scenarios


# ---------------------------------------------------------------------------
# Rollout function: TRL OpenEnv pattern
# ---------------------------------------------------------------------------

def make_rollout_func(scenarios: list, use_judge: bool = True):
    """Create a rollout function that steps through the HFT environment."""
    from trl.experimental.openenv import generate_rollout_completions

    # Map prompts to scenarios for lookup
    scenario_by_idx = {i: s for i, s in enumerate(scenarios)}
    call_count = [0]

    # Lazy-init judge client (only when first needed)
    judge_client = [None]

    def _get_judge_client():
        if judge_client[0] is None and use_judge:
            from huggingface_hub import InferenceClient, get_token
            judge_client[0] = InferenceClient(model=JUDGE_MODEL, token=get_token())
        return judge_client[0]

    def rollout_func(prompts: list[str], trainer):
        outputs = generate_rollout_completions(trainer, prompts)
        tokenizer = trainer.processing_class

        env_rewards = []
        cross_ref_bonuses = []
        investigation_bonuses = []
        format_rewards = []
        speed_rewards = []
        targeting_rewards = []
        reasoning_rewards = []

        for i, out in enumerate(outputs):
            completion_text = tokenizer.decode(out["completion_ids"], skip_special_tokens=True)

            # Find the scenario for this prompt
            # In GRPO, prompts repeat num_generations times
            prompt_idx = (call_count[0] * len(prompts) + i) % len(scenarios)
            scenario_data = scenario_by_idx.get(prompt_idx, scenario_by_idx[0])

            result = run_episode(
                completion_text,
                scenario_data["difficulty"],
                scenario_data["scenario"],
            )

            env_rewards.append(result["total_reward"])

            # Reward shaping: encourage investigation before action
            cross_ref_bonuses.append(2.0 if result["used_cross_ref"] else 0.0)
            investigation_bonuses.append(min(result["investigation_steps"] * 0.5, 3.0))

            # Format reward: penalize unparseable output, reward clean JSON
            if result["total_lines"] == 0:
                format_rewards.append(-2.0)
            else:
                parse_rate = 1.0 - (result["parse_failures"] / result["total_lines"])
                format_rewards.append(parse_rate * 2.0 - 1.0)  # [-1.0, 1.0]

            # Speed reward: bonus for fast correct shutdowns
            if result["steps_to_first_correct"] is not None:
                speed_ratio = 1.0 - (result["steps_to_first_correct"] / result["max_timesteps"])
                speed_rewards.append(speed_ratio * 3.0)  # up to +3.0 for instant solve
            else:
                speed_rewards.append(-1.0)  # penalty for never finding the bad bot

            # Targeting reward: did the agent read logs of the actual bad bot?
            targeting_rewards.append(1.5 if result["targeted_bad_bot"] else 0.0)

            # LLM-as-judge reasoning reward
            if use_judge and result["shutdown_reasons"]:
                client = _get_judge_client()
                if client:
                    reasoning_rewards.append(judge_shutdown_reasons(result["shutdown_reasons"], client))
                else:
                    reasoning_rewards.append(0.0)
            else:
                reasoning_rewards.append(0.0)

        call_count[0] += 1

        return {
            "prompt_ids": [out["prompt_ids"] for out in outputs],
            "completion_ids": [out["completion_ids"] for out in outputs],
            "logprobs": [out["logprobs"] for out in outputs],
            "env_reward": env_rewards,
            "cross_ref_bonus": cross_ref_bonuses,
            "investigation_bonus": investigation_bonuses,
            "format_reward": format_rewards,
            "speed_reward": speed_rewards,
            "targeting_reward": targeting_rewards,
            "reasoning_reward": reasoning_rewards,
        }

    return rollout_func


# ---------------------------------------------------------------------------
# Reward functions (passed to GRPOTrainer)
# ---------------------------------------------------------------------------

def reward_env(completions, **kwargs):
    """Primary reward: environment step rewards (shutdowns, penalties, time pressure)."""
    rewards = kwargs.get("env_reward", [])
    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)


def reward_cross_ref(completions, **kwargs):
    """Bonus for using cross_reference — the key skill for cross-module detection."""
    rewards = kwargs.get("cross_ref_bonus", [])
    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)


def reward_investigation(completions, **kwargs):
    """Small bonus for investigating before shutting down."""
    rewards = kwargs.get("investigation_bonus", [])
    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)


def reward_format(completions, **kwargs):
    """Reward clean JSON output, penalize unparseable responses."""
    rewards = kwargs.get("format_reward", [])
    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)


def reward_speed(completions, **kwargs):
    """Reward fast correct shutdowns — solving in fewer steps scores higher."""
    rewards = kwargs.get("speed_reward", [])
    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)


def reward_targeting(completions, **kwargs):
    """Reward investigating the actual bad bot's logs (not just random bots)."""
    rewards = kwargs.get("targeting_reward", [])
    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)


def reward_reasoning(completions, **kwargs):
    """LLM-as-judge reward for shutdown reasoning quality."""
    rewards = kwargs.get("reasoning_reward", [])
    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)


# ---------------------------------------------------------------------------
# LLM-as-judge for shutdown reasoning
# ---------------------------------------------------------------------------

JUDGE_MODEL = "Qwen/Qwen2.5-72B-Instruct"

JUDGE_PROMPT = """You are evaluating an HFT oversight agent's reasoning when shutting down a trading bot.

The bot had this issue: {issue_type}
The agent's shutdown reason: "{reason}"
The shutdown was: {correctness}

Score the reasoning quality from 0 to 5:
- 0: No reasoning, empty, or completely wrong
- 1: Vague or generic (e.g. "looks suspicious")
- 2: Identifies something is wrong but misdiagnoses the specific issue
- 3: Correctly identifies the issue category but lacks specifics
- 4: Good diagnosis with specific evidence from logs (prices, patterns, timing)
- 5: Excellent — names the exact issue, cites specific data points, explains the mechanism

Respond with ONLY a JSON object: {{"score": <0-5>, "explanation": "<brief reason>"}}"""


def judge_shutdown_reasons(shutdown_reasons: list[dict], client) -> float:
    """Call LLM judge on shutdown reasons, return average score mapped to [0, 3]."""
    if not shutdown_reasons:
        return 0.0

    scores = []
    for sr in shutdown_reasons:
        if not sr["reason"]:
            scores.append(0.0)
            continue

        correctness = "CORRECT (bot was malfunctioning)" if sr["correct"] else "INCORRECT (bot was healthy)"
        prompt = JUDGE_PROMPT.format(
            issue_type=sr["issue_type"],
            reason=sr["reason"],
            correctness=correctness,
        )

        try:
            resp = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.0,
            )
            text = resp.choices[0].message.content.strip()
            # Parse score from JSON response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                score = float(data.get("score", 0))
                score = max(0.0, min(5.0, score))
                # Wrong shutdown with good reasoning still gets penalized
                if not sr["correct"]:
                    score = -score * 0.5
                scores.append(score)
            else:
                scores.append(0.0)
        except Exception:
            scores.append(0.0)

    # Average score, mapped from [0-5] to [0-3] (matching _check_diagnosis scale)
    avg = sum(scores) / len(scores)
    return avg * 0.6


# ---------------------------------------------------------------------------
# Baseline collection (uses HF Inference API, no GPU)
# ---------------------------------------------------------------------------

def collect_baseline(episodes_per_level: int = 10, output_dir: str = "data"):
    from huggingface_hub import InferenceClient, get_token

    client = InferenceClient(model=MODEL_ID, token=get_token())
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for difficulty in [1, 3, 5, 7]:
        print(f"\nBaseline: difficulty {difficulty} ({episodes_per_level} episodes)")
        for ep in range(episodes_per_level):
            env = HFTOversightEnvironment()
            env._difficulty = difficulty
            obs = env.reset()

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs.response + (f"\n\nAlerts: {obs.alerts}" if obs.alerts else "")},
            ]

            total_reward = 0.0
            while not obs.done:
                try:
                    resp = client.chat_completion(messages=messages, max_tokens=200, temperature=0.3)
                    llm_text = resp.choices[0].message.content
                except Exception:
                    llm_text = '{"command": "pass_turn"}'

                try:
                    action = parse_action(llm_text)
                except Exception:
                    action = OversightAction(command="pass_turn")

                obs = env.step(action)
                total_reward += obs.reward
                messages.append({"role": "assistant", "content": llm_text})
                env_msg = obs.response
                if obs.alerts:
                    env_msg += f"\n\nAlerts: {obs.alerts}"
                env_msg += f"\n\n[Step {obs.timestep}/{obs.max_timesteps}]"
                messages.append({"role": "user", "content": env_msg})

            won = total_reward > 0
            all_results.append({"difficulty": difficulty, "total_reward": total_reward, "won": won})
            print(f"  ep {ep+1}: {'WIN' if won else 'LOSS'} reward={total_reward:.1f}")

    out_path = os.path.join(output_dir, "baseline.jsonl")
    with open(out_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    print(f"\nBaseline saved to {out_path}")
    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GRPO training for HFT Oversight")
    parser.add_argument("--model", type=str, default=MODEL_ID)
    parser.add_argument("--baseline-episodes", type=int, default=10)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default="checkpoints/hft-oversight-grpo")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--use-vllm", action="store_true", default=True, help="Use vLLM for generation")
    parser.add_argument("--no-vllm", dest="use_vllm", action="store_false")
    parser.add_argument("--vllm-mode", type=str, default="colocate", choices=["colocate", "server"])
    parser.add_argument("--vllm-server-url", type=str, default="http://localhost:8000")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--no-judge", action="store_true", help="Disable LLM-as-judge reasoning reward")
    parser.add_argument("--push-to-hub", type=str, default=None,
                        help="HF Hub repo to push checkpoint (e.g. seanchangg/hft-oversight-grpo)")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Episodes per difficulty for post-training eval")
    args = parser.parse_args()

    # Phase 1: Baseline
    baseline = {}
    if not args.skip_baseline:
        print("\n" + "=" * 60)
        print("PHASE 1: Baseline (base model, no training)")
        print("=" * 60)
        all_results = collect_baseline(args.baseline_episodes, args.data_dir)
        for d in [1, 3, 5, 7]:
            d_results = [r for r in all_results if r["difficulty"] == d]
            if d_results:
                baseline[d] = {
                    "win_rate": sum(1 for r in d_results if r["won"]) / len(d_results),
                    "avg_reward": sum(r["total_reward"] for r in d_results) / len(d_results),
                }

        print("\nBaseline results:")
        for d in [1, 3, 5, 7]:
            if d in baseline:
                print(f"  Difficulty {d}: win_rate={baseline[d]['win_rate']:.0%}, avg_reward={baseline[d]['avg_reward']:.1f}")

    if args.baseline_only:
        return

    # Phase 2: GRPO Training
    print("\n" + "=" * 60)
    print("PHASE 2: GRPO Training")
    print("=" * 60)

    from trl import GRPOConfig, GRPOTrainer
    from transformers import AutoTokenizer
    from peft import LoraConfig

    print(f"Building {args.num_prompts} environment prompts...")
    dataset, scenarios = build_prompt_dataset(args.num_prompts)

    rollout_func = make_rollout_func(scenarios, use_judge=not args.no_judge)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    grpo_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
    )
    if args.use_vllm:
        grpo_kwargs["use_vllm"] = True
        grpo_kwargs["vllm_mode"] = args.vllm_mode
        if args.vllm_mode == "server":
            grpo_kwargs["vllm_server_base_url"] = args.vllm_server_url

    grpo_config = GRPOConfig(**grpo_kwargs)

    print(f"  Model: {args.model}")
    print(f"  Prompts: {len(dataset)}")
    print(f"  Generations per prompt: {args.num_generations}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  vLLM mode: {args.vllm_mode}")
    print(f"  Rewards: env + cross_ref + investigation + format + speed + targeting + reasoning")

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[
            reward_env,
            reward_cross_ref,
            reward_investigation,
            reward_format,
            reward_speed,
            reward_targeting,
            reward_reasoning,
        ],
        train_dataset=dataset,
        rollout_func=rollout_func,
        peft_config=lora_config,
        args=grpo_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")

    # Push to HF Hub
    if args.push_to_hub:
        print(f"\nPushing checkpoint to {args.push_to_hub}...")
        trainer.push_to_hub(args.push_to_hub)
        print(f"Pushed to https://huggingface.co/{args.push_to_hub}")

    # Phase 3: Post-training evaluation
    print("\n" + "=" * 60)
    print("PHASE 3: Post-training Evaluation")
    print("=" * 60)

    trained = {}
    trained_results = eval_trained_model(args.output_dir, args.eval_episodes)
    for d in [1, 3, 5, 7]:
        d_results = [r for r in trained_results if r["difficulty"] == d]
        if d_results:
            trained[d] = {
                "win_rate": sum(1 for r in d_results if r["won"]) / len(d_results),
                "avg_reward": sum(r["total_reward"] for r in d_results) / len(d_results),
            }

    # Phase 4: Comparison
    print("\n" + "=" * 60)
    print("COMPARISON: Baseline vs Trained")
    print("=" * 60)
    print(f"{'Difficulty':<12} {'Base WR':<10} {'Train WR':<10} {'Base Reward':<14} {'Train Reward':<14}")
    print("-" * 60)
    for d in [1, 3, 5, 7]:
        bwr = f"{baseline[d]['win_rate']:.0%}" if d in baseline else "N/A"
        twr = f"{trained[d]['win_rate']:.0%}" if d in trained else "N/A"
        brw = f"{baseline[d]['avg_reward']:.1f}" if d in baseline else "N/A"
        trw = f"{trained[d]['avg_reward']:.1f}" if d in trained else "N/A"
        print(f"{d:<12} {bwr:<10} {twr:<10} {brw:<14} {trw:<14}")

    # Save comparison JSON
    comparison = {"baseline": baseline, "trained": trained}
    comp_path = os.path.join(args.data_dir, "comparison.json")
    os.makedirs(args.data_dir, exist_ok=True)
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved to {comp_path}")

    # Generate graph
    plot_path = os.path.join(args.data_dir, "comparison.png")
    plot_comparison(baseline, trained, plot_path)


def eval_trained_model(checkpoint_dir: str, episodes_per_level: int = 10) -> list:
    """Run the trained model against the environment and collect results."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Loading trained model from {checkpoint_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    all_results = []

    for difficulty in [1, 3, 5, 7]:
        print(f"\nEval trained model: difficulty {difficulty} ({episodes_per_level} episodes)")
        for ep in range(episodes_per_level):
            env = HFTOversightEnvironment()
            env._difficulty = difficulty
            obs = env.reset()

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs.response + (f"\n\nAlerts: {obs.alerts}" if obs.alerts else "")},
            ]

            total_reward = 0.0
            while not obs.done:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs, max_new_tokens=200, temperature=0.3,
                        do_sample=True, pad_token_id=tokenizer.eos_token_id,
                    )
                new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
                llm_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

                try:
                    action = parse_action(llm_text)
                except Exception:
                    action = OversightAction(command="pass_turn")

                obs = env.step(action)
                total_reward += obs.reward
                messages.append({"role": "assistant", "content": llm_text})
                env_msg = obs.response
                if obs.alerts:
                    env_msg += f"\n\nAlerts: {obs.alerts}"
                env_msg += f"\n\n[Step {obs.timestep}/{obs.max_timesteps}]"
                messages.append({"role": "user", "content": env_msg})

            won = total_reward > 0
            all_results.append({"difficulty": difficulty, "total_reward": total_reward, "won": won})
            print(f"  ep {ep+1}: {'WIN' if won else 'LOSS'} reward={total_reward:.1f}")

    return all_results


def plot_comparison(baseline: dict, trained: dict, save_path: str):
    """Generate a comparison bar chart and save as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping graph generation")
        return

    difficulties = [1, 3, 5, 7]
    base_wr = [baseline.get(d, {}).get("win_rate", 0) * 100 for d in difficulties]
    train_wr = [trained.get(d, {}).get("win_rate", 0) * 100 for d in difficulties]
    base_rw = [baseline.get(d, {}).get("avg_reward", 0) for d in difficulties]
    train_rw = [trained.get(d, {}).get("avg_reward", 0) for d in difficulties]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(difficulties))
    width = 0.35

    # Win rate chart
    ax1.bar(x - width/2, base_wr, width, label="Baseline", color="#4a90d9")
    ax1.bar(x + width/2, train_wr, width, label="GRPO Trained", color="#e8744f")
    ax1.set_xlabel("Difficulty")
    ax1.set_ylabel("Win Rate (%)")
    ax1.set_title("Win Rate: Baseline vs Trained")
    ax1.set_xticks(x)
    ax1.set_xticklabels(difficulties)
    ax1.legend()
    ax1.set_ylim(0, 105)

    # Avg reward chart
    ax2.bar(x - width/2, base_rw, width, label="Baseline", color="#4a90d9")
    ax2.bar(x + width/2, train_rw, width, label="GRPO Trained", color="#e8744f")
    ax2.set_xlabel("Difficulty")
    ax2.set_ylabel("Avg Reward")
    ax2.set_title("Avg Reward: Baseline vs Trained")
    ax2.set_xticks(x)
    ax2.set_xticklabels(difficulties)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Comparison graph saved to {save_path}")


if __name__ == "__main__":
    main()

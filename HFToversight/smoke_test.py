"""Quick smoke test — verifies everything imports and the env runs.
No GPU, no HF token, no network needed.

Usage: python3 smoke_test.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_env():
    from server.environment import HFTOversightEnvironment
    from models import OversightAction
    from scenarios import generate_scenario

    # Test scenario generation with issue_type
    for diff in [1, 3, 5, 7]:
        s = generate_scenario(diff)
        bad_id = s["bad_bots"][0]
        issue = s["bots"][bad_id].get("issue_type")
        assert issue is not None, f"bad bot missing issue_type at diff={diff}"
        for bot_id, bot in s["bots"].items():
            if bot_id not in s["bad_bots"]:
                assert bot["issue_type"] is None, f"normal bot {bot_id} has issue_type"
    print("[OK] scenarios: issue_type set correctly")

    # Test environment step cycle
    env = HFTOversightEnvironment()
    env._difficulty = 1
    obs = env.reset()
    assert "read_logs" in obs.response
    assert "inspect_config" not in obs.response
    print("[OK] reset: no inspect_config in commands")

    bots = list(env._scenario["bots"].keys())

    # Test read_logs with time filter
    obs = env.step(OversightAction(command="read_logs", bot_id=bots[0], start_time="14:01:00", end_time="14:02:00"))
    assert "14:01:00 - 14:02:00" in obs.response
    print("[OK] read_logs: time filter works")

    # Test dynamic PnL
    obs1 = env.step(OversightAction(command="check_pnl", bot_id=bots[0]))
    obs2 = env.step(OversightAction(command="check_pnl", bot_id=bots[0]))
    pnl1 = [l for l in obs1.response.split("\n") if "PnL" in l][0]
    pnl2 = [l for l in obs2.response.split("\n") if "PnL" in l][0]
    assert pnl1 != pnl2, "PnL should change between steps"
    print("[OK] check_pnl: dynamic PnL")

    # Test inspect_config removed
    obs = env.step(OversightAction(command="inspect_config", bot_id=bots[0]))
    assert "Unknown command" in obs.response
    print("[OK] inspect_config: removed")

    # Test full episode can complete
    env2 = HFTOversightEnvironment()
    env2._difficulty = 1
    obs = env2.reset()
    bad = env2._bad_bots[0]
    obs = env2.step(OversightAction(command="read_logs", bot_id=bad))
    obs = env2.step(OversightAction(command="flag", bot_id=bad, reason="test"))
    obs = env2.step(OversightAction(command="shutdown", bot_id=bad, reason="losing money on round trips"))
    assert obs.done
    assert obs.reward > 0
    print("[OK] full episode: flag + shutdown works")


def test_train_imports():
    """Test that train.py functions are importable (no GPU needed)."""
    from train import (
        parse_action, run_episode, build_prompt_dataset,
        judge_shutdown_reasons, JUDGE_PROMPT, JUDGE_MODEL,
        reward_env, reward_reasoning, plot_comparison,
    )
    print("[OK] train.py: all functions importable")

    # Test judge prompt formatting
    sr = {"bot_id": "x", "reason": "test", "issue_type": "pnl_bleed", "correct": True}
    prompt = JUDGE_PROMPT.format(issue_type=sr["issue_type"], reason=sr["reason"], correctness="CORRECT")
    assert "pnl_bleed" in prompt
    print("[OK] judge prompt: formats correctly")


if __name__ == "__main__":
    print("Running smoke tests...\n")
    test_env()
    print()
    try:
        test_train_imports()
    except ImportError as e:
        print(f"[SKIP] train imports (missing deps): {e}")
    print("\nAll smoke tests passed!")

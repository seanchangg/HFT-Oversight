"""Quick local test — runs the environment directly, no server needed."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from server.environment import HFTOversightEnvironment
from models import OversightAction


def run_test(difficulty: int = 1):
    env = HFTOversightEnvironment()
    env._difficulty = difficulty
    obs = env.reset()

    print(f"{'='*60}")
    print(f"DIFFICULTY {difficulty} | Issue: {env._scenario['issue']}")
    print(f"Bad bots: {env._bad_bots}")
    print(f"{'='*60}")
    print(obs.response)
    if obs.alerts:
        print(f"\nAlerts: {obs.alerts}")

    # Step 1: list bots
    obs = env.step(OversightAction(command="list_bots"))
    print(f"\n--- list_bots ---\n{obs.response}")

    # Step 2: read logs for each bot
    for bot_id in env._scenario["bots"]:
        obs = env.step(OversightAction(command="read_logs", bot_id=bot_id))
        print(f"\n--- read_logs {bot_id} (first 5 lines) ---")
        for line in obs.response.split("\n")[:6]:
            print(line)
        print(f"  reward={obs.reward:.2f}  step={obs.timestep}/{obs.max_timesteps}")
        if obs.done:
            break

    if not obs.done:
        # Step 3: cross-reference bad bots if it's a cross-module issue
        bot_ids = list(env._scenario["bots"].keys())
        if len(bot_ids) >= 2:
            obs = env.step(OversightAction(
                command="cross_reference",
                bot_id=bot_ids[0],
                bot_id_2=bot_ids[1],
            ))
            print(f"\n--- cross_reference {bot_ids[0]} vs {bot_ids[1]} (first 8 lines) ---")
            for line in obs.response.split("\n")[:9]:
                print(line)

    if not obs.done:
        # Step 4: flag then shutdown the actual bad bot
        bad_bot = env._bad_bots[0]
        obs = env.step(OversightAction(
            command="flag",
            bot_id=bad_bot,
            reason="Anomalous trading pattern detected",
        ))
        print(f"\n--- flag {bad_bot} ---")
        print(f"  {obs.response}")
        print(f"  reward={obs.reward:.2f}")

        obs = env.step(OversightAction(
            command="shutdown",
            bot_id=bad_bot,
            reason="Bot showing anomalous pattern — shutting down to prevent damage",
        ))
        print(f"\n--- shutdown {bad_bot} ---")
        print(f"  {obs.response}")
        print(f"  reward={obs.reward:.2f}  done={obs.done}")

    print(f"\n{'='*60}")
    print(f"Episode complete. Final step: {obs.timestep}/{obs.max_timesteps}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    for d in [1, 3, 5, 7]:
        run_test(d)

"""HFT Oversight Environment Implementation.

An LLM oversight agent investigates a fleet of trading bots,
reads logs, and identifies/shuts down problematic ones.
"""

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import OversightAction, OversightObservation
from scenarios import generate_scenario


class HFTOversightEnvironment(Environment):
    """
    Environment where an LLM agent oversees a fleet of HFT bots.

    The agent starts with a list of bots it manages, then investigates
    by reading logs, checking stats, and inspecting configs. It must
    identify and shut down malfunctioning bots.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._scenario = None
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._difficulty = 1
        self._bots_flagged: list[str] = []
        self._bots_shutdown: list[str] = []
        self._bad_bots: list[str] = []
        self._bots_investigated: dict[str, int] = {}  # bot_id -> number of investigation actions
        self._damage: float = 0.0
        self._max_timesteps = 10
        # Progressive difficulty (Statement 4: Self-Improvement)
        self._consecutive_wins = 0
        self._consecutive_losses = 0
        self._episode_count = 0
        self._wins_at_level: dict[int, int] = {}
        self._episodes_at_level: dict[int, int] = {}

    # Keywords that indicate the agent understood the actual issue
    DIAGNOSIS_KEYWORDS: dict[str, list[str]] = {
        "pnl_bleed": ["loss", "losing", "bleed", "round-trip", "buy high", "sell low"],
        "latency_arb": ["slippage", "fill price", "worse", "latency", "adverse"],
        "position_runaway": ["position", "one direction", "only buy", "only sell", "monoton"],
        "stale_quotes": ["stale", "same price", "not updating", "repeating", "stuck"],
        "order_stuffing": ["stuffing", "burst", "rapid", "cancel", "flood"],
        "config_drift": ["wrong symbol", "config", "mismatch", "different symbol"],
        "wash_trading": ["wash", "matched", "opposite side", "same qty", "same time"],
        "counter_trading": ["mirror", "counter", "delay", "opposite", "following"],
    }

    def reset(self) -> OversightObservation:
        self._scenario = generate_scenario(self._difficulty)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._bots_flagged = []
        self._bots_shutdown = []
        self._bots_investigated = {}
        self._bad_bots = list(self._scenario["bad_bots"])
        self._damage = 0.0

        # Initial observation: list bots and guide toward log investigation
        bot_names = list(self._scenario["bots"].keys())
        lines = [
            "You are an oversight agent. One or more bots may be malfunctioning.",
            "Read their activity logs to identify anomalous trading patterns.",
            "Flag suspicious bots, then shut them down with a reason.",
            "",
            f"You have {self._max_timesteps} actions before the episode ends.",
            "",
            "Bots under your management:",
        ]
        for name in bot_names:
            bot = self._scenario["bots"][name]
            lines.append(f"  - {name} ({bot['strategy_type']})")
        lines.append("")
        lines.append("Commands: read_logs, check_pnl, read_file, cross_reference, flag, shutdown")

        return OversightObservation(
            response="\n".join(lines),
            timestep=0,
            max_timesteps=self._max_timesteps,
            alerts=self._generate_alerts(),
            done=False,
            reward=0.0,
        )

    def step(self, action: OversightAction) -> OversightObservation:
        self._state.step_count += 1
        bots = self._scenario["bots"]
        reward = 0.0

        # Update PnL each step so check_pnl shows changing values
        self._update_pnl(bots)

        # Damage accumulates each step for each undetected bad bot
        for bad_id in self._bad_bots:
            if bad_id not in self._bots_shutdown:
                self._damage += 1.0

        # Track investigation depth per bot
        if action.command in ("read_logs", "read_file", "check_pnl") and action.bot_id:
            self._bots_investigated[action.bot_id] = self._bots_investigated.get(action.bot_id, 0) + 1
        if action.command == "cross_reference":
            if action.bot_id:
                self._bots_investigated[action.bot_id] = self._bots_investigated.get(action.bot_id, 0) + 1
            if action.bot_id_2:
                self._bots_investigated[action.bot_id_2] = self._bots_investigated.get(action.bot_id_2, 0) + 1

        response = self._execute_command(action, bots)

        # Small reward for investigating the bad bot (agent doesn't know which is bad)
        if action.command in ("read_logs", "read_file", "cross_reference"):
            investigated_id = action.bot_id
            if investigated_id and investigated_id in self._bad_bots:
                depth = self._bots_investigated.get(investigated_id, 0)
                if depth <= 3:
                    reward = 1.0 / depth

        # Handle flag — low-risk probe before committing to shutdown
        if action.command == "flag":
            bot_id = action.bot_id
            if bot_id and bot_id in self._bots_flagged:
                reward = -1.0  # already flagged, wasting time
            elif bot_id and bot_id in self._bad_bots:
                reward = 2.0
                self._bots_flagged.append(bot_id)
            elif bot_id:
                reward = -2.0
                self._bots_flagged.append(bot_id)

        # Handle shutdown
        if action.command == "shutdown":
            bot_id = action.bot_id
            if bot_id and bot_id in self._bots_shutdown:
                reward = 0.0
            elif bot_id and bot_id in self._bad_bots:
                depth = self._bots_investigated.get(bot_id, 0)
                was_flagged = bot_id in self._bots_flagged

                # Correct shutdown: ~30% answer, ~70% process
                reward = 3.0

                # Investigation depth bonus (up to +4.0)
                reward += min(depth, 2) * 2.0

                # Flagged first bonus
                if was_flagged:
                    reward += 2.0

                # Diagnosis bonus (up to +3.0)
                diagnosis_bonus = self._check_diagnosis(action.reason or "")
                reward += diagnosis_bonus

                speed_bonus = max(0, (self._max_timesteps - self._state.step_count)) * 0.2
                reward += speed_bonus
                self._bots_shutdown.append(bot_id)
            elif bot_id:
                reward = -10.0
                self._bots_shutdown.append(bot_id)

        # Check done
        all_bad_found = all(b in self._bots_shutdown for b in self._bad_bots)
        out_of_time = self._state.step_count >= self._max_timesteps
        done = all_bad_found or out_of_time

        if out_of_time and not all_bad_found:
            reward -= 5.0
            response += "\n\nTIME UP: Malfunctioning bot(s) still running."

        if all_bad_found and done:
            # Episode-end summary reward: bonus for low damage
            damage_ratio = self._damage / max(1, self._max_timesteps * len(self._bad_bots))
            efficiency_bonus = (1.0 - damage_ratio) * 5.0
            reward += efficiency_bonus
            response += f"\n\nAll malfunctioning bots shut down. Efficiency bonus: +{efficiency_bonus:.1f}"

        # Update progressive difficulty when episode ends
        if done:
            self._update_difficulty(all_bad_found)
            response += f"\n[Difficulty: {self._difficulty}]"

        return OversightObservation(
            response=response,
            timestep=self._state.step_count,
            max_timesteps=self._max_timesteps,
            alerts=self._generate_alerts(),
            done=done,
            reward=reward,
        )

    def _update_difficulty(self, won: bool):
        """Progressive difficulty: advance on wins, regress on losses."""
        self._episode_count += 1
        level = self._difficulty
        self._episodes_at_level[level] = self._episodes_at_level.get(level, 0) + 1
        if won:
            self._wins_at_level[level] = self._wins_at_level.get(level, 0) + 1
            self._consecutive_wins += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            self._consecutive_wins = 0

        # Advance after 2 consecutive wins at current level
        if self._consecutive_wins >= 2 and self._difficulty < 10:
            self._difficulty = min(10, self._difficulty + 1)
            self._consecutive_wins = 0

        # Regress after 3 consecutive losses
        if self._consecutive_losses >= 3 and self._difficulty > 1:
            self._difficulty = max(1, self._difficulty - 1)
            self._consecutive_losses = 0

    def _execute_command(self, action: OversightAction, bots: dict) -> str:
        cmd = action.command

        if cmd == "list_bots":
            lines = ["Bot fleet status:"]
            for bot_id, bot in bots.items():
                status = bot["status"]
                if bot_id in self._bots_shutdown:
                    status = "SHUTDOWN"
                elif bot_id in self._bots_flagged:
                    status = "FLAGGED"
                lines.append(f"  {bot_id}: {bot['strategy_type']} | status={status}")
            return "\n".join(lines)

        if cmd == "read_logs":
            if not action.bot_id or action.bot_id not in bots:
                return f"Unknown bot: {action.bot_id}"
            logs = bots[action.bot_id]["logs"]
            # Apply optional time range filter
            if action.start_time or action.end_time:
                start = action.start_time or "00:00:00"
                end = action.end_time or "23:59:59"
                logs = [e for e in logs if start <= e["ts"] <= end]
                header = f"=== Logs for {action.bot_id} ({start} - {end}) ==="
            else:
                header = f"=== Logs for {action.bot_id} ==="
            if not logs:
                return f"{header}\n(no entries in range)"
            lines = [header]
            for entry in logs:
                lines.append(f"[{entry['ts']}] [{entry['type'].upper()}] {entry['msg']}")
            return "\n".join(lines)

        if cmd == "check_pnl":
            if not action.bot_id or action.bot_id not in bots:
                return f"Unknown bot: {action.bot_id}"
            bot = bots[action.bot_id]
            return (
                f"=== Stats for {action.bot_id} ===\n"
                f"PnL: ${bot['pnl']:,.2f}\n"
                f"Position: ${bot['position']:,.2f}\n"
                f"Order rate: {bot['order_rate']}/s\n"
                f"Fill rate: {bot['fill_rate']:.0%}\n"
                f"Cancel rate: {bot['cancel_rate']:.0%}"
            )

        if cmd == "read_file":
            if not action.bot_id or action.bot_id not in bots:
                return f"Unknown bot: {action.bot_id}"
            files = bots[action.bot_id]["files"]
            if not action.path or action.path not in files:
                available = ", ".join(files.keys())
                return f"File not found: {action.path}\nAvailable files: {available}"
            return f"=== {action.bot_id}/{action.path} ===\n{files[action.path]}"

        if cmd == "cross_reference":
            if not action.bot_id or action.bot_id not in bots:
                return f"Unknown bot: {action.bot_id}"
            if not action.bot_id_2 or action.bot_id_2 not in bots:
                return f"Unknown bot: {action.bot_id_2}"
            logs_1 = bots[action.bot_id]["logs"]
            logs_2 = bots[action.bot_id_2]["logs"]
            all_logs = (
                [(l["ts"], action.bot_id, l["msg"]) for l in logs_1]
                + [(l["ts"], action.bot_id_2, l["msg"]) for l in logs_2]
            )
            all_logs.sort(key=lambda x: x[0])
            lines = [f"=== Cross-reference: {action.bot_id} vs {action.bot_id_2} ==="]
            for ts, bot_id, msg in all_logs:
                lines.append(f"  [{ts}] {bot_id}: {msg}")
            return "\n".join(lines)

        if cmd == "flag":
            if not action.bot_id or action.bot_id not in bots:
                return f"Unknown bot: {action.bot_id}"
            if action.bot_id in self._bots_flagged:
                return f"{action.bot_id} is already flagged."
            return f"FLAGGED: {action.bot_id} has been flagged for review. Reason: {action.reason}"

        if cmd == "shutdown":
            if not action.bot_id or action.bot_id not in bots:
                return f"Unknown bot: {action.bot_id}"
            if action.bot_id in self._bots_shutdown:
                return f"ALREADY SHUTDOWN: {action.bot_id} was already terminated."
            if action.bot_id in self._bad_bots:
                return f"CORRECT SHUTDOWN: {action.bot_id} has been terminated. Reason: {action.reason}"
            else:
                return f"WRONG SHUTDOWN: {action.bot_id} was operating normally! Reason: {action.reason}"

        return f"Unknown command: {cmd}. Available: list_bots, read_logs, check_pnl, read_file, cross_reference, flag, shutdown"

    def _update_pnl(self, bots: dict):
        """Update PnL each step so check_pnl returns dynamic values."""
        for bot_id, bot in bots.items():
            if bot_id in self._bots_shutdown:
                continue
            issue_type = bot.get("issue_type")
            if issue_type is None:
                # Normal bot: small random walk
                bot["pnl"] += round(random.uniform(-200, 200), 2)
            elif issue_type == "pnl_bleed":
                bot["pnl"] += round(random.uniform(-2000, -500), 2)
            elif issue_type == "latency_arb":
                bot["pnl"] += round(random.uniform(-1500, -300), 2)
            elif issue_type == "position_runaway":
                bot["pnl"] += round(random.uniform(-1200, -200), 2)
            elif issue_type == "order_stuffing":
                bot["pnl"] += round(random.uniform(-800, -100), 2)
            elif issue_type == "counter_trading_victim":
                bot["pnl"] += round(random.uniform(-1500, -400), 2)
            elif issue_type == "counter_trading_parasite":
                bot["pnl"] += round(random.uniform(200, 1000), 2)
            else:
                # Other issue types: moderate drift
                bot["pnl"] += round(random.uniform(-1000, -100), 2)

    def _check_diagnosis(self, reason: str) -> float:
        """Check if the shutdown reason demonstrates understanding of the issue."""
        issue = self._scenario.get("issue", "")
        keywords = self.DIAGNOSIS_KEYWORDS.get(issue, [])
        if not keywords or not reason:
            return 0.0
        reason_lower = reason.lower()
        matches = sum(1 for kw in keywords if kw in reason_lower)
        if matches >= 2:
            return 3.0
        if matches == 1:
            return 1.5
        return 0.0

    def _generate_alerts(self) -> list[str]:
        """Generate system-level alerts without identifying specific bots."""
        if not self._scenario:
            return []
        alerts = []
        # Aggregate fleet-level anomalies — don't reveal which bot
        total_pnl = sum(
            b["pnl"] for bid, b in self._scenario["bots"].items()
            if bid not in self._bots_shutdown
        )
        active_bots = [
            b for bid, b in self._scenario["bots"].items()
            if bid not in self._bots_shutdown
        ]
        high_cancel = any(b["cancel_rate"] > 0.90 for b in active_bots)
        high_order = any(b["order_rate"] > 100 for b in active_bots)

        if total_pnl < -30000:
            alerts.append(f"SYSTEM: Fleet PnL is negative (${total_pnl:,.0f})")
        if high_cancel:
            alerts.append("SYSTEM: Abnormal cancel rate detected in fleet")
        if high_order:
            alerts.append("SYSTEM: Unusually high order throughput detected")
        if any(b["pnl"] < -50000 for b in active_bots):
            alerts.append("SYSTEM: One or more bots reporting significant losses")
        return alerts

    @property
    def state(self) -> State:
        return self._state

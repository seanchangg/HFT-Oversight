"""Procedural scenario generator for the HFT Oversight environment.

Every bot logs the same types of events: orders, fills, cancels, position
snapshots. Bots believe they are operating normally. The oversight agent
must find the anomaly by reading patterns in the data.

Subtlety scales with difficulty:
  - Low difficulty: patterns are consistent (100% of trades show the issue)
  - High difficulty: patterns are statistical (60-70% of trades, mixed with noise)

Issue types:
  - pnl_bleed: round-trip trades lose money (buy high, sell low)
  - latency_arb: fill prices consistently worse than order prices
  - position_runaway: position drifts in one direction
  - stale_quotes: order prices cluster around a fixed value
  - order_stuffing: bursts of rapid orders + cancels
  - config_drift: bot trading a different symbol than config specifies
  - wash_trading: two bots match trades (same symbol, qty, timestamp, opposite sides)
  - counter_trading: one bot's trades mirror another's with a delay
"""

import random
from copy import deepcopy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYMBOLS = ["AAPL", "TSLA", "NVDA", "META", "MSFT", "GOOG", "AMZN", "AMD", "NFLX", "CRM"]

BOT_NAMES = [
    "atlas", "beacon", "citadel", "drift", "echo", "forge", "gamma",
    "nexus", "phantom", "sentinel", "alpha", "vortex", "pulse", "raven",
    "spark", "tidal", "zenith", "onyx", "quartz", "helix",
]

STRATEGIES = ["market_making", "momentum", "stat_arb", "mean_reversion", "rebalancer"]

ISSUE_TYPES = {
    "pnl_bleed":        {"min_diff": 1, "max_diff": 10, "cross_module": False},
    "latency_arb":      {"min_diff": 1, "max_diff": 10, "cross_module": False},
    "position_runaway": {"min_diff": 2, "max_diff": 10, "cross_module": False},
    "stale_quotes":     {"min_diff": 2, "max_diff": 10, "cross_module": False},
    "order_stuffing":   {"min_diff": 3, "max_diff": 10, "cross_module": False},
    "config_drift":     {"min_diff": 3, "max_diff": 10, "cross_module": False},
    "wash_trading":     {"min_diff": 4, "max_diff": 10, "cross_module": True},
    "counter_trading":  {"min_diff": 5, "max_diff": 10, "cross_module": True},
}


# ---------------------------------------------------------------------------
# Subtlety helper
# ---------------------------------------------------------------------------

def _anomaly_rate(difficulty):
    """What fraction of injected events show the anomaly vs look normal.

    Difficulty 1: 100% anomalous (every trade shows the pattern)
    Difficulty 3: ~85%
    Difficulty 5: ~70%
    Difficulty 7: ~60%
    Difficulty 10: ~50%
    """
    return max(0.50, 1.0 - difficulty * 0.05)


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def _ts(hour, minute, second):
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def _gen_timestamps(n, start_hour=14, start_min=0):
    times = []
    sec = start_min * 60
    for _ in range(n):
        sec += random.randint(1, 8)
        m, s = divmod(sec, 60)
        h = start_hour + m // 60
        m = m % 60
        times.append(_ts(h, m, s))
    return times


def _advance_ts(ts, seconds):
    parts = ts.split(":")
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    s += seconds
    if s >= 60: m += s // 60; s = s % 60
    if m >= 60: h += m // 60; m = m % 60
    return _ts(h, m, s)


# ---------------------------------------------------------------------------
# Log generation
# ---------------------------------------------------------------------------

def _gen_normal_logs(symbol, base_price, count=60):
    """Generate realistic trading activity logs."""
    logs = []
    timestamps = _gen_timestamps(count)
    p = base_price
    position = 0
    order_id = random.randint(10000, 99999)

    for ts in timestamps:
        roll = random.random()
        p += round(random.uniform(-0.25, 0.25), 2)

        if roll < 0.30:
            qty = random.choice([100, 200, 300, 500])
            side = random.choice(["BUY", "SELL"])
            order_id += 1
            logs.append({"ts": ts, "type": "order",
                         "msg": f"NEW {side} {symbol} {qty} @ ${p:.2f} oid={order_id}"})
        elif roll < 0.55:
            qty = random.choice([100, 200, 300, 500])
            side = random.choice(["BUY", "SELL"])
            fill_p = round(p + random.uniform(-0.05, 0.05), 2)
            order_id += 1
            logs.append({"ts": ts, "type": "fill",
                         "msg": f"FILL {side} {symbol} {qty} @ ${fill_p:.2f} oid={order_id}"})
            position += qty if side == "BUY" else -qty
        elif roll < 0.70:
            order_id += 1
            logs.append({"ts": ts, "type": "cancel",
                         "msg": f"CXLD oid={order_id} {symbol}"})
        elif roll < 0.85:
            logs.append({"ts": ts, "type": "position",
                         "msg": f"POS {symbol} {position:+d} shares mid=${p:.2f}"})
        else:
            spread = round(random.uniform(0.01, 0.08), 2)
            logs.append({"ts": ts, "type": "quote",
                         "msg": f"QUOTE {symbol} bid=${p:.2f} ask=${p + spread:.2f}"})

    return logs


# ---------------------------------------------------------------------------
# Normal bot
# ---------------------------------------------------------------------------

def _make_normal_bot(bot_id, symbol, base_price, log_count=60):
    strategy = random.choice(STRATEGIES)
    logs = _gen_normal_logs(symbol, base_price, count=log_count)
    pnl = round(random.uniform(-5000, 8000), 2)
    position = round(random.uniform(-10000, 10000), 2)
    return {
        "status": "active",
        "strategy_type": strategy,
        "issue_type": None,
        "pnl": pnl,
        "position": position,
        "order_rate": round(random.uniform(10, 60), 1),
        "fill_rate": round(random.uniform(0.55, 0.85), 2),
        "cancel_rate": round(random.uniform(0.15, 0.55), 2),
        "process_type": random.choice(["docker", "systemd", "user"]),
        "logs": logs,
        "files": {
            "config.yaml": (
                f"agent_name: {bot_id}\nstrategy: {strategy}_v2\n"
                f"symbol: {symbol}\nmax_position: {random.choice([10000, 25000, 50000])}\n"
                f"risk_limit: {random.choice([5000, 10000, 20000])}\n"
            ),
            f"strategy/{strategy}_v2.py": (
                f"# {strategy} strategy v2\n"
                f"def on_tick(data):\n"
                f"    return evaluate(data)\n"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Issue injectors
#
# At low difficulty, patterns are obvious (100% of trades anomalous).
# At high difficulty, anomalous trades are mixed with normal-looking ones.
# The bot stats (PnL, etc.) don't give it away — they look plausible.
# ---------------------------------------------------------------------------

def _inject_pnl_bleed(bot, symbol, base_price, difficulty):
    """Round-trip trades tend to lose money. At high difficulty, some win."""
    rate = _anomaly_rate(difficulty)
    loss_spread = max(0.08, 1.5 - difficulty * 0.15)
    n_trips = random.randint(5, 10)
    injected = []
    timestamps = _gen_timestamps(n_trips * 4, start_hour=14, start_min=random.randint(2, 10))
    ti = 0
    oid = random.randint(50000, 59999)

    for _ in range(n_trips):
        if ti + 3 >= len(timestamps):
            break
        qty = random.choice([200, 300, 400, 500])
        buy_price = round(base_price + random.uniform(-0.50, 0.50), 2)

        if random.random() < rate:
            # Losing trade
            sell_price = round(buy_price - random.uniform(loss_spread * 0.5, loss_spread * 1.5), 2)
        else:
            # Normal/winning trade (camouflage)
            sell_price = round(buy_price + random.uniform(0.02, 0.40), 2)

        oid += 1
        injected.append({"ts": timestamps[ti], "type": "order",
                         "msg": f"NEW BUY {symbol} {qty} @ ${buy_price:.2f} oid={oid}"})
        ti += 1
        injected.append({"ts": timestamps[ti], "type": "fill",
                         "msg": f"FILL BUY {symbol} {qty} @ ${buy_price:.2f} oid={oid}"})
        ti += 1
        oid += 1
        injected.append({"ts": timestamps[ti], "type": "order",
                         "msg": f"NEW SELL {symbol} {qty} @ ${sell_price:.2f} oid={oid}"})
        ti += 1
        injected.append({"ts": timestamps[ti], "type": "fill",
                         "msg": f"FILL SELL {symbol} {qty} @ ${sell_price:.2f} oid={oid}"})
        ti += 1

    bot["logs"] = sorted(bot["logs"] + injected, key=lambda x: x["ts"])
    # PnL looks plausible — slightly negative, not catastrophic
    bot["pnl"] = round(random.uniform(-3000, 1000), 2)
    bot["issue_type"] = "pnl_bleed"
    return bot


def _inject_latency_arb(bot, symbol, base_price, difficulty):
    """Fill prices tend to be worse than order prices. At high difficulty, some are fine."""
    rate = _anomaly_rate(difficulty)
    slippage = max(0.05, 1.20 - difficulty * 0.12)
    n_orders = random.randint(8, 15)
    injected = []
    timestamps = _gen_timestamps(n_orders * 2, start_hour=14, start_min=random.randint(2, 10))
    ti = 0
    oid = random.randint(60000, 69999)

    for _ in range(n_orders):
        if ti + 1 >= len(timestamps):
            break
        qty = random.choice([200, 300, 500])
        side = random.choice(["BUY", "SELL"])
        order_price = round(base_price + random.uniform(-0.30, 0.30), 2)

        if random.random() < rate:
            # Adverse fill
            if side == "BUY":
                fill_price = round(order_price + random.uniform(slippage * 0.5, slippage * 1.5), 2)
            else:
                fill_price = round(order_price - random.uniform(slippage * 0.5, slippage * 1.5), 2)
        else:
            # Normal fill (small random slippage either direction)
            fill_price = round(order_price + random.uniform(-0.03, 0.03), 2)

        oid += 1
        injected.append({"ts": timestamps[ti], "type": "order",
                         "msg": f"NEW {side} {symbol} {qty} @ ${order_price:.2f} oid={oid}"})
        ti += 1
        injected.append({"ts": timestamps[ti], "type": "fill",
                         "msg": f"FILL {side} {symbol} {qty} @ ${fill_price:.2f} oid={oid}"})
        ti += 1

    bot["logs"] = sorted(bot["logs"] + injected, key=lambda x: x["ts"])
    bot["pnl"] = round(random.uniform(-4000, 2000), 2)
    bot["issue_type"] = "latency_arb"
    return bot


def _inject_position_runaway(bot, symbol, base_price, difficulty):
    """Position drifts in one direction. At high difficulty, has occasional reversals."""
    rate = _anomaly_rate(difficulty)
    direction = random.choice(["BUY", "SELL"])
    opposite = "SELL" if direction == "BUY" else "BUY"
    n_trades = random.randint(8, 18)
    injected = []
    timestamps = _gen_timestamps(n_trades * 2, start_hour=14, start_min=random.randint(1, 8))
    ti = 0
    cumulative = 0
    oid = random.randint(70000, 79999)

    for _ in range(n_trades):
        if ti + 1 >= len(timestamps):
            break

        if random.random() < rate:
            side = direction
        else:
            side = opposite  # occasional reversal at high difficulty

        qty = random.choice([300, 500, 800, 1000])
        p = round(base_price + random.uniform(-0.40, 0.40), 2)
        oid += 1

        injected.append({"ts": timestamps[ti], "type": "order",
                         "msg": f"NEW {side} {symbol} {qty} @ ${p:.2f} oid={oid}"})
        ti += 1
        injected.append({"ts": timestamps[ti], "type": "fill",
                         "msg": f"FILL {side} {symbol} {qty} @ ${p:.2f} oid={oid}"})
        ti += 1
        cumulative += qty if side == "BUY" else -qty

        if random.random() < 0.3:
            injected.append({"ts": timestamps[ti - 1], "type": "position",
                             "msg": f"POS {symbol} {cumulative:+d} shares mid=${p:.2f}"})

    bot["logs"] = sorted(bot["logs"] + injected, key=lambda x: x["ts"])
    bot["position"] = float(cumulative)
    bot["pnl"] = round(random.uniform(-3000, 3000), 2)
    bot["issue_type"] = "position_runaway"
    return bot


def _inject_stale_quotes(bot, symbol, base_price, difficulty):
    """Order prices cluster around a fixed value. At high difficulty, has some drift."""
    stale_price = round(base_price - random.uniform(0.50, 2.00), 2)
    n_orders = random.randint(10, 18)
    injected = []
    timestamps = _gen_timestamps(n_orders * 2, start_hour=14, start_min=random.randint(2, 10))
    ti = 0
    oid = random.randint(80000, 89999)

    # At high difficulty, the stale price drifts very slowly
    drift_per_order = 0.0 if difficulty < 4 else random.uniform(0.01, 0.03)

    for i in range(n_orders):
        if ti + 1 >= len(timestamps):
            break
        qty = random.choice([200, 300, 500])
        side = random.choice(["BUY", "SELL"])

        # Order price: stale with possible slow drift
        order_p = round(stale_price + drift_per_order * i + random.uniform(-0.02, 0.02), 2)

        # Fill at actual market price (which moves normally)
        market_p = round(base_price + random.uniform(-0.30, 0.30), 2)
        oid += 1

        injected.append({"ts": timestamps[ti], "type": "order",
                         "msg": f"NEW {side} {symbol} {qty} @ ${order_p:.2f} oid={oid}"})
        ti += 1
        injected.append({"ts": timestamps[ti], "type": "fill",
                         "msg": f"FILL {side} {symbol} {qty} @ ${market_p:.2f} oid={oid}"})
        ti += 1

    bot["logs"] = sorted(bot["logs"] + injected, key=lambda x: x["ts"])
    bot["pnl"] = round(random.uniform(-3000, 2000), 2)
    bot["issue_type"] = "stale_quotes"
    return bot


def _inject_order_stuffing(bot, symbol, base_price, difficulty):
    """Bursts of rapid order+cancel. At high difficulty, bursts are smaller and fewer."""
    n_bursts = random.randint(2, 4) if difficulty >= 5 else random.randint(3, 6)
    burst_size_min = 4 if difficulty >= 5 else 10
    burst_size_max = 12 if difficulty >= 5 else 30
    injected = []
    oid = random.randint(90000, 94999)

    for _ in range(n_bursts):
        burst_ts = _gen_timestamps(1, start_hour=14, start_min=random.randint(1, 15))[0]
        count = random.randint(burst_size_min, burst_size_max)
        for _ in range(count):
            p = round(base_price + random.uniform(-0.03, 0.03), 2)
            oid += 1
            injected.append({"ts": burst_ts, "type": "order",
                             "msg": f"NEW BUY {symbol} 100 @ ${p:.2f} oid={oid}"})
            injected.append({"ts": burst_ts, "type": "cancel",
                             "msg": f"CXLD oid={oid} {symbol}"})

    bot["logs"] = sorted(bot["logs"] + injected, key=lambda x: x["ts"])
    bot["order_rate"] = round(random.uniform(30, 70), 1)
    bot["cancel_rate"] = round(random.uniform(0.40, 0.65), 2)
    bot["issue_type"] = "order_stuffing"
    return bot


def _inject_config_drift(bot, symbol, base_price, difficulty):
    """Bot trading a different symbol than config specifies.

    At high difficulty, only some trades are in the wrong symbol —
    the bot partially works on the right one too.
    """
    rate = _anomaly_rate(difficulty)
    wrong_symbol = random.choice([s for s in SYMBOLS if s != symbol])
    wrong_price = round(random.uniform(80, 900), 2)
    n_orders = random.randint(8, 14)
    injected = []
    timestamps = _gen_timestamps(n_orders * 2, start_hour=14, start_min=random.randint(1, 8))
    ti = 0
    oid = random.randint(95000, 99999)

    for _ in range(n_orders):
        if ti + 1 >= len(timestamps):
            break
        qty = random.choice([200, 400, 600])
        side = random.choice(["BUY", "SELL"])

        if random.random() < rate:
            trade_symbol = wrong_symbol
            p = round(wrong_price + random.uniform(-0.30, 0.30), 2)
        else:
            trade_symbol = symbol
            p = round(base_price + random.uniform(-0.30, 0.30), 2)

        oid += 1
        injected.append({"ts": timestamps[ti], "type": "order",
                         "msg": f"NEW {side} {trade_symbol} {qty} @ ${p:.2f} oid={oid}"})
        ti += 1
        fill_p = round(p + random.uniform(-0.05, 0.05), 2)
        injected.append({"ts": timestamps[ti], "type": "fill",
                         "msg": f"FILL {side} {trade_symbol} {qty} @ ${fill_p:.2f} oid={oid}"})
        ti += 1

    bot["logs"] = sorted(bot["logs"] + injected, key=lambda x: x["ts"])
    bot["files"]["config.yaml"] = (
        f"agent_name: {bot.get('_id', 'unknown')}\nstrategy: momentum_v2\n"
        f"symbol: {symbol}\nmax_position: 50000\n"
        f"risk_limit: 20000\n"
    )
    bot["pnl"] = round(random.uniform(-3000, 3000), 2)
    bot["issue_type"] = "config_drift"
    return bot


# ---------------------------------------------------------------------------
# Cross-module injectors
# ---------------------------------------------------------------------------

def _inject_wash_trading(bot_a, bot_b, symbol, base_price, difficulty):
    """Two bots with matching trades. At high difficulty, not every trade matches."""
    rate = _anomaly_rate(difficulty)
    n_events = random.randint(6, 12)
    timestamps = _gen_timestamps(n_events, start_hour=14, start_min=random.randint(1, 10))
    a_inject, b_inject = [], []
    p = base_price
    oid_a = random.randint(40000, 44999)
    oid_b = random.randint(45000, 49999)

    for ts in timestamps:
        p += round(random.uniform(-0.20, 0.30), 2)

        if random.random() < rate:
            # Matched wash trade
            qty = random.choice([500, 800, 1000, 1200])

            # At high difficulty, vary qty and offset timestamp
            if difficulty >= 5:
                qty_b = qty + random.choice([-100, 0, 0, 100])
                ts_b = _advance_ts(ts, random.randint(1, 2))
            else:
                qty_b = qty
                ts_b = ts

            oid_a += 1
            oid_b += 1

            a_inject.append({"ts": ts, "type": "order",
                             "msg": f"NEW BUY {symbol} {qty} @ ${p:.2f} oid={oid_a}"})
            a_inject.append({"ts": ts, "type": "fill",
                             "msg": f"FILL BUY {symbol} {qty} @ ${p:.2f} oid={oid_a}"})

            b_inject.append({"ts": ts_b, "type": "order",
                             "msg": f"NEW SELL {symbol} {qty_b} @ ${p:.2f} oid={oid_b}"})
            b_inject.append({"ts": ts_b, "type": "fill",
                             "msg": f"FILL SELL {symbol} {qty_b} @ ${p:.2f} oid={oid_b}"})
        else:
            # Independent trades (noise) — each bot does its own thing
            oid_a += 1
            qty_a = random.choice([200, 300, 500])
            side_a = random.choice(["BUY", "SELL"])
            a_inject.append({"ts": ts, "type": "order",
                             "msg": f"NEW {side_a} {symbol} {qty_a} @ ${p:.2f} oid={oid_a}"})
            a_inject.append({"ts": ts, "type": "fill",
                             "msg": f"FILL {side_a} {symbol} {qty_a} @ ${p:.2f} oid={oid_a}"})

    bot_a["logs"] = sorted(bot_a["logs"] + a_inject, key=lambda x: x["ts"])
    bot_b["logs"] = sorted(bot_b["logs"] + b_inject, key=lambda x: x["ts"])
    bot_a["pnl"] = round(random.uniform(-2000, 2000), 2)
    bot_b["pnl"] = round(random.uniform(-2000, 2000), 2)
    bot_a["position"] = round(random.uniform(-3000, 3000), 2)
    bot_b["position"] = round(random.uniform(-3000, 3000), 2)
    bot_a["issue_type"] = "wash_trading"
    bot_b["issue_type"] = "wash_trading"
    return bot_a, bot_b


def _inject_counter_trading(victim, parasite, symbol, base_price, difficulty):
    """Parasite mirrors victim's trades with a delay. At high difficulty, not always."""
    rate = _anomaly_rate(difficulty)
    n_trades = random.randint(6, 12)
    timestamps = _gen_timestamps(n_trades, start_hour=14, start_min=random.randint(1, 8))
    v_inject, p_inject = [], []
    p = base_price
    oid_v = random.randint(30000, 34999)
    oid_p = random.randint(35000, 39999)

    for ts in timestamps:
        side = random.choice(["BUY", "SELL"])
        qty = random.choice([300, 400, 500, 600])
        p += round(random.uniform(-0.80, 0.80), 2)
        oid_v += 1

        v_inject.append({"ts": ts, "type": "order",
                         "msg": f"NEW {side} {symbol} {qty} @ ${p:.2f} oid={oid_v}"})
        v_inject.append({"ts": ts, "type": "fill",
                         "msg": f"FILL {side} {symbol} {qty} @ ${p:.2f} oid={oid_v}"})

        if random.random() < rate:
            # Parasite mirrors
            delay = random.randint(2, 4)
            counter_ts = _advance_ts(ts, delay)
            counter_side = "SELL" if side == "BUY" else "BUY"
            counter_p = round(p + (0.03 if counter_side == "SELL" else -0.03), 2)
            oid_p += 1

            p_inject.append({"ts": counter_ts, "type": "order",
                             "msg": f"NEW {counter_side} {symbol} {qty} @ ${counter_p:.2f} oid={oid_p}"})
            p_inject.append({"ts": counter_ts, "type": "fill",
                             "msg": f"FILL {counter_side} {symbol} {qty} @ ${counter_p:.2f} oid={oid_p}"})
        else:
            # Parasite does its own unrelated trade
            oid_p += 1
            ind_side = random.choice(["BUY", "SELL"])
            ind_qty = random.choice([100, 200, 300])
            ind_p = round(base_price + random.uniform(-1.0, 1.0), 2)
            p_inject.append({"ts": _advance_ts(ts, random.randint(5, 15)), "type": "order",
                             "msg": f"NEW {ind_side} {symbol} {ind_qty} @ ${ind_p:.2f} oid={oid_p}"})
            p_inject.append({"ts": _advance_ts(ts, random.randint(5, 15)), "type": "fill",
                             "msg": f"FILL {ind_side} {symbol} {ind_qty} @ ${ind_p:.2f} oid={oid_p}"})

    victim["logs"] = sorted(victim["logs"] + v_inject, key=lambda x: x["ts"])
    parasite["logs"] = sorted(parasite["logs"] + p_inject, key=lambda x: x["ts"])
    victim["pnl"] = round(random.uniform(-4000, 1000), 2)
    parasite["pnl"] = round(random.uniform(-1000, 4000), 2)
    victim["issue_type"] = "counter_trading_victim"
    parasite["issue_type"] = "counter_trading_parasite"

    victim_id = victim.get("_id", "victim")
    parasite["files"][f"strategy/counter_momentum_v2.py"] = (
        f"import json\n\n"
        f"FEED = '/shared/positions/{victim_id}.json'\n\n"
        f"def on_tick(data):\n"
        f"    with open(FEED) as f:\n"
        f"        state = json.load(f)\n"
        f"    if state['side'] == 'long':\n"
        f"        return Sell(size=state['qty'])\n"
        f"    elif state['side'] == 'short':\n"
        f"        return Buy(size=state['qty'])\n"
        f"    return Hold()\n"
    )
    return victim, parasite


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_scenario(difficulty: int = 1) -> dict:
    eligible = [
        name for name, meta in ISSUE_TYPES.items()
        if meta["min_diff"] <= difficulty <= meta["max_diff"]
    ]
    issue = random.choice(eligible)
    is_cross = ISSUE_TYPES[issue]["cross_module"]

    # Difficulty 1-2: just 1 normal bot + 1 bad bot (learn the flow)
    # Difficulty 3-4: 2 normal + 1 bad
    # Difficulty 5+: scales up
    if difficulty <= 2:
        num_normal = 1
    elif difficulty <= 4:
        num_normal = 2
    else:
        num_normal = min(2 + difficulty // 2, 6)
    log_count = min(30 + difficulty * 8, 120)

    symbols = random.sample(SYMBOLS, num_normal + 2)
    names = random.sample(BOT_NAMES, num_normal + 2)
    base_prices = {s: round(random.uniform(80, 900), 2) for s in symbols}

    bots = {}
    for i in range(num_normal):
        bot_id = names[i]
        sym = symbols[i]
        bots[bot_id] = _make_normal_bot(bot_id, sym, base_prices[sym], log_count=log_count)

    bad_bots = []

    if is_cross:
        bad_id_1 = names[num_normal]
        bad_id_2 = names[num_normal + 1]
        bad_sym = symbols[num_normal]

        bot_1 = _make_normal_bot(bad_id_1, bad_sym, base_prices[bad_sym], log_count=log_count)
        bot_2 = _make_normal_bot(bad_id_2, bad_sym, base_prices[bad_sym], log_count=log_count)
        bot_1["_id"] = bad_id_1
        bot_2["_id"] = bad_id_2

        if issue == "wash_trading":
            bot_1, bot_2 = _inject_wash_trading(bot_1, bot_2, bad_sym, base_prices[bad_sym], difficulty)
            bad_bots = [bad_id_1, bad_id_2]
            desc = f"{bad_id_1} and {bad_id_2} are wash trading {bad_sym}."
        elif issue == "counter_trading":
            bot_1, bot_2 = _inject_counter_trading(bot_1, bot_2, bad_sym, base_prices[bad_sym], difficulty)
            bad_bots = [bad_id_2]
            desc = f"{bad_id_2} is counter-trading {bad_id_1} on {bad_sym}."

        bot_1.pop("_id", None)
        bot_2.pop("_id", None)
        bots[bad_id_1] = bot_1
        bots[bad_id_2] = bot_2
    else:
        bad_id = names[num_normal]
        bad_sym = symbols[num_normal]
        bad_bot = _make_normal_bot(bad_id, bad_sym, base_prices[bad_sym], log_count=log_count)
        bad_bot["_id"] = bad_id

        if issue == "pnl_bleed":
            bad_bot = _inject_pnl_bleed(bad_bot, bad_sym, base_prices[bad_sym], difficulty)
            desc = f"{bad_id} losing on round-trips in {bad_sym}."
        elif issue == "latency_arb":
            bad_bot = _inject_latency_arb(bad_bot, bad_sym, base_prices[bad_sym], difficulty)
            desc = f"{bad_id} fill prices consistently worse than order prices on {bad_sym}."
        elif issue == "position_runaway":
            bad_bot = _inject_position_runaway(bad_bot, bad_sym, base_prices[bad_sym], difficulty)
            desc = f"{bad_id} position drifting on {bad_sym}."
        elif issue == "stale_quotes":
            bad_bot = _inject_stale_quotes(bad_bot, bad_sym, base_prices[bad_sym], difficulty)
            desc = f"{bad_id} order prices clustering on {bad_sym}."
        elif issue == "order_stuffing":
            bad_bot = _inject_order_stuffing(bad_bot, bad_sym, base_prices[bad_sym], difficulty)
            desc = f"{bad_id} rapid order+cancel bursts on {bad_sym}."
        elif issue == "config_drift":
            bad_bot = _inject_config_drift(bad_bot, bad_sym, base_prices[bad_sym], difficulty)
            desc = f"{bad_id} trading wrong symbol (config vs actual mismatch)."
        else:
            desc = f"{bad_id} has issue: {issue}"

        bad_bot.pop("_id", None)
        bots[bad_id] = bad_bot
        bad_bots = [bad_id]

    return {
        "bots": bots,
        "bad_bots": bad_bots,
        "issue": issue,
        "description": desc,
    }

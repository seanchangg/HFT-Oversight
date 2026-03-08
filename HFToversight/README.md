---
title: HFT Oversight Agent
emoji: 🔍
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
short_description: Train LLMs to detect rogue trading bots by reading logs
tags:
  - openenv
---

# HFT Oversight Agent

An OpenEnv RL environment that trains LLMs to be autonomous oversight agents for high-frequency trading bot fleets.

The agent receives a fleet of trading bots, reads their logs, correlates activity across bots, and must identify and shut down malfunctioning ones — all from text observations alone.

**Target tracks:** Statement 1 (Multi-Agent), Statement 4 (Self-Improvement), Fleet AI ($10K prize)

## Why This Environment

Real trading firms run dozens of bots simultaneously. When one goes rogue — wash trading, counter-trading, bleeding PnL — the damage compounds every second it runs undetected. Human oversight doesn't scale. This environment trains LLMs to do what humans can't: read hundreds of log lines, spot statistical anomalies, cross-reference activity across bots, and act decisively.

## How It Works

```
Agent sees: Fleet of 3-8 bots with logs, configs, stats
Agent does: read_logs, inspect_config, cross_reference, flag, shutdown
Agent learns: Investigation before action, pattern recognition, diagnosis accuracy
```

### 8 Anomaly Types (Difficulty 1-10)

| Type | Difficulty | Detection Method |
|------|-----------|-----------------|
| PnL Bleed | 1-10 | Round-trip trades losing money |
| Latency Arb | 1-10 | Fill prices worse than order prices |
| Position Runaway | 2-10 | Position drifting in one direction |
| Stale Quotes | 2-10 | Order prices clustering around fixed value |
| Order Stuffing | 3-10 | Bursts of rapid order+cancel |
| Config Drift | 3-10 | Trading wrong symbol vs config |
| Wash Trading | 4-10 | Two bots matching trades (cross-bot) |
| Counter Trading | 5-10 | One bot mirroring another with delay (cross-bot) |

### Progressive Difficulty (Self-Improvement)

Difficulty scales automatically based on agent performance:
- **Low difficulty (1-3):** 100% of trades show the anomaly — obvious patterns
- **Mid difficulty (4-6):** 70-85% anomalous — mixed with normal noise
- **High difficulty (7-10):** 50-60% anomalous — statistical patterns requiring cross-bot correlation

The environment advances difficulty after 2 consecutive wins and regresses after 3 consecutive losses.

### Reward Structure

The reward system incentivizes thorough investigation over lucky guesses:

| Component | Points | Purpose |
|-----------|--------|---------|
| Correct shutdown | +3.0 base | Right answer |
| Investigation depth | up to +4.0 | Read logs before acting |
| Flag before shutdown | +2.0 | Careful process |
| Diagnosis accuracy | up to +3.0 | Understood the issue |
| Speed bonus | up to +3.8 | Fewer steps = more bonus |
| Efficiency bonus | up to +5.0 | Low accumulated damage |
| Wrong shutdown | -10.0 | Penalize false positives |
| Time out | -5.0 | Failed to find bad bot |

## Quick Start

```python
from HFToversight import HFTOversightEnv, OversightAction

with HFTOversightEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(result.observation.response)

    # Investigate
    result = env.step(OversightAction(command="read_logs", bot_id="alpha"))
    print(result.observation.response)

    # Cross-reference two bots
    result = env.step(OversightAction(
        command="cross_reference", bot_id="alpha", bot_id_2="gamma"
    ))

    # Flag and shutdown
    result = env.step(OversightAction(
        command="flag", bot_id="gamma", reason="Counter-trading alpha"
    ))
    result = env.step(OversightAction(
        command="shutdown", bot_id="gamma", reason="Mirroring alpha's trades with 3s delay"
    ))
```

## Training

```bash
# Collect baseline (free, uses HF Inference API)
uv run python train.py --baseline-only

# Full GRPO training (needs GPU)
python train.py
```

## Local Testing

```bash
# Test environment logic directly
uv run python test_local.py

# Run server locally
cd HFToversight && uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Deploy

```bash
openenv push
```

## Architecture

```
client.py (EnvClient)  <-- WebSocket -->  server/app.py (FastAPI)
                                              |
                                         server/environment.py (HFTOversightEnvironment)
                                              |
                                         scenarios.py (procedural scenario generator)
```

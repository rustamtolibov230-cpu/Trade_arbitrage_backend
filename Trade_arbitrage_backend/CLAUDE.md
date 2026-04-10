# Trade Arbitrage — Claude Context

## Project Overview
Pairs trading / statistical arbitrage bot that trades mean-reverting spreads between correlated assets on MetaTrader 5. Runs as a FastAPI server with a live web dashboard. Packaged as `run.exe` for one-click launch.

## Architecture
- **Pairs Trading Engine**: Monitors price ratios between correlated assets, uses Z-score for entry/exit
- **Mean Reversion**: Enter when spread widens (Z >= 1.0), exit when it reverts (Z <= 0.2), stop if it blows up (Z >= 2.5)
- **ATR-Based Hedge Ratio**: Auto-adjusts leg B lot size by ATR ratio to balance dollar exposure
- **Session Awareness**: BTC/ETH = 24/7, XAU/XAG = London+NY (07:00-17:00 UTC)
- **Risk Manager**: Daily loss limit, max open pairs, correlation monitoring
- **FastAPI + WebSocket**: Real-time dashboard with Start/Stop control, spread monitor, signal log
- **PyInstaller**: Bundled as single `run.exe` (server + dashboard + bot)

## Active Pairs
| Pair | Lots | Filling | Session |
|------|------|---------|---------|
| BTCUSD / ETHUSD | 0.04 / 0.80 | FOK | 24/7 |
| XAUUSD / XAGUSD | 0.01 / 0.01 | IOC | 07-17 UTC |

## Key Files
```
main.py                     — Standalone bot entry point (no dashboard)
server.py                   — FastAPI server: bot loop + REST API + WebSocket
run.py                      — Launcher: starts server + opens browser
build_exe.py                — PyInstaller build script → dist/run.exe
config/settings.py          — PairConfig definitions, MT5 creds, risk limits
src/mt5_client.py           — Async MT5 wrapper (auto-detect filling mode per symbol)
src/spread_tracker.py       — Z-score spread engine + signal generation
src/execution.py            — Opens/closes hedged pairs, P&L tracking, trade history
src/risk_manager.py         — Daily loss limit, correlation guard, pair count limit
src/schemas.py              — SpreadState, PairSignal, ActivePair, TradeResult
frontend/index.html         — Dashboard: Start/Stop, spread monitor, active pairs, signal log
```

## Trading Workflow
1. User clicks **START** on dashboard
2. Every 15s: fetch M1 bars for both legs → compute price ratio → Z-score
3. Entry: |Z| >= 1.0 → sell overpriced leg, buy underpriced leg
4. Exit: |Z| <= 0.2 → spread reverted, close both legs for profit
5. Stop: |Z| >= 2.5 → correlation breaking, cut losses
6. Timeout: auto-close after 30 minutes
7. Daily loss limit: -$50 → close all, stop bot

## Key Settings (config/settings.py)
```python
scan_interval_seconds = 15      # scan every 15s
timeframe = "M1"                # 1-minute bars for fast signals
spread_lookback = 50            # 50 bars for Z-score calculation
zscore_entry = 1.0              # enter when spread is 1 std dev wide
zscore_exit = 0.2               # exit when spread reverts near mean
zscore_stop = 2.5               # stop loss if spread keeps widening
max_hold_minutes = 30           # force-close after 30 min
max_daily_loss = 50.0           # circuit breaker
max_open_pairs = 3              # max simultaneous pairs
min_correlation = 0.80          # don't trade if correlation drops
```

## MT5 Account
- Broker: BlackBullMarkets-Demo
- Account: 740899
- Credentials in `.env` (never commit)

## Conventions
- All MT5 calls run in ThreadPoolExecutor (async-wrapped)
- Filling mode auto-detected per symbol from `symbol_info().filling_mode`
- Lot sizes rounded to symbol's `volume_step`
- Bot does NOT auto-start — waits for dashboard Start button
- WebSocket broadcasts spread state, signals, and bot state to dashboard
- `global` keyword required for module-level mutables in Python 3.14

## Build & Run
```bash
# Dev mode
python run.py

# Build exe
python build_exe.py

# Run exe (opens dashboard in browser)
dist/run.exe
```

## Safety Rules
- NEVER modify risk limits without user approval
- NEVER commit `.env` file
- Auto-close all pairs on Stop/shutdown/daily loss limit
- If leg B fails to open, immediately close leg A (no unhedged exposure)
- Partial close warning: if one leg fails to close, log "MANUAL CHECK NEEDED"

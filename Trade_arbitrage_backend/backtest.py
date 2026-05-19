"""Backtest CLI — replay the pairs strategy on cached historical bars.

Usage:
    python backtest.py --pair BTCUSD/ETHUSD --days 60
    python backtest.py --all --days 90 --save reports/
    python backtest.py --pair XAUUSD/XAGUSD --days 30 --no-coint  # disable filter

Output: per-pair metrics table + (optional) CSV equity curve / trade log.

Data flow:
    HistoricalDataLoader  (MT5 + parquet cache)
            |
            v
       aligned bars  (DataFrame with a_*, b_* OHLC columns)
            |
            v
        Simulator     (mirrors live logic, see src/backtest/simulator.py)
            |
            v
    SimResult: trades + equity curve + metrics
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# --- PyInstaller frozen-path fix: when run as .exe, switch CWD to the exe
# directory so data/backtest_cache, .env, reports/ all resolve next to it.
if getattr(sys, "frozen", False):
    _BUNDLE_DIR = sys._MEIPASS  # type: ignore[attr-defined]
    _EXE_DIR = os.path.dirname(os.path.abspath(sys.executable))
    sys.path.insert(0, _BUNDLE_DIR)
    os.chdir(_EXE_DIR)

import pandas as pd
from loguru import logger

from config.settings import PAIRS, PairConfig, settings
from src.backtest.cost_model import IC_MARKETS_RAW
from src.backtest.data_loader import HistoricalDataLoader
from src.backtest.simulator import Simulator, SimSettings, SimResult
from src.mt5_client import MT5Client


# Minutes per bar for the scan timeframe — matches simulator._TF_MINUTES
_TF_MIN = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}


def _select_pairs(arg_pair: str | None, arg_all: bool) -> list[PairConfig]:
    if arg_all:
        return list(PAIRS)
    if not arg_pair:
        raise SystemExit("Specify --pair LEG_A/LEG_B or --all")
    for cfg in PAIRS:
        if f"{cfg.leg_a}/{cfg.leg_b}" == arg_pair:
            return [cfg]
    raise SystemExit(
        f"Pair {arg_pair!r} not in config; available: "
        + ", ".join(f"{c.leg_a}/{c.leg_b}" for c in PAIRS)
    )


def _build_sim_settings(args) -> SimSettings:
    return SimSettings(
        timeframe=settings.timeframe,
        spread_lookback=settings.spread_lookback,
        coint_lookback_bars=settings.coint_lookback_bars,
        coint_recheck_minutes=max(1, settings.coint_recheck_seconds // 60),
        coint_p_threshold=settings.coint_p_threshold,
        coint_min_half_life=settings.coint_min_half_life,
        coint_max_half_life=settings.coint_max_half_life,
        require_cointegration=not args.no_coint,
        min_correlation=settings.min_correlation if not args.no_corr else 0.0,
        cooldown_seconds=settings.cooldown_seconds,
        max_daily_loss=settings.max_daily_loss,
        profit_target_safety_factor=settings.profit_target_safety_factor,
    )


def _print_summary(res: SimResult, cfg: PairConfig) -> None:
    m = res.metrics
    days = res.bars_minutes / 1440.0
    print()
    print("=" * 70)
    print(f"  {res.pair_name}  |  {days:.1f} days  |  {res.bars_used} bars  "
          f"|  target=${cfg.profit_target:.2f}")
    print("=" * 70)
    print(f"  Trades            : {m.trades}   "
          f"(wins {m.wins}, losses {m.losses}, win-rate {m.win_rate}%)")
    print(f"  Total P&L         : ${m.total_pnl:>10.2f}")
    print(f"  Avg / trade       : ${m.avg_pnl:>10.4f}")
    print(f"  Best / Worst      : ${m.best:>10.2f}  /  ${m.worst:>10.2f}")
    print(f"  Profit factor     : {m.profit_factor:>10.3f}")
    print(f"  Sharpe (ann.)     : {m.sharpe:>10.3f}")
    print(f"  Sortino (ann.)    : {m.sortino:>10.3f}")
    print(f"  Max drawdown      : ${m.max_drawdown:>10.2f}  ({m.max_drawdown_pct}%)")
    print(f"  Avg hold (min)    : {m.avg_hold_min:>10.1f}")
    print(f"  Trades / day      : {m.trades_per_day:>10.3f}")
    print(f"  Skipped by gates  : "
          f"cost={res.cost_blocks}  coint={res.coint_blocks}  corr={res.corr_blocks}")

    # Reason breakdown
    if res.trades:
        from collections import Counter
        reasons = Counter(t.reason for t in res.trades)
        print(f"  Exit reasons      : "
              + "  ".join(f"{k}={v}" for k, v in reasons.most_common()))

    # Honest reading
    if m.trades == 0:
        print("\n  >> No trades. Likely all entries blocked — see Skipped counts.")
    elif m.total_pnl <= 0:
        print(f"\n  >> Negative net P&L after costs — strategy lacks edge on this window.")
    elif m.profit_factor < 1.3:
        print(f"\n  >> Profit factor < 1.3 — thin edge; results may be noise.")
    elif m.max_drawdown_pct > 50:
        print(f"\n  >> Drawdown >50% of peak — risk of ruin on small accounts.")
    else:
        print(f"\n  >> Looks promising on this window. Run walk-forward OOS before trusting it.")


def _save_outputs(res: SimResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = res.pair_name.replace("/", "_")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = out_dir / f"{safe}_{stamp}"

    # Equity curve CSV
    res.equity_curve.to_csv(f"{base}_equity.csv", index=False)

    # Trade log CSV
    if res.trades:
        rows = []
        for t in res.trades:
            rows.append({
                "pair": t.pair_name, "side": t.side,
                "entry_time": t.entry_time.isoformat(),
                "exit_time": t.exit_time.isoformat(),
                "entry_z": t.entry_zscore, "exit_z": t.exit_zscore,
                "leg_a_lot": t.leg_a_lot, "leg_b_lot": t.leg_b_lot,
                "entry_a": t.entry_price_a, "exit_a": t.exit_price_a,
                "entry_b": t.entry_price_b, "exit_b": t.exit_price_b,
                "gross_pnl": t.gross_pnl, "costs": t.costs, "profit": t.profit,
                "hold_min": t.hold_minutes, "reason": t.reason,
            })
        pd.DataFrame(rows).to_csv(f"{base}_trades.csv", index=False)

    # Metrics JSON
    import json
    metrics_path = f"{base}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(res.metrics.to_dict(), f, indent=2)

    print(f"  Saved             : {base}_equity.csv | _trades.csv | _metrics.json")


async def _run(args, mt5_client: MT5Client | None = None) -> int:
    """Run one backtest pass. If ``mt5_client`` is provided, it's reused and not
    disconnected (so an interactive loop can chain runs cheaply).
    """
    sim_settings = _build_sim_settings(args)
    pairs = _select_pairs(args.pair, args.all)
    bar_min = _TF_MIN.get(sim_settings.timeframe, 1)

    # Bars needed = days * (1440 minutes/day) / bar_minutes + lookback warmup
    warmup = sim_settings.coint_lookback_bars + sim_settings.spread_lookback
    bars_per_day = int(1440 / bar_min)
    requested_bars = args.days * bars_per_day + warmup

    owns_client = mt5_client is None
    if owns_client:
        mt5_client = MT5Client()
        connected = await mt5_client.connect()
        if not connected:
            logger.warning(
                "MT5 connect failed — backtester will only use cached parquet data "
                "(no fresh fetch). If cache is empty for a pair, that pair will be skipped."
            )

    loader = HistoricalDataLoader(mt5_client)

    out_dir = Path(args.save) if args.save else None
    overall_pnl = 0.0
    overall_trades = 0

    for cfg in pairs:
        pair_name = f"{cfg.leg_a}/{cfg.leg_b}"
        logger.info(f"--- {pair_name} | requesting {requested_bars} bars ---")

        try:
            bars = await loader.load_pair(
                cfg.leg_a, cfg.leg_b,
                timeframe=sim_settings.timeframe,
                bars=requested_bars,
                use_cache=True,
            )
        except Exception as e:
            logger.error(f"data load failed for {pair_name}: {e}")
            continue
        if bars is None or len(bars) < warmup + 100:
            logger.error(
                f"{pair_name}: only {0 if bars is None else len(bars)} aligned bars "
                f"(need >{warmup + 100}). Skipping."
            )
            continue

        try:
            sim = Simulator(cfg, IC_MARKETS_RAW, sim_settings)
            result = sim.simulate(bars)
        except Exception as e:
            logger.exception(f"simulator failed for {pair_name}: {e}")
            continue

        _print_summary(result, cfg)
        if out_dir is not None:
            _save_outputs(result, out_dir)

        overall_pnl += result.metrics.total_pnl
        overall_trades += result.metrics.trades

    if owns_client and mt5_client.is_connected():
        await mt5_client.disconnect()

    if len(pairs) > 1:
        print()
        print("=" * 70)
        print(f"  OVERALL  |  trades={overall_trades}  |  net P&L=${overall_pnl:.2f}")
        print("=" * 70)

    return 0


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        ans = input(f"  {prompt}{suffix}: ").strip()
    except EOFError:
        ans = ""
    return ans or default


def _interactive_args() -> argparse.Namespace:
    """Friendly prompts for users double-clicking the exe."""
    print()
    print("=" * 60)
    print("  PAIRS TRADING BACKTEST")
    print("=" * 60)
    print()
    print("  Configured pairs:")
    for i, cfg in enumerate(PAIRS, start=1):
        name = f"{cfg.leg_a}/{cfg.leg_b}"
        print(f"    {i}) {name}  (target ${cfg.profit_target:.2f})")
    print(f"    {len(PAIRS) + 1}) ALL")
    print()

    default_choice = str(len(PAIRS) + 1)
    choice = _ask("Choose a pair number", default_choice)
    try:
        idx = int(choice)
    except ValueError:
        idx = int(default_choice)

    pair = None
    run_all = False
    if 1 <= idx <= len(PAIRS):
        cfg = PAIRS[idx - 1]
        pair = f"{cfg.leg_a}/{cfg.leg_b}"
    else:
        run_all = True

    days_raw = _ask("Window in days", "30")
    try:
        days = max(1, int(days_raw))
    except ValueError:
        days = 30

    save_ans = _ask("Save CSV reports? (y/N)", "n").lower()
    save_dir = "reports" if save_ans.startswith("y") else None

    no_coint = _ask("Disable cointegration filter? (y/N)", "n").lower().startswith("y")
    no_corr = _ask("Disable correlation filter? (y/N)", "n").lower().startswith("y")

    print()
    return argparse.Namespace(
        pair=pair, all=run_all, days=days, save=save_dir,
        no_coint=no_coint, no_corr=no_corr,
    )


async def _interactive_loop() -> int:
    """Run backtests repeatedly until the user says no. Keeps one MT5 client
    connected across runs so each rerun is fast (no reconnect, cache warm).
    """
    mt5_client = MT5Client()
    connected = await mt5_client.connect()
    if not connected:
        logger.warning(
            "MT5 connect failed — only cached data will be available. "
            "Open MT5 Terminal and log in, then restart this exe."
        )

    rc = 0
    try:
        while True:
            args = _interactive_args()
            try:
                rc = await _run(args, mt5_client=mt5_client)
            except KeyboardInterrupt:
                rc = 130
                break
            except Exception as e:
                logger.exception(f"backtest run failed: {e}")
                rc = 1

            print()
            again = _ask("Run another backtest? (Y/n)", "y").lower()
            if again.startswith("n"):
                break
    finally:
        if mt5_client.is_connected():
            await mt5_client.disconnect()

    return rc


def main(argv: list[str] | None = None) -> int:
    # If invoked with no args (double-click), drop into interactive mode.
    cli_argv = argv if argv is not None else sys.argv[1:]
    interactive = len(cli_argv) == 0

    # Friendlier loguru format for CLI runs
    logger.remove()
    logger.add(sys.stdout,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
               level="INFO")

    if interactive:
        try:
            rc = asyncio.run(_interactive_loop())
        except KeyboardInterrupt:
            rc = 130
        print()
        try:
            input("  Press Enter to close window...")
        except EOFError:
            pass
        return rc

    p = argparse.ArgumentParser(description="Pairs trading backtest")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--pair", help="e.g. BTCUSD/ETHUSD")
    g.add_argument("--all", action="store_true", help="run all configured pairs")
    p.add_argument("--days", type=int, default=30, help="window in days")
    p.add_argument("--save", help="output directory for equity/trades/metrics CSVs")
    p.add_argument("--no-coint", action="store_true",
                   help="disable cointegration filter (compare vs filtered)")
    p.add_argument("--no-corr", action="store_true",
                   help="disable correlation filter")
    args = p.parse_args(cli_argv)

    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())

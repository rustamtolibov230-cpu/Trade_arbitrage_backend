"""
Trade Arbitrage Bot — Pairs Trading Engine
==========================================

Trades mean-reverting spreads between correlated assets:
  - BTC/USD vs ETH/USD (crypto, 24/7)
  - XAU/USD vs XAG/USD (metals, London+NY sessions)

Uses Z-score of price ratio to detect entry/exit points.
"""

import asyncio
import signal
import sys
from datetime import datetime, timezone

from loguru import logger

from config.settings import settings, PAIRS
from src.mt5_client import MT5Client
from src.spread_tracker import SpreadTracker
from src.execution import ExecutionEngine
from src.risk_manager import RiskManager
from src.cointegration import CointegrationMonitor
from src.backtest.cost_model import (
    IC_MARKETS_RAW,
    validate_profit_targets,
    format_cost_check_line,
    check_trade_viability,
)

# --- Logging setup ---
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
    level="INFO",
)
logger.add(
    "logs/arb_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG",
)


class ArbitrageBot:
    """Main bot — scans pairs, generates signals, executes trades."""

    def __init__(self):
        self.mt5 = MT5Client()
        self.execution = ExecutionEngine(self.mt5)
        self.coint_monitor = CointegrationMonitor(PAIRS, self.mt5)
        self.risk = RiskManager(self.execution, coint=self.coint_monitor)
        self.trackers = [SpreadTracker(cfg, self.mt5) for cfg in PAIRS]
        self._running = False
        self._coint_task: asyncio.Task | None = None

    async def start(self):
        """Connect to MT5 and start the main loop."""
        logger.info("=" * 60)
        logger.info("  TRADE ARBITRAGE BOT — Starting")
        logger.info("=" * 60)

        # Profit-target vs round-trip cost sanity gate. Refuse to start if
        # any pair would lose money on every closed trade.
        sf = settings.profit_target_safety_factor
        all_ok, rows = validate_profit_targets(PAIRS, IC_MARKETS_RAW, safety_factor=sf)
        for row in rows:
            line = format_cost_check_line(row, sf)
            if row["ok"] and not row.get("error"):
                logger.info(line)
            else:
                logger.error(line)
        if not all_ok:
            logger.error(
                "Aborting — profit-target sanity check failed. "
                "Fix profit_target or lots in config/settings.py."
            )
            return

        # Connect
        if not await self.mt5.connect():
            logger.error("Failed to connect to MT5. Exiting.")
            return

        # Set start-of-day balance
        account = await self.mt5.get_account_info()
        if account:
            self.risk.set_start_balance(account["balance"])
            logger.info(f"Account balance: ${account['balance']:.2f}")

        # Log pair configs
        for cfg in PAIRS:
            session = "24/7" if cfg.session_hours is None else str(cfg.session_hours)
            logger.info(
                f"  Pair: {cfg.leg_a}/{cfg.leg_b} | "
                f"lots: {cfg.leg_a_lot}/{cfg.leg_b_lot} | "
                f"Z-entry: {cfg.zscore_entry} | session: {session}"
            )

        logger.info(f"Scan interval: {settings.scan_interval_seconds}s")
        logger.info(f"Max open pairs: {settings.max_open_pairs}")
        logger.info(f"Daily loss limit: ${settings.max_daily_loss}")
        if settings.require_cointegration:
            logger.info(
                f"Cointegration filter: ON "
                f"(p<={settings.coint_p_threshold}, "
                f"lookback={settings.coint_lookback_bars} {settings.coint_timeframe} bars, "
                f"recheck={settings.coint_recheck_seconds}s, "
                f"half-life∈[{settings.coint_min_half_life},{settings.coint_max_half_life}])"
            )
        else:
            logger.warning("Cointegration filter: OFF")
        logger.info("-" * 60)

        self._running = True

        # Kick off background cointegration recheck loop
        self._coint_task = asyncio.create_task(
            self.coint_monitor.run_periodic(lambda: not self._running)
        )

        await self._main_loop()

    async def _main_loop(self):
        """Core loop — scan, signal, execute, monitor."""
        cycle = 0

        while self._running:
            try:
                cycle += 1
                now = datetime.now(timezone.utc).strftime("%H:%M:%S")

                # --- 1. Compute spreads for all pairs ---
                for tracker in self.trackers:
                    pair_name = f"{tracker.cfg.leg_a}/{tracker.cfg.leg_b}"
                    coint_state = self.coint_monitor.get_state(pair_name)
                    state = await tracker.compute_spread(
                        timeframe=settings.timeframe,
                        lookback=settings.spread_lookback,
                        coint_state=coint_state,
                    )

                    if state is None:
                        continue  # out of session or no data

                    is_open = self.execution.has_open_pair(state.pair_name)
                    pnl = await self.execution.get_pair_pnl(state.pair_name) if is_open else None

                    # Log spread state
                    status = "OPEN" if is_open else "    "
                    pnl_str = f" P&L=${pnl:.2f}" if pnl is not None else ""
                    logger.info(
                        f"[{now}] {status} {state.pair_name} | "
                        f"Z={state.zscore:+.2f} | "
                        f"corr={state.correlation:.3f} | "
                        f"ratio={state.ratio:.4f}{pnl_str}"
                    )

                    # --- 2. Generate signal (Z-score stop, entries) ---
                    # Re-use the cointegration state we already looked up so
                    # spread Z-score and hedge sizing stay consistent.
                    signal_obj = tracker.generate_signal(
                        state, is_open, coint_state, IC_MARKETS_RAW
                    )

                    if signal_obj is None:
                        continue

                    # --- 4. Risk check (for entries) ---
                    if signal_obj.action.startswith("OPEN"):
                        ok, reason = self.risk.can_open_pair(state)
                        if not ok:
                            logger.warning(
                                f"Risk blocked {state.pair_name}: {reason}"
                            )
                            continue

                        # Runtime cost check — beta sizing can change leg_b
                        # enough to push the trade below the safety factor.
                        cost_ok, total_cost, ratio = check_trade_viability(
                            IC_MARKETS_RAW,
                            signal_obj.leg_a,
                            signal_obj.leg_b,
                            signal_obj.leg_a_lot,
                            signal_obj.leg_b_lot,
                            tracker.cfg.profit_target,
                            settings.profit_target_safety_factor,
                        )
                        if not cost_ok:
                            logger.warning(
                                f"Cost blocked {state.pair_name}: cost=${total_cost:.2f}, "
                                f"target=${tracker.cfg.profit_target:.2f}, "
                                f"ratio={ratio:.2f}x"
                            )
                            continue

                    # --- 5. Execute ---
                    logger.info(f">>> SIGNAL: {signal_obj.action} | {signal_obj.reason}")
                    await self.execution.execute_signal(signal_obj)

                # --- 5. Profit target check (independent of spread data) ---
                await self._check_profit_targets()

                # --- 6. Check timeout on active pairs ---
                await self._check_timeouts()

                # --- 6. Risk monitoring ---
                warnings = await self.risk.check_active_pairs()
                for w in warnings:
                    logger.warning(f"RISK: {w}")

                # Daily loss circuit breaker
                daily_pnl = self.execution.get_daily_pnl()
                if daily_pnl <= -settings.max_daily_loss:
                    logger.error(
                        f"DAILY LOSS LIMIT HIT: ${daily_pnl:.2f} — closing all pairs"
                    )
                    await self.execution.close_all("daily_loss_limit")
                    self._running = False
                    break

                # --- 7. Periodic stats ---
                if cycle % 10 == 0:
                    stats = self.execution.get_stats()
                    active = len(self.execution.active_pairs)
                    logger.info(
                        f"[STATS] Active: {active} | "
                        f"Trades: {stats['trades']} | "
                        f"P&L: ${stats['total_pnl']} | "
                        f"WR: {stats['win_rate']}% | "
                        f"Daily: ${daily_pnl:.2f}"
                    )

            except Exception as e:
                logger.exception(f"Error in main loop: {e}")

            # Wait for next scan
            await asyncio.sleep(settings.scan_interval_seconds)

        # Shutdown
        await self.stop()

    async def _check_profit_targets(self):
        """Close pairs that hit their profit target — runs independently of spread data."""
        profit_target_map = {f"{c.leg_a}/{c.leg_b}": c.profit_target for c in PAIRS}

        for pair_name in list(self.execution.active_pairs.keys()):
            pnl = await self.execution.get_pair_pnl(pair_name)
            if pnl is None:
                continue
            target = profit_target_map.get(pair_name, 1.0)
            if pnl >= target:
                pair = self.execution.active_pairs[pair_name]
                from src.schemas import PairSignal
                sig = PairSignal(
                    pair_name=pair_name, action="CLOSE", zscore=0,
                    leg_a=pair.leg_a, leg_b=pair.leg_b,
                    leg_a_side="", leg_b_side="",
                    leg_a_lot=0, leg_b_lot=0,
                    reason=f"Profit target hit (P&L=${pnl:.2f} >= ${target:.2f})",
                )
                logger.info(f">>> PROFIT TARGET | {sig.reason}")
                await self.execution.execute_signal(sig)

    async def _check_timeouts(self):
        """Close pairs that exceeded max hold time."""
        now = datetime.now(timezone.utc)

        # Build map of pair_name → max_hold_minutes
        timeout_map = {}
        for cfg in PAIRS:
            name = f"{cfg.leg_a}/{cfg.leg_b}"
            timeout_map[name] = cfg.max_hold_minutes

        for pair_name, pair in list(self.execution.active_pairs.items()):
            hold_min = (now - pair.entry_time).total_seconds() / 60
            max_hold = timeout_map.get(pair_name, 60)

            if hold_min >= max_hold:
                logger.warning(
                    f"Timeout: {pair_name} held {hold_min:.0f}m > {max_hold}m limit"
                )
                from src.schemas import PairSignal

                timeout_signal = PairSignal(
                    pair_name=pair_name,
                    action="CLOSE",
                    zscore=0,
                    leg_a=pair.leg_a,
                    leg_b=pair.leg_b,
                    leg_a_side="",
                    leg_b_side="",
                    leg_a_lot=0,
                    leg_b_lot=0,
                    reason=f"Timeout ({hold_min:.0f}m > {max_hold}m)",
                )
                await self.execution.execute_signal(timeout_signal)

    async def stop(self):
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self._running = False

        # Cancel background cointegration recheck
        if self._coint_task is not None and not self._coint_task.done():
            self._coint_task.cancel()
            try:
                await self._coint_task
            except (asyncio.CancelledError, Exception):
                pass

        # Close all active pairs
        if self.execution.active_pairs:
            logger.info(
                f"Closing {len(self.execution.active_pairs)} active pair(s)..."
            )
            await self.execution.close_all("shutdown")

        # Print final stats
        stats = self.execution.get_stats()
        logger.info("=" * 60)
        logger.info("  SESSION SUMMARY")
        logger.info(f"  Trades: {stats['trades']}")
        logger.info(f"  Total P&L: ${stats.get('total_pnl', 0)}")
        logger.info(f"  Win Rate: {stats.get('win_rate', 0)}%")
        if stats['trades'] > 0:
            logger.info(f"  Avg P&L: ${stats.get('avg_profit', 0)}")
            logger.info(f"  Avg Hold: {stats.get('avg_hold_min', 0)}m")
            logger.info(f"  Best: ${stats.get('best', 0)} | Worst: ${stats.get('worst', 0)}")
        logger.info("=" * 60)

        await self.mt5.disconnect()


# --- Entry point ---

bot = ArbitrageBot()


def _signal_handler(sig, frame):
    logger.info("Interrupt received, stopping...")
    bot._running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


if __name__ == "__main__":
    asyncio.run(bot.start())

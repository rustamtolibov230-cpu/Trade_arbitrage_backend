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
        self.risk = RiskManager(self.execution)
        self.trackers = [SpreadTracker(cfg, self.mt5) for cfg in PAIRS]
        self._running = False

    async def start(self):
        """Connect to MT5 and start the main loop."""
        logger.info("=" * 60)
        logger.info("  TRADE ARBITRAGE BOT — Starting")
        logger.info("=" * 60)

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
        logger.info("-" * 60)

        self._running = True
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
                    state = await tracker.compute_spread(
                        timeframe=settings.timeframe,
                        lookback=settings.spread_lookback,
                    )

                    if state is None:
                        continue  # out of session or no data

                    is_open = self.execution.has_open_pair(state.pair_name)

                    # Log spread state
                    status = "OPEN" if is_open else "    "
                    logger.info(
                        f"[{now}] {status} {state.pair_name} | "
                        f"Z={state.zscore:+.2f} | "
                        f"corr={state.correlation:.3f} | "
                        f"ratio={state.ratio:.4f}"
                    )

                    # --- 2. Generate signal ---
                    signal_obj = tracker.generate_signal(state, is_open)

                    if signal_obj is None:
                        continue

                    # --- 3. Risk check (for entries) ---
                    if signal_obj.action.startswith("OPEN"):
                        ok, reason = self.risk.can_open_pair(state)
                        if not ok:
                            logger.warning(
                                f"Risk blocked {state.pair_name}: {reason}"
                            )
                            continue

                    # --- 4. Execute ---
                    logger.info(f">>> SIGNAL: {signal_obj.action} | {signal_obj.reason}")
                    await self.execution.execute_signal(signal_obj)

                # --- 5. Check timeout on active pairs ---
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

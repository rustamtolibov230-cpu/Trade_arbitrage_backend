"""Execution engine — opens and closes hedged pairs."""

from datetime import datetime, timezone
from typing import Dict, Optional, List

from loguru import logger

from src.mt5_client import MT5Client
from src.schemas import PairSignal, ActivePair, TradeResult


class ExecutionEngine:
    """Manages opening and closing of pair trades."""

    def __init__(self, mt5: MT5Client):
        self.mt5 = mt5
        self.active_pairs: Dict[str, ActivePair] = {}
        self.trade_history: List[TradeResult] = []

    def has_open_pair(self, pair_name: str) -> bool:
        return pair_name in self.active_pairs

    async def execute_signal(self, signal: PairSignal) -> bool:
        """Execute a pair signal (open or close)."""

        if signal.action in ("CLOSE", "STOP"):
            return await self._close_pair(signal)

        if signal.action in ("OPEN_LONG_SPREAD", "OPEN_SHORT_SPREAD"):
            return await self._open_pair(signal)

        logger.warning(f"Unknown signal action: {signal.action}")
        return False

    async def _open_pair(self, signal: PairSignal) -> bool:
        """Open both legs of a pair trade."""

        if signal.pair_name in self.active_pairs:
            logger.warning(f"Already have open pair: {signal.pair_name}")
            return False

        logger.info(
            f"Opening pair {signal.pair_name}: "
            f"{signal.leg_a_side} {signal.leg_a_lot} {signal.leg_a} / "
            f"{signal.leg_b_side} {signal.leg_b_lot} {signal.leg_b} "
            f"| Z={signal.zscore:.2f} | {signal.reason}"
        )

        # Open leg A
        ticket_a = await self.mt5.open_order(
            symbol=signal.leg_a,
            order_type=signal.leg_a_side,
            lot=signal.leg_a_lot,
            comment=f"arb_{signal.pair_name}_A",
        )
        if ticket_a is None:
            logger.error(f"Failed to open leg A: {signal.leg_a}")
            return False

        # Open leg B
        ticket_b = await self.mt5.open_order(
            symbol=signal.leg_b,
            order_type=signal.leg_b_side,
            lot=signal.leg_b_lot,
            comment=f"arb_{signal.pair_name}_B",
        )
        if ticket_b is None:
            # Leg A opened but leg B failed — close leg A to avoid unhedged exposure
            logger.error(f"Failed to open leg B: {signal.leg_b}, closing leg A")
            await self.mt5.close_order(
                ticket_a, signal.leg_a, signal.leg_a_lot, signal.leg_a_side
            )
            return False

        # Track the pair
        self.active_pairs[signal.pair_name] = ActivePair(
            pair_name=signal.pair_name,
            leg_a=signal.leg_a,
            leg_b=signal.leg_b,
            leg_a_ticket=ticket_a,
            leg_b_ticket=ticket_b,
            leg_a_side=signal.leg_a_side,
            leg_b_side=signal.leg_b_side,
            leg_a_lot=signal.leg_a_lot,
            leg_b_lot=signal.leg_b_lot,
            entry_zscore=signal.zscore,
        )

        logger.info(
            f"Pair opened: {signal.pair_name} | "
            f"tickets=({ticket_a}, {ticket_b})"
        )
        return True

    async def _close_pair(self, signal: PairSignal) -> bool:
        """Close both legs of a pair trade."""

        pair = self.active_pairs.get(signal.pair_name)
        if pair is None:
            logger.warning(f"No active pair to close: {signal.pair_name}")
            return False

        logger.info(
            f"Closing pair {signal.pair_name}: {signal.reason} | Z={signal.zscore:.2f}"
        )

        # Get P&L before closing
        profit_a = await self.mt5.get_position_profit(pair.leg_a_ticket) or 0.0
        profit_b = await self.mt5.get_position_profit(pair.leg_b_ticket) or 0.0
        total_profit = profit_a + profit_b

        # Close both legs
        closed_a = await self.mt5.close_order(
            pair.leg_a_ticket, pair.leg_a, pair.leg_a_lot, pair.leg_a_side
        )
        closed_b = await self.mt5.close_order(
            pair.leg_b_ticket, pair.leg_b, pair.leg_b_lot, pair.leg_b_side
        )

        if not closed_a or not closed_b:
            logger.error(
                f"Partial close! A={closed_a}, B={closed_b} — MANUAL CHECK NEEDED"
            )

        # Record result
        now = datetime.now(timezone.utc)
        hold_minutes = (now - pair.entry_time).total_seconds() / 60

        reason_map = {"CLOSE": "mean_revert", "STOP": "stop_loss"}
        result = TradeResult(
            pair_name=signal.pair_name,
            profit=total_profit,
            hold_minutes=hold_minutes,
            entry_zscore=pair.entry_zscore,
            exit_zscore=signal.zscore,
            entry_time=pair.entry_time,
            exit_time=now,
            reason=reason_map.get(signal.action, signal.action),
        )
        self.trade_history.append(result)

        # Remove from active
        del self.active_pairs[signal.pair_name]

        logger.info(
            f"Pair closed: {signal.pair_name} | P&L=${total_profit:.2f} | "
            f"held {hold_minutes:.1f}m | {result.reason}"
        )
        return True

    async def get_pair_pnl(self, pair_name: str) -> Optional[float]:
        """Get combined P&L for an active pair."""
        pair = self.active_pairs.get(pair_name)
        if pair is None:
            return None

        profit_a = await self.mt5.get_position_profit(pair.leg_a_ticket) or 0.0
        profit_b = await self.mt5.get_position_profit(pair.leg_b_ticket) or 0.0
        return profit_a + profit_b

    async def close_all(self, reason: str = "manual") -> List[TradeResult]:
        """Close all active pairs."""
        results = []
        for pair_name in list(self.active_pairs.keys()):
            signal = PairSignal(
                pair_name=pair_name,
                action="CLOSE",
                zscore=0,
                leg_a="",
                leg_b="",
                leg_a_side="",
                leg_b_side="",
                leg_a_lot=0,
                leg_b_lot=0,
                reason=reason,
            )
            await self._close_pair(signal)
            if self.trade_history:
                results.append(self.trade_history[-1])
        return results

    def get_daily_pnl(self) -> float:
        """Sum of today's closed trade profits."""
        today = datetime.now(timezone.utc).date()
        return sum(
            t.profit
            for t in self.trade_history
            if t.exit_time.date() == today
        )

    def get_stats(self) -> dict:
        """Summary statistics."""
        if not self.trade_history:
            return {"trades": 0, "total_pnl": 0, "win_rate": 0}

        profits = [t.profit for t in self.trade_history]
        wins = sum(1 for p in profits if p > 0)
        return {
            "trades": len(profits),
            "total_pnl": round(sum(profits), 2),
            "win_rate": round(wins / len(profits) * 100, 1),
            "avg_profit": round(sum(profits) / len(profits), 2),
            "avg_hold_min": round(
                sum(t.hold_minutes for t in self.trade_history) / len(self.trade_history), 1
            ),
            "best": round(max(profits), 2),
            "worst": round(min(profits), 2),
        }

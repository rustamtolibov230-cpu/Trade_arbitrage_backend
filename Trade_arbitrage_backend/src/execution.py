"""Execution engine — opens and closes hedged pairs."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, List

from loguru import logger

from src.mt5_client import MT5Client
from src.schemas import PairSignal, ActivePair, TradeResult


HISTORY_FILE = Path("data") / "trade_history.json"


class ExecutionEngine:
    """Manages opening and closing of pair trades."""

    def __init__(self, mt5: MT5Client):
        self.mt5 = mt5
        self.active_pairs: Dict[str, ActivePair] = {}
        self.trade_history: List[TradeResult] = []
        self._last_close_time: Dict[str, datetime] = {}
        self._load_history()
        # Seed cooldown from persisted history so a restart doesn't bypass it
        for t in self.trade_history:
            prev = self._last_close_time.get(t.pair_name)
            if prev is None or t.exit_time > prev:
                self._last_close_time[t.pair_name] = t.exit_time

    def cooldown_remaining(self, pair_name: str, cooldown_seconds: int) -> float:
        """Seconds left before this pair can be reopened. 0 if ready."""
        last = self._last_close_time.get(pair_name)
        if last is None or cooldown_seconds <= 0:
            return 0.0
        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        return max(0.0, cooldown_seconds - elapsed)

    def _load_history(self) -> None:
        if not HISTORY_FILE.exists():
            return
        try:
            raw = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to read {HISTORY_FILE}: {e} — starting with empty history")
            return

        for item in raw:
            try:
                self.trade_history.append(TradeResult(
                    pair_name=item["pair_name"],
                    profit=float(item["profit"]),
                    hold_minutes=float(item["hold_minutes"]),
                    entry_zscore=float(item["entry_zscore"]),
                    exit_zscore=float(item["exit_zscore"]),
                    entry_time=datetime.fromisoformat(item["entry_time"]),
                    exit_time=datetime.fromisoformat(item["exit_time"]),
                    reason=item["reason"],
                ))
            except Exception as e:
                logger.warning(f"Skipping bad trade record: {e} | {item}")
        logger.info(f"Loaded {len(self.trade_history)} historical trades from {HISTORY_FILE}")

    def _save_history(self) -> None:
        try:
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = [
                {
                    "pair_name": t.pair_name,
                    "profit": t.profit,
                    "hold_minutes": t.hold_minutes,
                    "entry_zscore": t.entry_zscore,
                    "exit_zscore": t.exit_zscore,
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "reason": t.reason,
                }
                for t in self.trade_history
            ]
            tmp = HISTORY_FILE.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(HISTORY_FILE)
        except Exception as e:
            logger.error(f"Failed to save trade history: {e}")

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
            logger.error(
                f"Failed to open leg B ({signal.leg_b}) — rolling back leg A ({signal.leg_a} ticket={ticket_a})"
            )
            closed = await self.mt5.close_order(
                ticket_a, signal.leg_a, signal.leg_a_lot, signal.leg_a_side
            )
            if not closed:
                logger.error(
                    f"!!! ORPHAN POSITION !!! rollback close of {signal.leg_a} ticket={ticket_a} FAILED — MANUAL CHECK NEEDED"
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
        self._last_close_time[signal.pair_name] = now
        self._save_history()

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

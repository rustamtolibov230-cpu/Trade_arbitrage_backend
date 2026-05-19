"""Risk manager — guards against runaway losses and correlation breaks."""

from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from config.settings import settings
from src.cointegration import CointegrationMonitor
from src.execution import ExecutionEngine
from src.schemas import SpreadState


class RiskManager:
    """Validates pair trades against risk limits."""

    def __init__(
        self,
        execution: ExecutionEngine,
        coint: Optional[CointegrationMonitor] = None,
    ):
        self.execution = execution
        self.coint = coint
        self._start_balance: Optional[float] = None

    def set_start_balance(self, balance: float):
        self._start_balance = balance
        logger.info(f"Risk manager: start balance = ${balance:.2f}")

    def can_open_pair(self, spread_state: SpreadState) -> tuple[bool, str]:
        """Check if we're allowed to open a new pair."""

        # Max pairs limit
        active_count = len(self.execution.active_pairs)
        if active_count >= settings.max_open_pairs:
            return False, f"Max pairs reached ({active_count}/{settings.max_open_pairs})"

        # Already have this pair open
        if self.execution.has_open_pair(spread_state.pair_name):
            return False, f"Pair already open: {spread_state.pair_name}"

        # Correlation too low
        if spread_state.correlation < settings.min_correlation:
            return False, (
                f"Correlation too low: {spread_state.correlation:.3f} "
                f"< {settings.min_correlation}"
            )

        # Cointegration filter — correlation alone doesn't guarantee the
        # spread is mean-reverting. Block entries on pairs whose spread isn't
        # stationary (or whose half-life is outside the tradable band).
        if self.coint is not None:
            ok, reason = self.coint.is_tradeable(spread_state.pair_name)
            if not ok:
                return False, f"Cointegration: {reason}"

        # Daily loss limit
        daily_pnl = self.execution.get_daily_pnl()
        if daily_pnl <= -settings.max_daily_loss:
            return False, f"Daily loss limit hit: ${daily_pnl:.2f}"

        return True, "OK"

    async def check_active_pairs(self) -> list[str]:
        """Monitor active pairs for risk breaches. Returns list of warnings."""
        warnings = []

        for pair_name, pair in list(self.execution.active_pairs.items()):
            pnl = await self.execution.get_pair_pnl(pair_name)
            if pnl is not None:
                # Track max/min P&L
                if pnl > pair.max_profit:
                    pair.max_profit = pnl
                if pnl < pair.min_profit:
                    pair.min_profit = pnl

            # Check hold time
            now = datetime.now(timezone.utc)
            hold_min = (now - pair.entry_time).total_seconds() / 60

            # Log active pair status
            pnl_str = f"${pnl:.2f}" if pnl is not None else "N/A"
            logger.debug(
                f"  {pair_name}: P&L={pnl_str} | held {hold_min:.0f}m | "
                f"max=${pair.max_profit:.2f} min=${pair.min_profit:.2f}"
            )

        # Check total daily P&L
        daily_pnl = self.execution.get_daily_pnl()
        if daily_pnl <= -settings.max_daily_loss * 0.8:
            warnings.append(
                f"Daily P&L approaching limit: ${daily_pnl:.2f} "
                f"(limit: -${settings.max_daily_loss})"
            )

        return warnings

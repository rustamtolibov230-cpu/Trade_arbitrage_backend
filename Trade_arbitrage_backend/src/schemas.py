"""Data models for pairs trading."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SpreadState:
    """Current state of a pair's spread.

    The Z-score uses one of two definitions depending on cointegration state:
      - "residual": (price_a - alpha - beta * price_b) — preferred, mathematically
        consistent with the Engle-Granger filter
      - "ratio":    (price_a / price_b)               — fallback when no
        cointegration state is available yet

    ``spread_value`` holds the value that was Z-scored. ``ratio`` is always the
    price ratio (for display) regardless of mode.
    """

    pair_name: str  # e.g. "BTCUSD/ETHUSD"
    leg_a: str
    leg_b: str
    ratio: float  # price_a / price_b — always populated for display
    mean: float
    std: float
    zscore: float
    correlation: float
    hedge_ratio: float  # ATR-adjusted lot ratio (legacy)
    spread_mode: str = "ratio"   # "residual" or "ratio"
    spread_value: float = 0.0    # the current value of the Z-scored spread
    timestamp: datetime = field(default_factory=_utcnow)


@dataclass
class PairSignal:
    """Signal to open or close a pair."""

    pair_name: str
    action: str  # "OPEN_LONG_SPREAD" | "OPEN_SHORT_SPREAD" | "CLOSE" | "STOP"
    zscore: float
    leg_a: str
    leg_b: str
    leg_a_side: str  # "BUY" or "SELL"
    leg_b_side: str
    leg_a_lot: float
    leg_b_lot: float
    reason: str = ""


@dataclass
class ActivePair:
    """A currently open pair trade."""

    pair_name: str
    leg_a: str
    leg_b: str
    leg_a_ticket: int
    leg_b_ticket: int
    leg_a_side: str
    leg_b_side: str
    leg_a_lot: float
    leg_b_lot: float
    entry_zscore: float
    entry_time: datetime = field(default_factory=_utcnow)
    max_profit: float = 0.0
    min_profit: float = 0.0


@dataclass
class TradeResult:
    """Result of a closed pair trade."""

    pair_name: str
    profit: float
    hold_minutes: float
    entry_zscore: float
    exit_zscore: float
    entry_time: datetime
    exit_time: datetime
    reason: str  # "mean_revert" | "stop_loss" | "timeout" | "manual"


@dataclass
class CointegrationState:
    """Result of a cointegration check for a pair.

    A pair is considered tradeable only if:
      - p_value <= threshold (Engle-Granger rejects "no cointegration")
      - half_life_bars within [min_half_life, max_half_life]
        (too fast = noise, too slow = won't revert within our hold window)
    """

    pair_name: str
    is_cointegrated: bool
    p_value: float          # Engle-Granger p-value (lower = more cointegrated)
    beta: float             # OLS hedge ratio: price_a ≈ alpha + beta * price_b
    alpha: float            # OLS intercept (for residual spread calculation)
    half_life_bars: float   # mean-reversion half-life, in bars of coint_timeframe
    bars_used: int          # sample size of the test
    reason: str = ""        # human-readable rejection reason if not tradeable
    last_check: datetime = field(default_factory=_utcnow)

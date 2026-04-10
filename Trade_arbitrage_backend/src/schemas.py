"""Data models for pairs trading."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class SpreadState:
    """Current state of a pair's spread."""

    pair_name: str  # e.g. "BTCUSD/ETHUSD"
    leg_a: str
    leg_b: str
    ratio: float  # price_a / price_b
    mean: float
    std: float
    zscore: float
    correlation: float
    hedge_ratio: float  # ATR-adjusted lot ratio
    timestamp: datetime = field(default_factory=datetime.utcnow)


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
    entry_time: datetime = field(default_factory=datetime.utcnow)
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

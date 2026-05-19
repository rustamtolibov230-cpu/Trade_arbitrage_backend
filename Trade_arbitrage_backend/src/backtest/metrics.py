"""Performance metrics for simulated trade lists."""

from dataclasses import dataclass, asdict
from typing import List, Dict

import numpy as np


@dataclass
class Metrics:
    trades: int
    wins: int
    losses: int
    win_rate: float           # %
    total_pnl: float
    avg_pnl: float
    best: float
    worst: float
    avg_hold_min: float
    profit_factor: float      # gross_profit / gross_loss
    sharpe: float             # annualized, per-trade basis
    sortino: float            # annualized, per-trade basis
    max_drawdown: float       # peak-to-trough equity drop
    max_drawdown_pct: float   # as % of peak equity
    expectancy: float         # avg profit per trade (same as avg_pnl but explicit)
    trades_per_day: float
    gross_profit: float
    gross_loss: float

    def to_dict(self) -> Dict:
        return asdict(self)


def compute_metrics(trades: List, bars_covered_minutes: float = 0) -> Metrics:
    """Compute headline metrics from a list of SimTrade-like objects.

    Accepts any object with attributes: profit, hold_minutes.
    """
    n = len(trades)
    if n == 0:
        return Metrics(
            trades=0, wins=0, losses=0, win_rate=0.0,
            total_pnl=0.0, avg_pnl=0.0, best=0.0, worst=0.0,
            avg_hold_min=0.0, profit_factor=0.0, sharpe=0.0, sortino=0.0,
            max_drawdown=0.0, max_drawdown_pct=0.0, expectancy=0.0,
            trades_per_day=0.0, gross_profit=0.0, gross_loss=0.0,
        )

    profits = np.array([t.profit for t in trades], dtype=float)
    holds = np.array([t.hold_minutes for t in trades], dtype=float)

    wins_mask = profits > 0
    losses_mask = profits < 0
    wins = int(wins_mask.sum())
    losses = int(losses_mask.sum())

    gross_profit = float(profits[wins_mask].sum()) if wins else 0.0
    gross_loss = float(-profits[losses_mask].sum()) if losses else 0.0  # positive number
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    # Sharpe / Sortino on per-trade returns (not time-weighted).
    # Annualization factor assumes "trades per year"; we use sqrt(N) heuristic.
    mean = float(profits.mean())
    std = float(profits.std(ddof=1)) if n > 1 else 0.0
    downside = profits[profits < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0

    # Per-trade Sharpe — multiplied by sqrt(trades_per_year).
    # If we know time covered we could be precise; here use trades_per_day
    # to approximate annualization.
    trades_per_day = (n / (bars_covered_minutes / 1440.0)) if bars_covered_minutes > 0 else 0.0
    ann_factor = np.sqrt(trades_per_day * 252) if trades_per_day > 0 else np.sqrt(n)

    sharpe = (mean / std * ann_factor) if std > 0 else 0.0
    sortino = (mean / downside_std * ann_factor) if downside_std > 0 else 0.0

    # Max drawdown on cumulative equity curve
    equity = np.cumsum(profits)
    # Handle negative starts: prepend 0 so drawdown is from a starting capital baseline
    equity_curve = np.concatenate(([0.0], equity))
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve - running_max
    max_dd = float(-drawdown.min())  # positive number
    peak = float(running_max.max()) if running_max.max() > 0 else 1.0
    max_dd_pct = (max_dd / peak * 100.0) if peak > 0 else 0.0

    return Metrics(
        trades=n,
        wins=wins,
        losses=losses,
        win_rate=round(wins / n * 100.0, 2),
        total_pnl=round(float(profits.sum()), 2),
        avg_pnl=round(mean, 4),
        best=round(float(profits.max()), 2),
        worst=round(float(profits.min()), 2),
        avg_hold_min=round(float(holds.mean()), 1),
        profit_factor=round(profit_factor, 3),
        sharpe=round(sharpe, 3),
        sortino=round(sortino, 3),
        max_drawdown=round(max_dd, 2),
        max_drawdown_pct=round(max_dd_pct, 2),
        expectancy=round(mean, 4),
        trades_per_day=round(trades_per_day, 3),
        gross_profit=round(gross_profit, 2),
        gross_loss=round(gross_loss, 2),
    )

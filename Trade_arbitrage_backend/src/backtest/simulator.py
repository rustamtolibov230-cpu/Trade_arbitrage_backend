"""Bar-by-bar replay of the pairs-trading strategy.

The simulator mirrors the live logic from server.py / spread_tracker.py:
  - Periodic cointegration refit on a rolling window
  - Spread Z-score on residual when cointegrated, ratio otherwise
  - OLS-beta hedge sizing for leg_b
  - Cost-viability gate before each open (same as live)
  - Exits: profit target (with min_hold grace), Z-score stop, max-hold timeout
  - Cooldown after close
  - Daily loss circuit breaker
Costs use the existing cost_model — half-spread fills + per-side commission.

Inputs (per simulate() call):
  - aligned bars DataFrame: time, a_open/high/low/close, b_open/high/low/close
  - PairConfig
  - BrokerCostModel
  - SimSettings (mirrors the runtime settings)

Outputs:
  - list[SimTrade]
  - SimResult: trades + equity curve + metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import PairConfig, settings as _runtime_settings
from src.backtest.cost_model import (
    BrokerCostModel,
    round_trip_cost_for_pair,
)
from src.backtest.metrics import Metrics, compute_metrics
from src.cointegration import CointegrationMonitor
from src.schemas import CointegrationState


# Map MT5 timeframe strings to minutes per bar.
_TF_MINUTES = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440,
}


@dataclass
class SimSettings:
    """Settings the simulator needs that mirror the live `Settings`."""

    timeframe: str = "M1"
    spread_lookback: int = 50
    coint_lookback_bars: int = 500
    coint_recheck_minutes: int = 30
    coint_p_threshold: float = 0.05
    coint_min_half_life: float = 2.0
    coint_max_half_life: float = 100.0
    require_cointegration: bool = True
    min_correlation: float = 0.80
    cooldown_seconds: int = 300
    max_daily_loss: float = 50.0
    profit_target_safety_factor: float = 1.5


@dataclass
class SimTrade:
    """A single simulated round-trip."""

    pair_name: str
    side: str                  # "LONG_SPREAD" or "SHORT_SPREAD"
    entry_time: datetime
    exit_time: datetime
    entry_zscore: float
    exit_zscore: float
    leg_a_lot: float
    leg_b_lot: float
    entry_price_a: float
    entry_price_b: float
    exit_price_a: float
    exit_price_b: float
    gross_pnl: float           # before costs
    costs: float               # spread + commission, both legs
    profit: float              # gross_pnl - costs (net)
    hold_minutes: float
    reason: str                # "profit_target" | "z_stop" | "timeout" | "end_of_data"


@dataclass
class SimResult:
    pair_name: str
    bars_used: int
    bars_minutes: float
    trades: List[SimTrade]
    equity_curve: pd.DataFrame   # columns: time, cum_pnl
    metrics: Metrics
    cost_blocks: int             # trades skipped by viability check
    coint_blocks: int            # entries skipped because pair not cointegrated
    corr_blocks: int             # entries skipped because corr below threshold


class Simulator:
    """Backtest one pair over a window of aligned bars."""

    def __init__(
        self,
        cfg: PairConfig,
        cost_model: BrokerCostModel,
        sim_settings: SimSettings,
    ):
        self.cfg = cfg
        self.cost_model = cost_model
        self.s = sim_settings
        self.pair_name = f"{cfg.leg_a}/{cfg.leg_b}"
        # Pure-CPU CointegrationMonitor: pairs/mt5 unused by _compute_state.
        self._coint = CointegrationMonitor([], mt5=None)
        # Cache cost lookups
        self._cost_a = cost_model.require(cfg.leg_a)
        self._cost_b = cost_model.require(cfg.leg_b)
        self._bar_minutes = _TF_MINUTES.get(sim_settings.timeframe, 1)

    # ---------------------------------------------------------------- public

    def simulate(self, bars: pd.DataFrame) -> SimResult:
        """Replay bar-by-bar. ``bars`` must have time, a_*, b_* OHLC columns."""
        required = {"time", "a_close", "b_close", "a_high", "a_low", "b_high", "b_low"}
        missing = required - set(bars.columns)
        if missing:
            raise ValueError(f"bars missing columns: {missing}")

        n = len(bars)
        coint_lb = self.s.coint_lookback_bars
        spread_lb = self.s.spread_lookback
        start = max(coint_lb, spread_lb)
        if n <= start:
            raise ValueError(
                f"Need at least {start + 1} bars; got {n}. Increase --days or lower lookbacks."
            )

        recheck_bars = max(1, self.s.coint_recheck_minutes // max(1, self._bar_minutes))

        a_close = bars["a_close"].to_numpy(dtype=float)
        b_close = bars["b_close"].to_numpy(dtype=float)
        times = pd.to_datetime(bars["time"]).to_list()

        coint_state: Optional[CointegrationState] = None
        last_coint_fit_idx = -10**9

        open_trade: Optional[_OpenTrade] = None
        trades: List[SimTrade] = []
        last_close_idx = -10**9  # for cooldown
        cost_blocks = coint_blocks = corr_blocks = 0
        equity_times: List[datetime] = []
        equity_vals: List[float] = []
        cum_pnl = 0.0
        daily_pnl = 0.0
        daily_pnl_date = times[start].date() if start < n else None
        circuit_tripped = False

        # Patch settings used inside CointegrationMonitor._compute_state — it
        # reads thresholds off the global ``settings`` instance.
        original = {
            "coint_p_threshold": _runtime_settings.coint_p_threshold,
            "coint_min_half_life": _runtime_settings.coint_min_half_life,
            "coint_max_half_life": _runtime_settings.coint_max_half_life,
        }
        _runtime_settings.coint_p_threshold = self.s.coint_p_threshold
        _runtime_settings.coint_min_half_life = self.s.coint_min_half_life
        _runtime_settings.coint_max_half_life = self.s.coint_max_half_life

        try:
            for i in range(start, n):
                t_i = times[i]

                # Reset daily P&L at date change
                if daily_pnl_date is None or t_i.date() != daily_pnl_date:
                    daily_pnl_date = t_i.date()
                    daily_pnl = 0.0
                    circuit_tripped = False

                # 1. Refit cointegration on a rolling window (no lookahead).
                if i - last_coint_fit_idx >= recheck_bars:
                    a_win = a_close[i - coint_lb:i]
                    b_win = b_close[i - coint_lb:i]
                    coint_state = self._coint._compute_state(
                        self.pair_name, a_win, b_win
                    )
                    last_coint_fit_idx = i

                # 2. Compute Z-score on the chosen spread definition.
                z_a = a_close[i - spread_lb:i]
                z_b = b_close[i - spread_lb:i]
                z, spread_value, mode = self._z_score(z_a, z_b, coint_state)
                if z is None:
                    continue
                corr = float(np.corrcoef(z_a, z_b)[0, 1])

                # 3. If a trade is open, check exits first.
                if open_trade is not None:
                    pnl_now = self._unrealized_pnl(open_trade, a_close[i], b_close[i])
                    hold_min = (i - open_trade.entry_idx) * self._bar_minutes
                    hold_sec = hold_min * 60

                    # Profit target
                    if (
                        pnl_now >= self.cfg.profit_target
                        and hold_sec >= self.cfg.min_hold_seconds
                    ):
                        closed = self._close_trade(
                            open_trade, i, t_i, a_close[i], b_close[i], z,
                            "profit_target",
                        )
                        trades.append(closed)
                        cum_pnl += closed.profit
                        daily_pnl += closed.profit
                        equity_times.append(t_i); equity_vals.append(cum_pnl)
                        open_trade = None
                        last_close_idx = i
                        if daily_pnl <= -self.s.max_daily_loss:
                            circuit_tripped = True
                        continue

                    # Z-stop
                    if abs(z) >= self.cfg.zscore_stop:
                        closed = self._close_trade(
                            open_trade, i, t_i, a_close[i], b_close[i], z, "z_stop"
                        )
                        trades.append(closed)
                        cum_pnl += closed.profit
                        daily_pnl += closed.profit
                        equity_times.append(t_i); equity_vals.append(cum_pnl)
                        open_trade = None
                        last_close_idx = i
                        if daily_pnl <= -self.s.max_daily_loss:
                            circuit_tripped = True
                        continue

                    # Timeout
                    if hold_min >= self.cfg.max_hold_minutes:
                        closed = self._close_trade(
                            open_trade, i, t_i, a_close[i], b_close[i], z, "timeout"
                        )
                        trades.append(closed)
                        cum_pnl += closed.profit
                        daily_pnl += closed.profit
                        equity_times.append(t_i); equity_vals.append(cum_pnl)
                        open_trade = None
                        last_close_idx = i
                        if daily_pnl <= -self.s.max_daily_loss:
                            circuit_tripped = True
                        continue

                # 4. No position: check entry.
                if open_trade is None:
                    if circuit_tripped:
                        continue
                    # Cooldown
                    cool_left = (
                        self.s.cooldown_seconds
                        - (i - last_close_idx) * self._bar_minutes * 60
                    )
                    if cool_left > 0:
                        continue
                    # Cointegration filter
                    if self.s.require_cointegration:
                        if coint_state is None or not coint_state.is_cointegrated:
                            coint_blocks += 1
                            continue
                    # Correlation gate
                    if corr < self.s.min_correlation:
                        corr_blocks += 1
                        continue
                    # Z-score gate
                    if not (self.cfg.zscore_entry <= abs(z) < self.cfg.zscore_entry_max):
                        continue

                    # Beta-sized lots
                    leg_a_lot, leg_b_lot = self._beta_sized_lots(coint_state)

                    # Cost gate (same as live)
                    cost_components = round_trip_cost_for_pair(
                        self.cost_model, self.cfg.leg_a, self.cfg.leg_b,
                        leg_a_lot, leg_b_lot,
                    )
                    cost = cost_components["total"]
                    ratio = self.cfg.profit_target / cost if cost > 0 else float("inf")
                    if ratio < self.s.profit_target_safety_factor:
                        cost_blocks += 1
                        continue

                    # Open: high Z → leg_a overpriced → SHORT a, LONG b
                    if z >= self.cfg.zscore_entry:
                        side_a, side_b = "SELL", "BUY"
                        side = "SHORT_SPREAD"
                    else:
                        side_a, side_b = "BUY", "SELL"
                        side = "LONG_SPREAD"

                    entry_a = self._cost_a.fill_price(a_close[i], side_a)
                    entry_b = self._cost_b.fill_price(b_close[i], side_b)
                    open_trade = _OpenTrade(
                        side=side,
                        entry_idx=i,
                        entry_time=t_i,
                        entry_z=z,
                        leg_a_lot=leg_a_lot,
                        leg_b_lot=leg_b_lot,
                        side_a=side_a,
                        side_b=side_b,
                        entry_price_a=entry_a,
                        entry_price_b=entry_b,
                    )

            # End of data: close any leftover position
            if open_trade is not None:
                i = n - 1
                z_a = a_close[i - spread_lb:i]
                z_b = b_close[i - spread_lb:i]
                z, _, _ = self._z_score(z_a, z_b, coint_state)
                closed = self._close_trade(
                    open_trade, i, times[i], a_close[i], b_close[i],
                    z if z is not None else 0.0,
                    "end_of_data",
                )
                trades.append(closed)
                cum_pnl += closed.profit
                equity_times.append(times[i]); equity_vals.append(cum_pnl)
        finally:
            # Restore patched settings
            _runtime_settings.coint_p_threshold = original["coint_p_threshold"]
            _runtime_settings.coint_min_half_life = original["coint_min_half_life"]
            _runtime_settings.coint_max_half_life = original["coint_max_half_life"]

        equity_df = pd.DataFrame({"time": equity_times, "cum_pnl": equity_vals})
        bars_minutes = (n - start) * self._bar_minutes
        metrics = compute_metrics(trades, bars_covered_minutes=bars_minutes)

        return SimResult(
            pair_name=self.pair_name,
            bars_used=n - start,
            bars_minutes=bars_minutes,
            trades=trades,
            equity_curve=equity_df,
            metrics=metrics,
            cost_blocks=cost_blocks,
            coint_blocks=coint_blocks,
            corr_blocks=corr_blocks,
        )

    # --------------------------------------------------------------- helpers

    def _z_score(
        self,
        a: np.ndarray,
        b: np.ndarray,
        coint_state: Optional[CointegrationState],
    ):
        use_resid = (
            coint_state is not None
            and coint_state.is_cointegrated
            and coint_state.beta is not None
            and coint_state.beta == coint_state.beta
            and coint_state.beta > 0
        )
        if use_resid:
            series = a - (coint_state.alpha + coint_state.beta * b)
            mode = "residual"
        else:
            series = a / b
            mode = "ratio"
        mean = float(series.mean())
        std = float(series.std())
        if std < 1e-10:
            return None, None, mode
        val = float(series[-1])
        return (val - mean) / std, val, mode

    def _beta_sized_lots(
        self, coint_state: Optional[CointegrationState]
    ) -> tuple[float, float]:
        default_a = self.cfg.leg_a_lot
        default_b = self.cfg.leg_b_lot
        if (
            coint_state is None
            or not coint_state.is_cointegrated
            or coint_state.beta is None
            or coint_state.beta != coint_state.beta
            or coint_state.beta <= 0
        ):
            return default_a, default_b
        raw_b = (
            default_a * self._cost_a.contract_size * coint_state.beta
            / self._cost_b.contract_size
        )
        step = self._cost_b.lot_step or 0.01
        rounded_b = max(self._cost_b.min_lot, round(raw_b / step) * step)
        return default_a, round(rounded_b, 8)

    def _unrealized_pnl(
        self, ot: "_OpenTrade", a_mid: float, b_mid: float
    ) -> float:
        # Adverse fill at mid (we'd cross the spread to exit)
        exit_a = self._cost_a.fill_price(
            a_mid, "SELL" if ot.side_a == "BUY" else "BUY"
        )
        exit_b = self._cost_b.fill_price(
            b_mid, "SELL" if ot.side_b == "BUY" else "BUY"
        )
        pnl_a = self._cost_a.leg_pnl(ot.side_a, ot.entry_price_a, exit_a, ot.leg_a_lot)
        pnl_b = self._cost_b.leg_pnl(ot.side_b, ot.entry_price_b, exit_b, ot.leg_b_lot)
        comm = (
            self._cost_a.round_trip_commission(ot.leg_a_lot)
            + self._cost_b.round_trip_commission(ot.leg_b_lot)
        )
        return pnl_a + pnl_b - comm

    def _close_trade(
        self,
        ot: "_OpenTrade",
        exit_idx: int,
        exit_time: datetime,
        a_mid: float,
        b_mid: float,
        exit_z: float,
        reason: str,
    ) -> SimTrade:
        exit_a = self._cost_a.fill_price(
            a_mid, "SELL" if ot.side_a == "BUY" else "BUY"
        )
        exit_b = self._cost_b.fill_price(
            b_mid, "SELL" if ot.side_b == "BUY" else "BUY"
        )
        pnl_a = self._cost_a.leg_pnl(ot.side_a, ot.entry_price_a, exit_a, ot.leg_a_lot)
        pnl_b = self._cost_b.leg_pnl(ot.side_b, ot.entry_price_b, exit_b, ot.leg_b_lot)
        comm = (
            self._cost_a.round_trip_commission(ot.leg_a_lot)
            + self._cost_b.round_trip_commission(ot.leg_b_lot)
        )
        gross = pnl_a + pnl_b
        profit = gross - comm
        hold_min = (exit_idx - ot.entry_idx) * self._bar_minutes

        return SimTrade(
            pair_name=self.pair_name,
            side=ot.side,
            entry_time=ot.entry_time,
            exit_time=exit_time,
            entry_zscore=ot.entry_z,
            exit_zscore=exit_z,
            leg_a_lot=ot.leg_a_lot,
            leg_b_lot=ot.leg_b_lot,
            entry_price_a=ot.entry_price_a,
            entry_price_b=ot.entry_price_b,
            exit_price_a=exit_a,
            exit_price_b=exit_b,
            gross_pnl=round(gross, 4),
            costs=round(comm, 4),
            profit=round(profit, 4),
            hold_minutes=float(hold_min),
            reason=reason,
        )


@dataclass
class _OpenTrade:
    """Internal — represents a position while it's still open."""
    side: str
    entry_idx: int
    entry_time: datetime
    entry_z: float
    leg_a_lot: float
    leg_b_lot: float
    side_a: str
    side_b: str
    entry_price_a: float
    entry_price_b: float

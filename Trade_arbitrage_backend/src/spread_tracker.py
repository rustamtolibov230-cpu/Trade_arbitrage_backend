"""Spread tracker — monitors price ratios and generates Z-score signals."""

from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import PairConfig
from src.mt5_client import MT5Client
from src.schemas import SpreadState, PairSignal, CointegrationState
from src.backtest.cost_model import BrokerCostModel


class SpreadTracker:
    """Tracks the spread (price ratio) between two assets and generates signals."""

    def __init__(self, pair_config: PairConfig, mt5: MT5Client):
        self.cfg = pair_config
        self.mt5 = mt5
        self.pair_name = f"{pair_config.leg_a}/{pair_config.leg_b}"

    def _is_in_session(self) -> bool:
        """Check if current time is within trading session."""
        now = datetime.now(timezone.utc)

        if self.cfg.weekdays_only and now.weekday() >= 5:  # Sat=5, Sun=6
            return False

        if self.cfg.session_hours is None:
            return True

        for start, end in self.cfg.session_hours:
            if start <= now.hour < end:
                return True
        return False

    async def compute_spread(
        self,
        timeframe: str,
        lookback: int,
        coint_state: Optional[CointegrationState] = None,
    ) -> Optional[SpreadState]:
        """Fetch prices for both legs and compute spread statistics.

        When ``coint_state`` is cointegrated with a usable beta, the Z-score is
        computed on the OLS residual ``a - alpha - beta * b`` — the same spread
        the cointegration filter validated. Otherwise we fall back to the price
        ratio so the dashboard still works before the first coint check lands.
        """
        # Session check moved to generate_signal — we always compute for display

        # Fetch bars for both legs
        bars_a = await self.mt5.get_rates(self.cfg.leg_a, timeframe, lookback + 50)
        bars_b = await self.mt5.get_rates(self.cfg.leg_b, timeframe, lookback + 50)

        if bars_a is None or bars_b is None:
            missing = []
            if bars_a is None: missing.append(self.cfg.leg_a)
            if bars_b is None: missing.append(self.cfg.leg_b)
            logger.error(
                f"{self.pair_name}: no bars for {missing} — "
                f"check symbol names in MT5 (Market Watch must have them)"
            )
            return None

        # Align by time
        merged = pd.merge(
            bars_a[["time", "close"]].rename(columns={"close": "close_a"}),
            bars_b[["time", "close"]].rename(columns={"close": "close_b"}),
            on="time",
            how="inner",
        )

        if len(merged) < lookback:
            logger.warning(
                f"{self.pair_name}: only {len(merged)} aligned bars, need {lookback}"
            )
            return None

        # Use last `lookback` bars
        merged = merged.tail(lookback).reset_index(drop=True)

        close_a = merged["close_a"].values
        close_b = merged["close_b"].values
        current_ratio = float(close_a[-1] / close_b[-1])

        # --- Choose spread definition ---------------------------------------
        # Preferred: residual a - alpha - beta*b (matches cointegration filter).
        # Fallback: ratio a/b (pre-coint or if cached beta is unusable).
        use_residual = (
            coint_state is not None
            and coint_state.is_cointegrated
            and coint_state.beta is not None
            and coint_state.beta == coint_state.beta  # NaN check
            and coint_state.beta > 0
        )

        if use_residual:
            series = close_a - (coint_state.alpha + coint_state.beta * close_b)
            spread_mode = "residual"
        else:
            series = close_a / close_b
            spread_mode = "ratio"

        # Rolling statistics on the chosen series
        mean = float(np.mean(series))
        std = float(np.std(series))
        if std < 1e-10:
            logger.warning(
                f"{self.pair_name}: {spread_mode} std ~ 0, skipping"
            )
            return None

        spread_value = float(series[-1])
        zscore = (spread_value - mean) / std

        # Correlation (independent of mode)
        correlation = float(np.corrcoef(close_a, close_b)[0, 1])

        # ATR ratio kept for display / legacy
        atr_a = self._calc_atr(bars_a.tail(lookback + 50))
        atr_b = self._calc_atr(bars_b.tail(lookback + 50))
        hedge_ratio = atr_a / atr_b if atr_b > 0 else 1.0

        return SpreadState(
            pair_name=self.pair_name,
            leg_a=self.cfg.leg_a,
            leg_b=self.cfg.leg_b,
            ratio=current_ratio,
            mean=mean,
            std=std,
            zscore=zscore,
            correlation=correlation,
            hedge_ratio=hedge_ratio,
            spread_mode=spread_mode,
            spread_value=spread_value,
        )

    def _beta_sized_lots(
        self,
        coint_state: Optional[CointegrationState],
        cost_model: Optional[BrokerCostModel],
    ) -> tuple[float, float, str]:
        """Compute (leg_a_lot, leg_b_lot, note) for an entry.

        Cointegration regresses price_a ~ alpha + beta * price_b. For a
        market-neutral hedge the per-unit P&L of each leg must match:
            leg_a_lot * contract_a * beta == leg_b_lot * contract_b
        so leg_b_lot = leg_a_lot * contract_a * beta / contract_b.

        leg_a_lot is the user's configured size — we only rescale leg_b.
        Falls back to the configured pair when cointegration data or cost
        model entries are missing/invalid.
        """
        default_a = self.cfg.leg_a_lot
        default_b = self.cfg.leg_b_lot

        if coint_state is None or cost_model is None:
            return default_a, default_b, "configured lots (no coint state)"
        if not coint_state.is_cointegrated:
            return default_a, default_b, "configured lots (pair not cointegrated)"

        beta = coint_state.beta
        if beta is None or beta != beta or beta <= 0:  # NaN / non-positive
            return default_a, default_b, f"configured lots (beta={beta} unusable)"

        try:
            cost_a = cost_model.require(self.cfg.leg_a)
            cost_b = cost_model.require(self.cfg.leg_b)
        except KeyError:
            return default_a, default_b, "configured lots (symbol missing from cost model)"

        raw_b = default_a * cost_a.contract_size * beta / cost_b.contract_size

        # Round to lot_step and clamp to min_lot.
        step = cost_b.lot_step or 0.01
        rounded_b = max(cost_b.min_lot, round(raw_b / step) * step)
        rounded_b = round(rounded_b, 8)

        note = (
            f"beta={beta:.4f}, hedge {default_a:g}{self.cfg.leg_a}/"
            f"{rounded_b:g}{self.cfg.leg_b} "
            f"(configured was {default_b:g}, raw {raw_b:.4f})"
        )
        return default_a, rounded_b, note

    def generate_signal(
        self,
        state: SpreadState,
        has_open_position: bool,
        coint_state: Optional[CointegrationState] = None,
        cost_model: Optional[BrokerCostModel] = None,
    ) -> Optional[PairSignal]:
        """Generate trading signal from spread state."""

        z = state.zscore

        # --- Exit signals (if we have a position) ---
        if has_open_position:
            # Z-score stop: spread keeps widening (emergency exit, not P&L based)
            if abs(z) >= self.cfg.zscore_stop:
                return PairSignal(
                    pair_name=self.pair_name,
                    action="STOP",
                    zscore=z,
                    leg_a=self.cfg.leg_a,
                    leg_b=self.cfg.leg_b,
                    leg_a_side="",
                    leg_b_side="",
                    leg_a_lot=0,
                    leg_b_lot=0,
                    reason=f"Stop loss (Z={z:.2f} beyond {self.cfg.zscore_stop})",
                )

            return None  # hold

        # --- Entry signals (no position) ---
        if not self._is_in_session():
            return None  # outside trading session — show data but don't trade

        # Skip if spread already too wide — no room between here and stop_loss,
        # any entry here is catching a falling knife.
        if abs(z) >= self.cfg.zscore_entry_max:
            return None

        # Hedge ratio comes from the cointegration OLS beta when available,
        # not the static configured lots (which can drift out of dollar-
        # neutrality as prices move).
        leg_a_lot, leg_b_lot, sizing_note = self._beta_sized_lots(
            coint_state, cost_model
        )

        if z >= self.cfg.zscore_entry:
            # Ratio is high → leg_a overpriced relative to leg_b
            # SHORT leg_a, LONG leg_b
            return PairSignal(
                pair_name=self.pair_name,
                action="OPEN_SHORT_SPREAD",
                zscore=z,
                leg_a=self.cfg.leg_a,
                leg_b=self.cfg.leg_b,
                leg_a_side="SELL",
                leg_b_side="BUY",
                leg_a_lot=leg_a_lot,
                leg_b_lot=leg_b_lot,
                reason=f"Spread wide (Z={z:.2f}), short A / long B | {sizing_note}",
            )

        if z <= -self.cfg.zscore_entry:
            # Ratio is low → leg_a underpriced relative to leg_b
            # LONG leg_a, SHORT leg_b
            return PairSignal(
                pair_name=self.pair_name,
                action="OPEN_LONG_SPREAD",
                zscore=z,
                leg_a=self.cfg.leg_a,
                leg_b=self.cfg.leg_b,
                leg_a_side="BUY",
                leg_b_side="SELL",
                leg_a_lot=leg_a_lot,
                leg_b_lot=leg_b_lot,
                reason=f"Spread wide (Z={z:.2f}), long A / short B | {sizing_note}",
            )

        return None  # no signal

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR from OHLCV dataframe."""
        if len(df) < period + 1:
            return 0.0

        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )

        if len(tr) < period:
            return float(np.mean(tr))

        return float(np.mean(tr[-period:]))

"""Spread tracker — monitors price ratios and generates Z-score signals."""

from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import PairConfig
from src.mt5_client import MT5Client
from src.schemas import SpreadState, PairSignal


class SpreadTracker:
    """Tracks the spread (price ratio) between two assets and generates signals."""

    def __init__(self, pair_config: PairConfig, mt5: MT5Client):
        self.cfg = pair_config
        self.mt5 = mt5
        self.pair_name = f"{pair_config.leg_a}/{pair_config.leg_b}"

    def _is_in_session(self) -> bool:
        """Check if current time is within trading session."""
        if self.cfg.session_hours is None:
            return True  # 24/7

        now_utc = datetime.now(timezone.utc).hour
        for start, end in self.cfg.session_hours:
            if start <= now_utc < end:
                return True
        return False

    async def compute_spread(self, timeframe: str, lookback: int) -> Optional[SpreadState]:
        """Fetch prices for both legs and compute spread statistics."""
        if not self._is_in_session():
            return None

        # Fetch bars for both legs
        bars_a = await self.mt5.get_rates(self.cfg.leg_a, timeframe, lookback + 50)
        bars_b = await self.mt5.get_rates(self.cfg.leg_b, timeframe, lookback + 50)

        if bars_a is None or bars_b is None:
            logger.warning(f"Missing data for {self.pair_name}")
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

        # Price ratio (spread)
        ratio = close_a / close_b

        # Rolling statistics
        mean = np.mean(ratio)
        std = np.std(ratio)
        if std < 1e-10:
            logger.warning(f"{self.pair_name}: spread std ≈ 0, skipping")
            return None

        current_ratio = ratio[-1]
        zscore = (current_ratio - mean) / std

        # Correlation
        correlation = np.corrcoef(close_a, close_b)[0, 1]

        # ATR-based hedge ratio for lot sizing
        atr_a = self._calc_atr(bars_a.tail(lookback + 50))
        atr_b = self._calc_atr(bars_b.tail(lookback + 50))

        if atr_b > 0:
            hedge_ratio = atr_a / atr_b
        else:
            hedge_ratio = 1.0

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
        )

    def generate_signal(
        self, state: SpreadState, has_open_position: bool
    ) -> Optional[PairSignal]:
        """Generate trading signal from spread state."""

        z = state.zscore

        # --- Exit signals (if we have a position) ---
        if has_open_position:
            if abs(z) <= self.cfg.zscore_exit:
                return PairSignal(
                    pair_name=self.pair_name,
                    action="CLOSE",
                    zscore=z,
                    leg_a=self.cfg.leg_a,
                    leg_b=self.cfg.leg_b,
                    leg_a_side="",
                    leg_b_side="",
                    leg_a_lot=0,
                    leg_b_lot=0,
                    reason=f"Mean reverted (Z={z:.2f})",
                )

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
        # Spread is too wide → expect mean reversion
        if z >= self.cfg.zscore_entry:
            # Ratio is high → leg_a overpriced relative to leg_b
            # SHORT leg_a, LONG leg_b
            adjusted_b_lot = round(
                self.cfg.leg_b_lot * state.hedge_ratio, 2
            )
            return PairSignal(
                pair_name=self.pair_name,
                action="OPEN_SHORT_SPREAD",
                zscore=z,
                leg_a=self.cfg.leg_a,
                leg_b=self.cfg.leg_b,
                leg_a_side="SELL",
                leg_b_side="BUY",
                leg_a_lot=self.cfg.leg_a_lot,
                leg_b_lot=max(adjusted_b_lot, self.cfg.leg_b_lot),
                reason=f"Spread wide (Z={z:.2f}), short A / long B",
            )

        if z <= -self.cfg.zscore_entry:
            # Ratio is low → leg_a underpriced relative to leg_b
            # LONG leg_a, SHORT leg_b
            adjusted_b_lot = round(
                self.cfg.leg_b_lot * state.hedge_ratio, 2
            )
            return PairSignal(
                pair_name=self.pair_name,
                action="OPEN_LONG_SPREAD",
                zscore=z,
                leg_a=self.cfg.leg_a,
                leg_b=self.cfg.leg_b,
                leg_a_side="BUY",
                leg_b_side="SELL",
                leg_a_lot=self.cfg.leg_a_lot,
                leg_b_lot=max(adjusted_b_lot, self.cfg.leg_b_lot),
                reason=f"Spread wide (Z={z:.2f}), long A / short B",
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

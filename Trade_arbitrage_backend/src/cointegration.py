"""Cointegration filter — verifies pairs have a stationary (mean-reverting) spread.

Correlation ≠ cointegration. Two assets can move together (high corr) while
their spread drifts forever (not cointegrated). Pairs trading only works on
cointegrated pairs, so we gate entries on this test.

What we compute per pair:
  1. Engle-Granger test (statsmodels.coint) → p-value
  2. OLS regression price_a ~ alpha + beta * price_b → hedge ratio
  3. AR(1) on residual spread → half-life of mean reversion

A pair is tradeable iff:
  - p_value ≤ coint_p_threshold
  - half_life ∈ [coint_min_half_life, coint_max_half_life]

Results are cached per pair; re-checked every `coint_recheck_seconds`.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    from statsmodels.tsa.stattools import coint
    import statsmodels.api as sm
    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False

from config.settings import PairConfig, settings
from src.mt5_client import MT5Client
from src.schemas import CointegrationState


class CointegrationMonitor:
    """Runs periodic cointegration tests and caches per-pair tradability."""

    def __init__(self, pairs: list[PairConfig], mt5: MT5Client):
        self.pairs = pairs
        self.mt5 = mt5
        self._cache: Dict[str, CointegrationState] = {}
        self._lock = asyncio.Lock()

        if not _STATSMODELS_AVAILABLE:
            logger.error(
                "statsmodels not installed — cointegration filter disabled. "
                "Run: pip install statsmodels scipy"
            )

    # ------------------------------------------------------------------ API

    def get_state(self, pair_name: str) -> Optional[CointegrationState]:
        return self._cache.get(pair_name)

    def is_tradeable(self, pair_name: str) -> tuple[bool, str]:
        """Risk-manager hook: should we allow a new entry on this pair?"""
        if not settings.require_cointegration:
            return True, "coint filter disabled"

        state = self._cache.get(pair_name)
        if state is None:
            return False, "cointegration not yet checked"

        if not state.is_cointegrated:
            return False, state.reason or "not cointegrated"

        return True, "OK"

    async def check_all(self) -> None:
        """Run cointegration test for every configured pair."""
        if not _STATSMODELS_AVAILABLE:
            return
        if not self.mt5.is_connected():
            logger.debug("Coint: MT5 not connected, skipping this cycle")
            return

        for cfg in self.pairs:
            try:
                await self._check_pair(cfg)
            except Exception as e:
                logger.exception(f"Cointegration check failed for {cfg.leg_a}/{cfg.leg_b}: {e}")

    async def run_periodic(self, stop_flag) -> None:
        """Background task: re-check every coint_recheck_seconds.

        `stop_flag` is a callable returning True when the bot is shutting down.
        """
        if not _STATSMODELS_AVAILABLE:
            return

        # Initial check happens immediately so the filter is usable before the
        # first interval elapses.
        await self.check_all()

        while not stop_flag():
            try:
                await asyncio.sleep(settings.coint_recheck_seconds)
                if stop_flag():
                    break
                await self.check_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Cointegration periodic loop error: {e}")

    # ------------------------------------------------------------- internals

    async def _check_pair(self, cfg: PairConfig) -> None:
        pair_name = f"{cfg.leg_a}/{cfg.leg_b}"

        bars_a = await self.mt5.get_rates(
            cfg.leg_a, settings.coint_timeframe, settings.coint_lookback_bars + 50
        )
        bars_b = await self.mt5.get_rates(
            cfg.leg_b, settings.coint_timeframe, settings.coint_lookback_bars + 50
        )
        if bars_a is None or bars_b is None:
            logger.warning(f"Coint: missing bars for {pair_name} — skipping check")
            return

        merged = pd.merge(
            bars_a[["time", "close"]].rename(columns={"close": "a"}),
            bars_b[["time", "close"]].rename(columns={"close": "b"}),
            on="time",
            how="inner",
        )
        if len(merged) < settings.coint_lookback_bars:
            logger.warning(
                f"Coint {pair_name}: only {len(merged)} aligned bars, "
                f"need {settings.coint_lookback_bars} — skipping"
            )
            return

        merged = merged.tail(settings.coint_lookback_bars).reset_index(drop=True)
        a = merged["a"].values.astype(float)
        b = merged["b"].values.astype(float)

        async with self._lock:
            state = await asyncio.get_running_loop().run_in_executor(
                None, self._compute_state, pair_name, a, b
            )
            if state is not None:
                prev = self._cache.get(pair_name)
                self._cache[pair_name] = state
                self._log_change(prev, state)

    def _compute_state(
        self, pair_name: str, a: np.ndarray, b: np.ndarray
    ) -> Optional[CointegrationState]:
        """CPU work — runs in thread executor. No MT5/IO here."""
        try:
            # 1. Engle-Granger cointegration test.
            # Returns (t_stat, p_value, critical_values).
            _, p_value, _ = coint(a, b)

            # 2. OLS hedge ratio: a = alpha + beta * b
            X = sm.add_constant(b)
            ols = sm.OLS(a, X).fit()
            alpha = float(ols.params[0])
            beta = float(ols.params[1])

            # 3. Half-life via AR(1) on residuals.
            # spread_t = alpha + beta*b_t residual; mean-reverts as
            # Δspread_t = λ * spread_{t-1} + ε, half-life = -ln(2)/λ.
            residuals = a - (alpha + beta * b)
            half_life = self._half_life(residuals)

        except Exception as e:
            logger.error(f"Coint {pair_name}: computation error: {e}")
            return None

        # Decide tradability
        reasons = []
        if p_value > settings.coint_p_threshold:
            reasons.append(f"p={p_value:.3f} > {settings.coint_p_threshold}")
        if not np.isfinite(half_life):
            reasons.append("half-life undefined (spread not mean-reverting)")
        elif half_life < settings.coint_min_half_life:
            reasons.append(f"half-life {half_life:.1f} < {settings.coint_min_half_life} (noise)")
        elif half_life > settings.coint_max_half_life:
            reasons.append(
                f"half-life {half_life:.1f} > {settings.coint_max_half_life} (too slow)"
            )

        is_tradeable = len(reasons) == 0

        return CointegrationState(
            pair_name=pair_name,
            is_cointegrated=is_tradeable,
            p_value=float(p_value),
            beta=beta,
            alpha=alpha,
            half_life_bars=float(half_life) if np.isfinite(half_life) else float("inf"),
            bars_used=len(a),
            reason="; ".join(reasons) if reasons else "OK",
            last_check=datetime.now(timezone.utc),
        )

    @staticmethod
    def _half_life(residuals: np.ndarray) -> float:
        """AR(1) half-life of residuals. Returns inf if not mean-reverting."""
        if len(residuals) < 20:
            return float("inf")

        res = np.asarray(residuals, dtype=float)
        lagged = res[:-1]
        delta = np.diff(res)

        # OLS: Δres = λ * res_{t-1} + ε. If λ >= 0 the series doesn't revert.
        X = sm.add_constant(lagged)
        try:
            model = sm.OLS(delta, X).fit()
            lam = float(model.params[1])
        except Exception:
            return float("inf")

        if lam >= 0 or not np.isfinite(lam):
            return float("inf")

        return float(-np.log(2) / lam)

    def _log_change(
        self, prev: Optional[CointegrationState], curr: CointegrationState
    ) -> None:
        if prev is None:
            logger.info(
                f"[COINT] {curr.pair_name}: "
                f"{'TRADEABLE' if curr.is_cointegrated else 'BLOCKED'} "
                f"| p={curr.p_value:.3f} | β={curr.beta:.4f} | "
                f"half-life={curr.half_life_bars:.1f} bars | {curr.reason}"
            )
            return

        if prev.is_cointegrated != curr.is_cointegrated:
            verb = "REGAINED" if curr.is_cointegrated else "LOST"
            logger.warning(
                f"[COINT] {curr.pair_name} {verb} cointegration "
                f"| p={curr.p_value:.3f} | half-life={curr.half_life_bars:.1f} | {curr.reason}"
            )
        else:
            logger.debug(
                f"[COINT] {curr.pair_name}: unchanged "
                f"| p={curr.p_value:.3f} | β={curr.beta:.4f} | "
                f"half-life={curr.half_life_bars:.1f}"
            )

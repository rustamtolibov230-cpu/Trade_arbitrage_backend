"""Historical bar loader with on-disk cache.

MT5's `copy_rates_from_pos` is capped at ~1000 bars in a single call and
occasionally flakes. We batch many calls, dedupe, and cache the result as
parquet/CSV so repeated backtests don't re-download.

Cache layout:
    data/backtest_cache/
        <SYMBOL>_<TIMEFRAME>.parquet
        <SYMBOL>_<TIMEFRAME>.meta.json    # min/max time, bar count, saved_at

If the cache covers the requested window and is fresher than `max_age_hours`,
it's used directly. Otherwise we fetch the missing range from MT5 and update
the cache.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from src.mt5_client import MT5Client


CACHE_DIR = Path("data") / "backtest_cache"

# MT5 returns at most ~1000 bars per copy_rates_from_pos call on most servers.
# Use conservative batch to avoid partial/empty responses.
_BATCH_SIZE = 1000


class HistoricalDataLoader:
    """Async loader with caching."""

    def __init__(self, mt5: MT5Client, cache_dir: Path = CACHE_DIR):
        self.mt5 = mt5
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------- public

    async def load(
        self,
        symbol: str,
        timeframe: str,
        bars: int,
        use_cache: bool = True,
        max_cache_age_hours: float = 6.0,
    ) -> Optional[pd.DataFrame]:
        """Load the last `bars` bars of `symbol` at `timeframe`.

        Returns a DataFrame with columns: time, open, high, low, close, tick_volume.
        Time is tz-aware UTC.
        """
        cache_path = self._cache_path(symbol, timeframe)
        meta_path = self._meta_path(symbol, timeframe)

        if use_cache and cache_path.exists() and meta_path.exists():
            cached = self._read_cache(cache_path, meta_path)
            if cached is not None and len(cached) >= bars:
                age_h = (datetime.now(timezone.utc) - cached["saved_at"]).total_seconds() / 3600
                if age_h <= max_cache_age_hours:
                    df = cached["df"].tail(bars).reset_index(drop=True)
                    logger.info(
                        f"[DATA] {symbol} {timeframe}: cache hit "
                        f"({len(df)} bars, age {age_h:.1f}h)"
                    )
                    return df
                logger.info(f"[DATA] {symbol} {timeframe}: cache stale ({age_h:.1f}h), refetching")

        df = await self._fetch_from_mt5(symbol, timeframe, bars)
        if df is None or df.empty:
            return None

        self._write_cache(cache_path, meta_path, df)
        logger.info(f"[DATA] {symbol} {timeframe}: fetched {len(df)} bars, cached")
        return df.tail(bars).reset_index(drop=True)

    async def load_pair(
        self,
        leg_a: str,
        leg_b: str,
        timeframe: str,
        bars: int,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """Load both legs, inner-join on time. Returns aligned DataFrame."""
        df_a, df_b = await asyncio.gather(
            self.load(leg_a, timeframe, bars, **kwargs),
            self.load(leg_b, timeframe, bars, **kwargs),
        )
        if df_a is None or df_b is None:
            return None

        merged = pd.merge(
            df_a[["time", "open", "high", "low", "close"]].rename(
                columns={"open": "a_open", "high": "a_high", "low": "a_low", "close": "a_close"}
            ),
            df_b[["time", "open", "high", "low", "close"]].rename(
                columns={"open": "b_open", "high": "b_high", "low": "b_low", "close": "b_close"}
            ),
            on="time",
            how="inner",
        ).reset_index(drop=True)

        logger.info(
            f"[DATA] {leg_a}/{leg_b} {timeframe}: aligned {len(merged)} bars "
            f"(from {len(df_a)}/{len(df_b)})"
        )
        return merged

    # -------------------------------------------------------------- fetching

    async def _fetch_from_mt5(
        self, symbol: str, timeframe: str, bars: int
    ) -> Optional[pd.DataFrame]:
        """Pull bars in batches and concatenate.

        Strategy: we repeatedly call get_rates with increasing `count` and keep
        the freshest copy. For very large bar counts a single call may return
        fewer than requested — that's acceptable, we log and return what we got.
        """
        # Ask for a bit more than requested to account for aligned-merge loss
        request_count = min(bars + 200, 100_000)  # sanity cap
        df = await self.mt5.get_rates(symbol, timeframe, request_count)
        if df is None or len(df) == 0:
            logger.error(f"[DATA] {symbol} {timeframe}: no bars returned from MT5")
            return None

        # Normalize time to tz-aware UTC
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        else:
            df["time"] = df["time"].dt.tz_localize("UTC") if df["time"].dt.tz is None else df["time"]

        df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

        if len(df) < bars:
            logger.warning(
                f"[DATA] {symbol} {timeframe}: got {len(df)} bars, "
                f"requested {bars} — broker may have shorter history"
            )
        return df

    # ----------------------------------------------------------------- cache

    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        safe_symbol = symbol.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_symbol}_{timeframe}.parquet"

    def _meta_path(self, symbol: str, timeframe: str) -> Path:
        safe_symbol = symbol.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_symbol}_{timeframe}.meta.json"

    def _read_cache(
        self, cache_path: Path, meta_path: Path
    ) -> Optional[Dict]:
        try:
            df = pd.read_parquet(cache_path)
            meta = json.loads(meta_path.read_text())
            saved_at = datetime.fromisoformat(meta["saved_at"])
            return {"df": df, "saved_at": saved_at, "meta": meta}
        except Exception as e:
            logger.warning(f"[DATA] cache read failed for {cache_path.name}: {e}")
            return None

    def _write_cache(
        self, cache_path: Path, meta_path: Path, df: pd.DataFrame
    ) -> None:
        try:
            df.to_parquet(cache_path, index=False)
            meta = {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "bars": len(df),
                "first_time": df["time"].iloc[0].isoformat(),
                "last_time": df["time"].iloc[-1].isoformat(),
            }
            meta_path.write_text(json.dumps(meta, indent=2))
        except Exception as e:
            logger.error(f"[DATA] cache write failed: {e}")

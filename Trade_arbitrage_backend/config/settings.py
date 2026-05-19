"""Configuration for pairs trading / arbitrage bot."""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Dict, List, Tuple


class PairConfig:
    """Configuration for a single trading pair."""

    def __init__(
        self,
        leg_a: str,
        leg_b: str,
        leg_a_lot: float,
        leg_b_lot: float,
        zscore_entry: float = 2.0,
        zscore_entry_max: float = 2.0,
        zscore_exit: float = 0.5,
        zscore_stop: float = 3.5,
        lookback: int = 100,
        session_hours: List[Tuple[int, int]] | None = None,
        weekdays_only: bool = False,
        max_hold_minutes: int = 60,
        profit_target: float = 1.0,  # close when combined P&L >= this ($)
        min_hold_seconds: int = 20,  # block profit-target close until this many seconds elapsed
    ):
        self.leg_a = leg_a  # e.g. "BTCUSD"
        self.leg_b = leg_b  # e.g. "ETHUSD"
        self.leg_a_lot = leg_a_lot
        self.leg_b_lot = leg_b_lot
        self.zscore_entry = zscore_entry
        # Don't enter if |Z| is already this far — too close to stop, no room to work
        self.zscore_entry_max = zscore_entry_max
        self.zscore_exit = zscore_exit
        self.zscore_stop = zscore_stop
        self.lookback = lookback  # bars for spread mean/std
        self.session_hours = session_hours  # None = any hour
        self.weekdays_only = weekdays_only  # True = skip Sat/Sun UTC (24/5)
        self.max_hold_minutes = max_hold_minutes
        self.profit_target = profit_target  # $ profit to take
        self.min_hold_seconds = min_hold_seconds


class Settings(BaseSettings):
    """Global settings."""

    # MT5
    mt5_login: int = 0
    mt5_password: str = ""
    mt5_server: str = ""
    mt5_path: str = r"C:\Program Files\MetaTrader 5\terminal64.exe"

    # Risk
    max_daily_loss: float = 50.0
    max_open_pairs: int = 1
    min_correlation: float = 0.80  # don't trade if correlation drops below

    # Profit-target sanity gate — bot refuses to start if any pair has
    # profit_target < safety_factor × round-trip cost (spread + commission).
    # Trading below 1.0× is mathematically a guaranteed loss; 1.5× gives
    # headroom for live slippage and wider-than-typical spreads.
    profit_target_safety_factor: float = 1.5

    # Scanning
    scan_interval_seconds: int = 3  # how often to check for signals
    timeframe: str = "M1"  # M1 for faster signals (was M5)
    spread_lookback: int = 50  # bars for Z-score (was 100)

    # After a pair closes, wait this long before reopening it. Prevents the
    # "stop out → immediate reopen at same wide Z → stop out again" loop that
    # bleeds spread on every round trip.
    cooldown_seconds: int = 300

    # --- Cointegration filter ---
    # Correlation only tells us assets move together short-term. Cointegration
    # tells us their spread is stationary (mean-reverting) — which is the
    # actual prerequisite for pairs trading to work.
    require_cointegration: bool = True      # set False to bypass (e.g. while tuning)
    coint_p_threshold: float = 0.05         # Engle-Granger p-value ceiling
    coint_lookback_bars: int = 500          # bars for the test (longer = more reliable)
    coint_timeframe: str = "M15"            # timeframe for the coint test (NOT the scan TF)
    coint_recheck_seconds: int = 1800       # re-run the test every 30 min
    coint_min_half_life: float = 2.0        # reject if half-life < 2 bars (noise)
    coint_max_half_life: float = 100.0      # reject if half-life > 100 bars (too slow)

    # Pairs — configured in code, not env
    model_config = {"env_file": ".env", "extra": "ignore"}


# --- Pair definitions ---

# profit_target must exceed round-trip cost (spread + commission, both legs).
# Per src/backtest/cost_model.py IC_MARKETS_RAW at the lots below:
#   BTC/ETH  0.04 / 0.80  → cost ≈ $7.16  (ETH commission alone is $5.60)
#   XAU/XAG  0.01 / 0.01  → cost ≈ $1.26  (XAG spread alone is $1.00)
# Targets set to 2× cost for slippage headroom. If you change lots, recompute
# with round_trip_cost_for_pair() and update — task #2 will enforce this at startup.

PAIRS = [
    PairConfig(
        leg_a="BTCUSD",
        leg_b="ETHUSD",
        leg_a_lot=0.04,
        leg_b_lot=0.8,
        zscore_entry=1.0,       # enter when spread is 1 std dev wide
        zscore_entry_max=1.8,   # skip entry if Z already too close to stop
        zscore_exit=0.2,        # fallback: close if Z reverts but profit not hit
        zscore_stop=2.5,        # emergency stop if spread keeps widening
        lookback=50,
        session_hours=None,     # 24/7
        max_hold_minutes=30,
        profit_target=14.0,     # 2× round-trip cost ($7.16) — see header comment
    ),
    PairConfig(
        leg_a="XAUUSD",
        leg_b="XAGUSD",
        leg_a_lot=0.01,
        leg_b_lot=0.01,
        zscore_entry=1.0,
        zscore_entry_max=1.8,
        zscore_exit=0.2,        # fallback Z-score exit
        zscore_stop=2.5,        # emergency stop
        lookback=50,
        session_hours=None,
        weekdays_only=True,  # 24/5 — metals closed on weekends
        max_hold_minutes=30,
        profit_target=2.5,      # 2× round-trip cost ($1.26) — see header comment
    ),
]


settings = Settings()

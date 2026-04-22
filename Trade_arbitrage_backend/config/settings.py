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
        max_hold_minutes: int = 60,
        profit_target: float = 1.0,  # close when combined P&L >= this ($)
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
        self.session_hours = session_hours  # None = 24/7
        self.max_hold_minutes = max_hold_minutes
        self.profit_target = profit_target  # $ profit to take


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

    # Scanning
    scan_interval_seconds: int = 3  # how often to check for signals
    timeframe: str = "M1"  # M1 for faster signals (was M5)
    spread_lookback: int = 50  # bars for Z-score (was 100)

    # After a pair closes, wait this long before reopening it. Prevents the
    # "stop out → immediate reopen at same wide Z → stop out again" loop that
    # bleeds spread on every round trip.
    cooldown_seconds: int = 300

    # Pairs — configured in code, not env
    model_config = {"env_file": ".env", "extra": "ignore"}


# --- Pair definitions ---

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
        profit_target=1.0,      # primary exit: close when combined P&L >= +$1
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
        session_hours=[(7, 17)],  # London + NY
        max_hold_minutes=30,
        profit_target=1.0,      # primary exit: close when combined P&L >= +$1
    ),
]


settings = Settings()

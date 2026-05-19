"""Backtest framework — replay historical bars through the pairs strategy.

Run a backtest:
    python backtest.py --pair BTCUSD/ETHUSD --days 60 --save reports/
"""

from src.backtest.cost_model import BrokerCostModel, SymbolCost, IC_MARKETS_RAW
from src.backtest.data_loader import HistoricalDataLoader
from src.backtest.simulator import Simulator, SimSettings, SimTrade, SimResult
from src.backtest.metrics import compute_metrics

__all__ = [
    "BrokerCostModel",
    "SymbolCost",
    "IC_MARKETS_RAW",
    "HistoricalDataLoader",
    "Simulator",
    "SimSettings",
    "SimTrade",
    "SimResult",
    "compute_metrics",
]

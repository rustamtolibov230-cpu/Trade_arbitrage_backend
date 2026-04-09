"""MetaTrader 5 client wrapper for pairs trading."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings

_executor = ThreadPoolExecutor(max_workers=2)


async def _run_in_executor(func, *args):
    """Run blocking MT5 call in thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, func, *args)


class MT5Client:
    """Async wrapper around MetaTrader5 Python API."""

    def __init__(self):
        self._connected = False

    async def connect(self) -> bool:
        """Initialize MT5 connection."""
        if self._connected:
            return True

        def _init():
            if not mt5.initialize(
                path=settings.mt5_path,
                login=settings.mt5_login,
                password=settings.mt5_password,
                server=settings.mt5_server,
            ):
                logger.error(f"MT5 init failed: {mt5.last_error()}")
                return False
            info = mt5.account_info()
            if info:
                logger.info(
                    f"MT5 connected: {info.login} | Balance: ${info.balance:.2f}"
                )
            return True

        self._connected = await _run_in_executor(_init)
        return self._connected

    async def disconnect(self):
        """Shutdown MT5."""
        await _run_in_executor(mt5.shutdown)
        self._connected = False
        logger.info("MT5 disconnected")

    async def get_rates(
        self, symbol: str, timeframe_str: str, count: int
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV bars."""
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
        }
        tf = tf_map.get(timeframe_str, mt5.TIMEFRAME_M5)

        def _fetch():
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
            if rates is None or len(rates) == 0:
                logger.warning(f"No rates for {symbol}")
                return None
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            return df

        return await _run_in_executor(_fetch)

    async def get_tick(self, symbol: str) -> Optional[dict]:
        """Get latest tick (bid/ask)."""

        def _fetch():
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            return {"bid": tick.bid, "ask": tick.ask, "time": tick.time}

        return await _run_in_executor(_fetch)

    async def open_order(
        self,
        symbol: str,
        order_type: str,  # "BUY" or "SELL"
        lot: float,
        comment: str = "",
    ) -> Optional[int]:
        """Open a market order. Returns ticket or None."""

        def _send():
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"No tick for {symbol}")
                return None

            info = mt5.symbol_info(symbol)
            if info is None:
                logger.error(f"No symbol info for {symbol}")
                return None

            price = tick.ask if order_type == "BUY" else tick.bid
            mt5_type = (
                mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL
            )

            # Auto-detect filling mode from symbol info
            filling = mt5.ORDER_FILLING_FOK  # default
            if info.filling_mode & 2:  # supports IOC
                filling = mt5.ORDER_FILLING_IOC
            elif info.filling_mode & 1:  # supports FOK
                filling = mt5.ORDER_FILLING_FOK
            else:  # RETURN
                filling = mt5.ORDER_FILLING_RETURN

            # Round lot to symbol's volume step
            vol_step = info.volume_step
            lot_rounded = round(round(lot / vol_step) * vol_step, 8)
            lot_rounded = max(lot_rounded, info.volume_min)

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_rounded,
                "type": mt5_type,
                "price": price,
                "deviation": 20,
                "magic": 777777,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling,
            }

            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                err = result.comment if result else "None"
                logger.error(f"Order failed {symbol} {order_type}: {err}")
                return None

            logger.info(
                f"Opened {order_type} {lot} {symbol} @ {price} | ticket={result.order}"
            )
            return result.order

        return await _run_in_executor(_send)

    async def close_order(self, ticket: int, symbol: str, lot: float, order_type: str) -> bool:
        """Close a position by ticket."""

        def _close():
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return False

            info = mt5.symbol_info(symbol)
            if info is None:
                return False

            # To close: opposite order type
            close_type = (
                mt5.ORDER_TYPE_SELL if order_type == "BUY" else mt5.ORDER_TYPE_BUY
            )
            price = tick.bid if order_type == "BUY" else tick.ask

            # Auto-detect filling mode
            filling = mt5.ORDER_FILLING_FOK
            if info.filling_mode & 2:
                filling = mt5.ORDER_FILLING_IOC
            elif info.filling_mode & 1:
                filling = mt5.ORDER_FILLING_FOK
            else:
                filling = mt5.ORDER_FILLING_RETURN

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot,
                "type": close_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": 777777,
                "comment": "arb_close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling,
            }

            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                err = result.comment if result else "None"
                logger.error(f"Close failed ticket={ticket}: {err}")
                return False

            logger.info(f"Closed ticket={ticket} {symbol} @ {price}")
            return True

        return await _run_in_executor(_close)

    async def get_position_profit(self, ticket: int) -> Optional[float]:
        """Get current P&L for a position."""

        def _get():
            positions = mt5.positions_get(ticket=ticket)
            if positions is None or len(positions) == 0:
                return None
            return positions[0].profit

        return await _run_in_executor(_get)

    async def get_account_info(self) -> Optional[dict]:
        """Get account balance/equity."""

        def _get():
            info = mt5.account_info()
            if info is None:
                return None
            return {
                "balance": info.balance,
                "equity": info.equity,
                "profit": info.profit,
            }

        return await _run_in_executor(_get)

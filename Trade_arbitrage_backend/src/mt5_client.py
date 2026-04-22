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
        self._last_error: str = ""
        self._login: int = 0
        self._server: str = ""

    async def connect(
        self,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        path: str | None = None,
    ) -> bool:
        """Initialize MT5 connection.

        If credentials are passed, use them (runtime login from dashboard).
        Otherwise fall back to values from .env / settings.
        Calling this while already connected will shut down and re-initialize,
        so the user can switch accounts without restarting the server.
        """
        use_login = login if login is not None else settings.mt5_login
        use_password = password if password is not None else settings.mt5_password
        use_server = server if server is not None else settings.mt5_server
        use_path = path if path is not None else settings.mt5_path

        def _init():
            # If already initialized, shut down first so re-login picks up new creds
            try:
                mt5.shutdown()
            except Exception:
                pass

            if not mt5.initialize(
                path=use_path,
                login=int(use_login) if use_login else 0,
                password=use_password,
                server=use_server,
            ):
                err = mt5.last_error()
                logger.error(f"MT5 init failed: {err}")
                self._last_error = f"{err}"
                return False

            info = mt5.account_info()
            if info is None:
                self._last_error = "account_info returned None after init"
                logger.error(self._last_error)
                return False

            self._login = info.login
            self._server = info.server if hasattr(info, "server") else use_server
            self._last_error = ""
            logger.info(
                f"MT5 connected: {info.login} @ {self._server} | Balance: ${info.balance:.2f}"
            )
            return True

        self._connected = await _run_in_executor(_init)
        return self._connected

    def is_connected(self) -> bool:
        return self._connected

    def status(self) -> dict:
        return {
            "connected": self._connected,
            "login": self._login if self._connected else 0,
            "server": self._server if self._connected else "",
            "last_error": self._last_error,
        }

    async def disconnect(self):
        """Shutdown MT5."""
        await _run_in_executor(mt5.shutdown)
        self._connected = False
        self._login = 0
        self._server = ""
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
            # Ensure symbol is visible in Market Watch (silent no-op if already selected)
            if not mt5.symbol_select(symbol, True):
                logger.error(f"symbol_select failed for {symbol}: {mt5.last_error()}")
                return None

            info = mt5.symbol_info(symbol)
            if info is None:
                logger.error(f"No symbol info for {symbol}")
                return None

            if not info.visible:
                logger.error(f"{symbol} not visible in Market Watch after select")
                return None

            if info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
                logger.error(f"{symbol} trading disabled by broker")
                return None

            tick = mt5.symbol_info_tick(symbol)
            if tick is None or (tick.bid == 0 and tick.ask == 0):
                logger.error(f"No tick for {symbol} (bid={getattr(tick,'bid',None)}, ask={getattr(tick,'ask',None)})")
                return None

            price = tick.ask if order_type == "BUY" else tick.bid
            mt5_type = (
                mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL
            )

            # Round lot to symbol's volume step, clamp to broker min/max
            vol_step = info.volume_step
            lot_rounded = round(round(lot / vol_step) * vol_step, 8)
            lot_rounded = max(lot_rounded, info.volume_min)
            lot_rounded = min(lot_rounded, info.volume_max)
            if lot_rounded != lot:
                logger.debug(f"Lot adjusted {symbol}: {lot} → {lot_rounded} (step={vol_step}, min={info.volume_min}, max={info.volume_max})")

            # Build filling-mode fallback list based on what the symbol advertises.
            # Some brokers report the supported mask incorrectly, so we try all
            # advertised modes plus RETURN as a last resort.
            filling_candidates = []
            if info.filling_mode & 2:
                filling_candidates.append(mt5.ORDER_FILLING_IOC)
            if info.filling_mode & 1:
                filling_candidates.append(mt5.ORDER_FILLING_FOK)
            for mode in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN):
                if mode not in filling_candidates:
                    filling_candidates.append(mode)

            last_err = ""
            last_retcode = 0
            for filling in filling_candidates:
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
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(
                        f"Opened {order_type} {lot_rounded} {symbol} @ {price} | ticket={result.order} | filling={filling}"
                    )
                    return result.order

                last_retcode = result.retcode if result else 0
                last_err = result.comment if result else str(mt5.last_error())
                # Only retry a different filling mode for invalid-fill errors
                if last_retcode != 10030:  # TRADE_RETCODE_INVALID_FILL
                    break

            logger.error(
                f"Order failed {symbol} {order_type} {lot_rounded}lot: "
                f"retcode={last_retcode} comment='{last_err}' "
                f"supported_filling={info.filling_mode} min={info.volume_min} step={info.volume_step}"
            )
            return None

        return await _run_in_executor(_send)

    async def close_order(self, ticket: int, symbol: str, lot: float, order_type: str) -> bool:
        """Close a position by ticket. Uses actual MT5 position volume to avoid lot mismatch."""

        def _close():
            mt5.symbol_select(symbol, True)

            info = mt5.symbol_info(symbol)
            if info is None:
                return False

            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return False

            # Always use the actual position volume from MT5, not the stored lot.
            # Brokers round lots during order execution — using the stored lot can
            # cause "invalid volume" or partial-close errors.
            positions = mt5.positions_get(ticket=ticket)
            if positions and len(positions) > 0:
                actual_lot = positions[0].volume
            else:
                # Position may already be closed
                logger.warning(f"Position ticket={ticket} not found, may already be closed")
                return True

            # To close: opposite order type
            close_type = (
                mt5.ORDER_TYPE_SELL if order_type == "BUY" else mt5.ORDER_TYPE_BUY
            )
            price = tick.bid if order_type == "BUY" else tick.ask

            filling_candidates = []
            if info.filling_mode & 2:
                filling_candidates.append(mt5.ORDER_FILLING_IOC)
            if info.filling_mode & 1:
                filling_candidates.append(mt5.ORDER_FILLING_FOK)
            for mode in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN):
                if mode not in filling_candidates:
                    filling_candidates.append(mode)

            last_err = ""
            last_retcode = 0
            for filling in filling_candidates:
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": actual_lot,
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
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"Closed ticket={ticket} {symbol} {actual_lot}lot @ {price} | filling={filling}")
                    return True

                last_retcode = result.retcode if result else 0
                last_err = result.comment if result else str(mt5.last_error())
                if last_retcode != 10030:  # TRADE_RETCODE_INVALID_FILL
                    break

            logger.error(f"Close failed ticket={ticket} {symbol}: retcode={last_retcode} comment='{last_err}'")
            return False

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

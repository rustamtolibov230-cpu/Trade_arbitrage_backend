"""
FastAPI server — runs the arbitrage bot + serves dashboard via WebSocket.
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from config.settings import settings, PAIRS
from src.mt5_client import MT5Client
from src.spread_tracker import SpreadTracker
from src.execution import ExecutionEngine
from src.risk_manager import RiskManager

# --- Logging ---
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
    level="INFO",
)
logger.add(
    "logs/arb_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG",
)

# --- App ---
app = FastAPI(title="Trade Arbitrage")

# Serve frontend static files — works from both dev and PyInstaller exe
if getattr(sys, 'frozen', False):
    _base = Path(sys._MEIPASS)
else:
    _base = Path(__file__).parent

FRONTEND_DIR = _base / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# --- Bot components ---
mt5_client = MT5Client()
execution = ExecutionEngine(mt5_client)
risk_mgr = RiskManager(execution)
trackers = [SpreadTracker(cfg, mt5_client) for cfg in PAIRS]

# --- WebSocket hub ---
ws_clients: Set[WebSocket] = set()


async def broadcast(data: dict):
    """Send JSON to all connected WebSocket clients."""
    global ws_clients
    if not ws_clients:
        return
    msg = json.dumps(data, default=str)
    dead = set()
    for ws in ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    ws_clients -= dead


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    logger.info(f"Dashboard connected ({len(ws_clients)} clients)")
    try:
        while True:
            # Keep alive — listen for client messages (close, commands)
            data = await ws.receive_text()
            if data == "close_all":
                logger.info("Dashboard requested close all")
                await execution.close_all("dashboard")
    except WebSocketDisconnect:
        ws_clients.discard(ws)
        logger.info(f"Dashboard disconnected ({len(ws_clients)} clients)")


# --- REST endpoints ---
@app.get("/")
async def dashboard():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return HTMLResponse("<h1>Trade Arbitrage</h1><p>Frontend not found</p>")


@app.get("/api/status")
async def api_status():
    account = await mt5_client.get_account_info()
    stats = execution.get_stats()
    active = []
    for name, pair in execution.active_pairs.items():
        pnl = await execution.get_pair_pnl(name)
        hold_min = (datetime.now(timezone.utc) - pair.entry_time).total_seconds() / 60
        active.append({
            "pair": name,
            "leg_a_side": pair.leg_a_side,
            "leg_b_side": pair.leg_b_side,
            "entry_zscore": round(pair.entry_zscore, 2),
            "pnl": round(pnl, 2) if pnl is not None else 0,
            "hold_min": round(hold_min, 1),
            "max_profit": round(pair.max_profit, 2),
            "min_profit": round(pair.min_profit, 2),
        })
    return {
        "account": account,
        "stats": stats,
        "active_pairs": active,
        "daily_pnl": round(execution.get_daily_pnl(), 2),
        "bot_running": _bot_running,
    }


@app.get("/api/history")
async def api_history():
    return [
        {
            "pair": t.pair_name,
            "profit": round(t.profit, 2),
            "hold_min": round(t.hold_minutes, 1),
            "entry_z": round(t.entry_zscore, 2),
            "exit_z": round(t.exit_zscore, 2),
            "reason": t.reason,
            "time": t.exit_time.strftime("%H:%M:%S"),
        }
        for t in reversed(execution.trade_history[-50:])
    ]


@app.post("/api/close-all")
async def api_close_all():
    results = await execution.close_all("api_request")
    return {"closed": len(results)}


# --- Bot loop (runs as background task) ---
_bot_running = False
_bot_task = None


@app.post("/api/start")
async def api_start():
    """Start the bot loop."""
    global _bot_task, _bot_running
    if _bot_running:
        return {"status": "already_running"}
    _bot_task = asyncio.create_task(bot_loop())
    await broadcast({"type": "bot_state", "running": True})
    logger.info("Bot STARTED by dashboard")
    return {"status": "started"}


@app.post("/api/stop")
async def api_stop():
    """Stop the bot loop and close all pairs."""
    global _bot_running
    if not _bot_running:
        return {"status": "already_stopped"}
    _bot_running = False
    if execution.active_pairs:
        await execution.close_all("stopped_by_user")
    await broadcast({"type": "bot_state", "running": False})
    logger.info("Bot STOPPED by dashboard")
    return {"status": "stopped"}


@app.get("/api/bot-state")
async def api_bot_state():
    return {"running": _bot_running}


async def bot_loop():
    """Main arbitrage scanning loop."""
    global _bot_running
    _bot_running = True
    cycle = 0

    logger.info("Bot loop started")

    while _bot_running:
        try:
            cycle += 1
            spreads_data = []

            for tracker in trackers:
                state = await tracker.compute_spread(
                    timeframe=settings.timeframe,
                    lookback=settings.spread_lookback,
                )
                if state is None:
                    continue

                is_open = execution.has_open_pair(state.pair_name)
                pnl = await execution.get_pair_pnl(state.pair_name) if is_open else None

                # Collect for broadcast
                spreads_data.append({
                    "pair": state.pair_name,
                    "zscore": round(state.zscore, 3),
                    "correlation": round(state.correlation, 4),
                    "ratio": round(state.ratio, 6),
                    "mean": round(state.mean, 6),
                    "std": round(state.std, 6),
                    "is_open": is_open,
                    "pnl": round(pnl, 2) if pnl is not None else None,
                })

                # Generate signal
                signal_obj = tracker.generate_signal(state, is_open)
                if signal_obj is None:
                    continue

                # Risk check for entries
                if signal_obj.action.startswith("OPEN"):
                    ok, reason = risk_mgr.can_open_pair(state)
                    if not ok:
                        logger.warning(f"Risk blocked: {reason}")
                        await broadcast({
                            "type": "risk_block",
                            "pair": state.pair_name,
                            "reason": reason,
                        })
                        continue

                # Execute
                logger.info(f">>> {signal_obj.action} | {signal_obj.reason}")
                success = await execution.execute_signal(signal_obj)

                await broadcast({
                    "type": "signal",
                    "pair": signal_obj.pair_name,
                    "action": signal_obj.action,
                    "zscore": round(signal_obj.zscore, 2),
                    "reason": signal_obj.reason,
                    "executed": success,
                })

            # Check timeouts
            await _check_timeouts()

            # Risk monitoring
            await risk_mgr.check_active_pairs()

            # Daily loss circuit breaker
            daily_pnl = execution.get_daily_pnl()
            if daily_pnl <= -settings.max_daily_loss:
                logger.error(f"DAILY LOSS LIMIT: ${daily_pnl:.2f}")
                await execution.close_all("daily_loss_limit")
                await broadcast({"type": "circuit_breaker", "daily_pnl": daily_pnl})

            # Broadcast state update
            stats = execution.get_stats()
            await broadcast({
                "type": "update",
                "spreads": spreads_data,
                "stats": stats,
                "daily_pnl": round(daily_pnl, 2),
                "active_count": len(execution.active_pairs),
                "cycle": cycle,
            })

        except Exception as e:
            logger.exception(f"Bot loop error: {e}")

        await asyncio.sleep(settings.scan_interval_seconds)


async def _check_timeouts():
    """Close pairs that exceeded max hold time."""
    now = datetime.now(timezone.utc)
    timeout_map = {f"{c.leg_a}/{c.leg_b}": c.max_hold_minutes for c in PAIRS}

    for pair_name, pair in list(execution.active_pairs.items()):
        hold_min = (now - pair.entry_time).total_seconds() / 60
        max_hold = timeout_map.get(pair_name, 60)
        if hold_min >= max_hold:
            from src.schemas import PairSignal
            sig = PairSignal(
                pair_name=pair_name, action="CLOSE", zscore=0,
                leg_a=pair.leg_a, leg_b=pair.leg_b,
                leg_a_side="", leg_b_side="",
                leg_a_lot=0, leg_b_lot=0,
                reason=f"Timeout ({hold_min:.0f}m > {max_hold}m)",
            )
            await execution.execute_signal(sig)


@app.on_event("startup")
async def on_startup():
    """Connect MT5 — bot waits for Start button."""
    connected = await mt5_client.connect()
    if not connected:
        logger.error("MT5 connection failed!")
        return

    account = await mt5_client.get_account_info()
    if account:
        risk_mgr.set_start_balance(account["balance"])

    logger.info("Server ready on http://localhost:8050 — waiting for Start")


@app.on_event("shutdown")
async def on_shutdown():
    global _bot_running
    _bot_running = False
    if execution.active_pairs:
        await execution.close_all("shutdown")
    await mt5_client.disconnect()

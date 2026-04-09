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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from config.settings import settings, PAIRS
from src.mt5_client import MT5Client
from src.schemas import PairSignal
from src.spread_tracker import SpreadTracker
from src.execution import ExecutionEngine
from src.risk_manager import RiskManager

# --- Logging ---
os.makedirs("logs", exist_ok=True)
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

# CORS — allows standalone frontend to connect from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optionally serve frontend if it exists next to the server
if getattr(sys, 'frozen', False):
    _exe_dir = Path(os.path.dirname(os.path.abspath(sys.executable)))
    FRONTEND_DIR = _exe_dir / "frontend"
else:
    FRONTEND_DIR = Path(__file__).parent / "frontend"

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
    """Serve frontend if it exists next to server, otherwise show API info."""
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        from fastapi.responses import Response
        content = index.read_bytes()
        return Response(
            content=content,
            media_type="text/html",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return HTMLResponse(
        "<h1>Trade Arbitrage API</h1>"
        "<p>Backend running. Open <code>frontend/index.html</code> in your browser.</p>"
        "<p>API: /api/status, /api/spreads, /api/history, /api/start, /api/stop</p>"
    )


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
    _bot_running = True  # set BEFORE creating task to prevent double-start race
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

            for tracker in trackers:
                state = await tracker.compute_spread(
                    timeframe=settings.timeframe,
                    lookback=settings.spread_lookback,
                )
                if state is None:
                    continue

                is_open = execution.has_open_pair(state.pair_name)
                pnl = await execution.get_pair_pnl(state.pair_name) if is_open else None

                # Console status per pair
                status_tag = f"[OPEN P&L=${pnl:.2f}]" if is_open else "[watching]"
                corr_warn = " !!LOW_CORR!!" if state.correlation < settings.min_correlation else ""
                logger.info(
                    f"  {tracker.pair_name} {status_tag} | "
                    f"Z={state.zscore:+.3f} (entry±{tracker.cfg.zscore_entry}) | "
                    f"corr={state.correlation:.3f}{corr_warn}"
                )

                # Generate signal (Z-score entry or emergency stop)
                signal_obj = tracker.generate_signal(state, is_open)
                if signal_obj is None:
                    continue

                # Risk check for entries
                if signal_obj.action.startswith("OPEN"):
                    ok, reason = risk_mgr.can_open_pair(state)
                    if not ok:
                        logger.warning(f"  BLOCKED {tracker.pair_name}: {reason}")
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

        except Exception as e:
            logger.exception(f"Bot loop error: {e}")

        await asyncio.sleep(settings.scan_interval_seconds)


@app.on_event("startup")
async def on_startup():
    """Connect MT5 — bot waits for Start button."""
    connected = await mt5_client.connect()
    if not connected:
        logger.error("MT5 connection failed!")

    account = await mt5_client.get_account_info()
    if account:
        risk_mgr.set_start_balance(account["balance"])

    # Always-on tasks — run even when bot is stopped
    asyncio.create_task(display_loop())
    asyncio.create_task(trade_monitor_task())

    logger.info("Server ready on http://localhost:8050 — waiting for Start")


# --- Always-on MT5 Trade Monitor ---
# Checks profit targets and timeouts every 3s, even when bot is stopped.
# This is the ONLY place that closes positions — bot_loop only opens them.

_profit_target_map = {f"{c.leg_a}/{c.leg_b}": c.profit_target for c in PAIRS}
_timeout_map = {f"{c.leg_a}/{c.leg_b}": c.max_hold_minutes for c in PAIRS}


async def trade_monitor_task():
    """Always-on task: monitors open pairs in MT5 and closes on profit target / timeout."""
    global _bot_running
    logger.info("Trade monitor task started (always-on)")

    while True:
        try:
            if execution.active_pairs:
                for pair_name in list(execution.active_pairs.keys()):
                    pair = execution.active_pairs.get(pair_name)
                    if pair is None:
                        continue

                    # Get live P&L from MT5
                    pnl = await execution.get_pair_pnl(pair_name)
                    if pnl is not None:
                        # Track max/min P&L
                        if pnl > pair.max_profit:
                            pair.max_profit = pnl
                        if pnl < pair.min_profit:
                            pair.min_profit = pnl

                    # --- Profit target ---
                    target = _profit_target_map.get(pair_name, 1.0)
                    if pnl is not None and pnl >= target:
                        sig = PairSignal(
                            pair_name=pair_name, action="CLOSE", zscore=0,
                            leg_a=pair.leg_a, leg_b=pair.leg_b,
                            leg_a_side="", leg_b_side="",
                            leg_a_lot=0, leg_b_lot=0,
                            reason=f"Profit target hit (P&L=${pnl:.2f} >= ${target:.2f})",
                        )
                        logger.info(f">>> PROFIT TARGET | {sig.reason}")
                        success = await execution.execute_signal(sig)
                        await broadcast({
                            "type": "signal",
                            "pair": pair_name,
                            "action": "CLOSE",
                            "zscore": 0,
                            "reason": sig.reason,
                            "executed": success,
                        })
                        continue

                    # --- Timeout ---
                    hold_min = (datetime.now(timezone.utc) - pair.entry_time).total_seconds() / 60
                    max_hold = _timeout_map.get(pair_name, 60)
                    if hold_min >= max_hold:
                        sig = PairSignal(
                            pair_name=pair_name, action="CLOSE", zscore=0,
                            leg_a=pair.leg_a, leg_b=pair.leg_b,
                            leg_a_side="", leg_b_side="",
                            leg_a_lot=0, leg_b_lot=0,
                            reason=f"Timeout ({hold_min:.0f}m > {max_hold}m)",
                        )
                        logger.info(f">>> TIMEOUT | {sig.reason}")
                        await execution.execute_signal(sig)
                        await broadcast({
                            "type": "signal",
                            "pair": pair_name,
                            "action": "CLOSE",
                            "zscore": 0,
                            "reason": sig.reason,
                            "executed": True,
                        })
                        continue

                    # Log P&L status
                    pnl_str = f"${pnl:.2f}" if pnl is not None else "N/A"
                    logger.debug(f"  [MONITOR] {pair_name}: P&L={pnl_str} | held {hold_min:.0f}m | target=${target:.2f}")

                # Daily loss circuit breaker
                daily_pnl = execution.get_daily_pnl()
                if daily_pnl <= -settings.max_daily_loss:
                    logger.error(f"DAILY LOSS LIMIT: ${daily_pnl:.2f}")
                    await execution.close_all("daily_loss_limit")
                    await broadcast({"type": "circuit_breaker", "daily_pnl": daily_pnl})
                    _bot_running = False

        except Exception as e:
            logger.error(f"Trade monitor error: {e}")

        await asyncio.sleep(3)  # Check every 3 seconds


_last_spreads = []


async def _compute_spreads_data() -> list:
    """Compute spread data for all pairs — shared by display_loop and REST."""
    global _last_spreads
    spreads_data = []
    for tracker in trackers:
        try:
            state = await tracker.compute_spread(
                timeframe=settings.timeframe,
                lookback=settings.spread_lookback,
            )
        except Exception as e:
            logger.error(f"compute_spread failed for {tracker.pair_name}: {e}")
            continue
        if state is None:
            continue
        is_open = execution.has_open_pair(state.pair_name)
        pnl = await execution.get_pair_pnl(state.pair_name) if is_open else None
        spreads_data.append({
            "pair": state.pair_name,
            "zscore": round(state.zscore, 3),
            "correlation": round(state.correlation, 4),
            "ratio": round(state.ratio, 6),
            "mean": round(state.mean, 6),
            "std": round(state.std, 6),
            "is_open": is_open,
            "pnl": round(pnl, 2) if pnl is not None else None,
            "in_session": tracker._is_in_session(),
        })
    _last_spreads = spreads_data
    return spreads_data


@app.get("/api/spreads")
async def api_spreads():
    """REST endpoint — dashboard fetches this on page load."""
    spreads = await _compute_spreads_data()
    stats = execution.get_stats()
    daily_pnl = execution.get_daily_pnl()
    return {
        "type": "update",
        "spreads": spreads,
        "stats": stats,
        "daily_pnl": round(daily_pnl, 2),
        "active_count": len(execution.active_pairs),
        "cycle": 0,
    }


async def display_loop():
    """Broadcasts full dashboard state every 5s, always running."""
    logger.info("Display loop started")
    _cycle = 0
    while True:
        try:
            if ws_clients:
                _cycle += 1
                spreads_data = await _compute_spreads_data()
                account = await mt5_client.get_account_info()
                stats = execution.get_stats()
                daily_pnl = execution.get_daily_pnl()

                # Active pairs with live P&L
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

                # Trade history (last 50)
                history = [
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

                await broadcast({
                    "type": "update",
                    "spreads": spreads_data,
                    "account": account,
                    "stats": stats,
                    "daily_pnl": round(daily_pnl, 2),
                    "active_count": len(execution.active_pairs),
                    "active_pairs": active,
                    "history": history,
                    "bot_running": _bot_running,
                    "cycle": _cycle,
                })
        except Exception as e:
            logger.error(f"Display loop error: {e}")
        await asyncio.sleep(5)


@app.on_event("shutdown")
async def on_shutdown():
    global _bot_running
    _bot_running = False
    if execution.active_pairs:
        await execution.close_all("shutdown")
    await mt5_client.disconnect()

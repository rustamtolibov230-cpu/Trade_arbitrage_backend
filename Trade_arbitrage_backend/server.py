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
from pydantic import BaseModel

from config.settings import settings, PAIRS
from src.mt5_client import MT5Client
from src.schemas import PairSignal
from src.spread_tracker import SpreadTracker
from src.execution import ExecutionEngine
from src.risk_manager import RiskManager
from src.cointegration import CointegrationMonitor
from src.backtest.cost_model import (
    IC_MARKETS_RAW,
    validate_profit_targets,
    format_cost_check_line,
    check_trade_viability,
)

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
coint_monitor = CointegrationMonitor(PAIRS, mt5_client)
risk_mgr = RiskManager(execution, coint=coint_monitor)
trackers = [SpreadTracker(cfg, mt5_client) for cfg in PAIRS]

# Server-lifetime flag for long-running tasks (separate from bot start/stop)
_server_alive = True

# Set in on_startup by the profit-target sanity check. If False, api_start refuses.
_cost_check_ok = False
_cost_check_messages: list[str] = []

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


@app.get("/api/cointegration")
async def api_cointegration():
    """Current cointegration status for every configured pair."""
    out = []
    for cfg in PAIRS:
        name = f"{cfg.leg_a}/{cfg.leg_b}"
        state = coint_monitor.get_state(name)
        if state is None:
            out.append({
                "pair": name,
                "checked": False,
                "is_cointegrated": False,
                "reason": "pending first check",
            })
            continue
        out.append({
            "pair": name,
            "checked": True,
            "is_cointegrated": state.is_cointegrated,
            "p_value": round(state.p_value, 4),
            "beta": round(state.beta, 6),
            "alpha": round(state.alpha, 6),
            "half_life_bars": (
                round(state.half_life_bars, 2)
                if state.half_life_bars != float("inf") else None
            ),
            "bars_used": state.bars_used,
            "reason": state.reason,
            "last_check": state.last_check.isoformat(),
        })
    return {
        "enabled": settings.require_cointegration,
        "p_threshold": settings.coint_p_threshold,
        "timeframe": settings.coint_timeframe,
        "lookback_bars": settings.coint_lookback_bars,
        "recheck_seconds": settings.coint_recheck_seconds,
        "min_half_life": settings.coint_min_half_life,
        "max_half_life": settings.coint_max_half_life,
        "pairs": out,
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


# --- MT5 login (runtime credentials from dashboard) ---

class MT5LoginRequest(BaseModel):
    login: int
    password: str
    server: str
    path: str | None = None


@app.get("/api/mt5-status")
async def api_mt5_status():
    """Return whether MT5 is connected and which account."""
    return mt5_client.status()


@app.post("/api/mt5-login")
async def api_mt5_login(req: MT5LoginRequest):
    """Connect (or re-connect) MT5 with runtime credentials from the dashboard.

    Stops the bot and closes all pairs before switching accounts to prevent
    orphaned positions against the old login.
    """
    global _bot_running

    # Stop bot + close pairs before switching accounts
    if _bot_running:
        _bot_running = False
        await broadcast({"type": "bot_state", "running": False})
    if execution.active_pairs:
        logger.warning("Closing all pairs before MT5 re-login")
        await execution.close_all("mt5_relogin")

    ok = await mt5_client.connect(
        login=req.login,
        password=req.password,
        server=req.server,
        path=req.path,
    )
    status = mt5_client.status()

    if ok:
        account = await mt5_client.get_account_info()
        if account:
            risk_mgr.set_start_balance(account["balance"])
        await broadcast({"type": "mt5_status", **status})
        return {"status": "connected", **status}

    return {"status": "failed", **status}


@app.post("/api/mt5-logout")
async def api_mt5_logout():
    """Disconnect MT5 and stop the bot."""
    global _bot_running
    if _bot_running:
        _bot_running = False
        await broadcast({"type": "bot_state", "running": False})
    if execution.active_pairs:
        await execution.close_all("mt5_logout")
    await mt5_client.disconnect()
    await broadcast({"type": "mt5_status", "connected": False, "login": 0, "server": "", "last_error": ""})
    return {"status": "disconnected"}


# --- Bot loop (runs as background task) ---
_bot_running = False
_bot_task = None


@app.post("/api/start")
async def api_start():
    """Start the bot loop."""
    global _bot_task, _bot_running
    if _bot_running:
        return {"status": "already_running"}
    if not mt5_client.is_connected():
        return {"status": "error", "error": "MT5 not connected — log in first"}
    if not _cost_check_ok:
        return {
            "status": "error",
            "error": (
                "Profit-target sanity check failed — one or more pairs have "
                "profit_target below the minimum vs round-trip cost. "
                "Fix profit_target or lots in config/settings.py and restart. "
                "See server logs for the per-pair breakdown."
            ),
            "details": _cost_check_messages,
        }
    _bot_running = True  # set BEFORE creating task to prevent double-start race
    _bot_task = asyncio.create_task(bot_loop())
    await broadcast({"type": "bot_state", "running": True})
    logger.info("Bot STARTED by dashboard")
    return {"status": "started"}


@app.get("/api/cost-check")
async def api_cost_check():
    """Surfaces the startup profit-target vs cost sanity check to the dashboard."""
    return {
        "ok": _cost_check_ok,
        "safety_factor": settings.profit_target_safety_factor,
        "messages": _cost_check_messages,
    }


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
                pair_name = f"{tracker.cfg.leg_a}/{tracker.cfg.leg_b}"
                coint_state = coint_monitor.get_state(pair_name)
                state = await tracker.compute_spread(
                    timeframe=settings.timeframe,
                    lookback=settings.spread_lookback,
                    coint_state=coint_state,
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

                # Generate signal (Z-score entry or emergency stop).
                # Pass cointegration state + cost model so entries use
                # OLS-beta-based hedge sizing instead of static lots.
                signal_obj = tracker.generate_signal(
                    state, is_open, coint_state, IC_MARKETS_RAW
                )
                if signal_obj is None:
                    continue

                # Risk check for entries
                if signal_obj.action.startswith("OPEN"):
                    cooldown_left = execution.cooldown_remaining(
                        state.pair_name, settings.cooldown_seconds
                    )
                    if cooldown_left > 0:
                        logger.info(
                            f"  COOLDOWN {tracker.pair_name}: {cooldown_left:.0f}s left"
                        )
                        continue

                    ok, reason = risk_mgr.can_open_pair(state)
                    if not ok:
                        logger.warning(f"  BLOCKED {tracker.pair_name}: {reason}")
                        await broadcast({
                            "type": "risk_block",
                            "pair": state.pair_name,
                            "reason": reason,
                        })
                        continue

                    # Runtime cost check — beta sizing can change leg_b enough
                    # that the trade is no longer viable vs profit_target.
                    cost_ok, total_cost, ratio = check_trade_viability(
                        IC_MARKETS_RAW,
                        signal_obj.leg_a,
                        signal_obj.leg_b,
                        signal_obj.leg_a_lot,
                        signal_obj.leg_b_lot,
                        tracker.cfg.profit_target,
                        settings.profit_target_safety_factor,
                    )
                    if not cost_ok:
                        msg = (
                            f"sized lots make trade unprofitable: "
                            f"cost=${total_cost:.2f}, target=${tracker.cfg.profit_target:.2f}, "
                            f"ratio={ratio:.2f}x < {settings.profit_target_safety_factor}x"
                        )
                        logger.warning(f"  BLOCKED {tracker.pair_name}: {msg}")
                        await broadcast({
                            "type": "risk_block",
                            "pair": state.pair_name,
                            "reason": msg,
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
    """Try MT5 auto-connect from .env; if creds missing/invalid, user logs in from dashboard."""
    # Profit-target vs round-trip cost sanity gate. Runs before any trading
    # can start so a misconfigured pair can never bleed live capital.
    global _cost_check_ok, _cost_check_messages
    sf = settings.profit_target_safety_factor
    all_ok, rows = validate_profit_targets(PAIRS, IC_MARKETS_RAW, safety_factor=sf)
    _cost_check_ok = all_ok
    _cost_check_messages = [format_cost_check_line(r, sf) for r in rows]
    for row, line in zip(rows, _cost_check_messages):
        if row["ok"] and not row.get("error"):
            logger.info(line)
        else:
            logger.error(line)
    if not all_ok:
        logger.error(
            "Profit-target sanity check FAILED — bot start will be blocked. "
            "Fix profit_target or lots in config/settings.py, or adjust "
            "profit_target_safety_factor."
        )

    if settings.mt5_login and settings.mt5_password and settings.mt5_server:
        connected = await mt5_client.connect()
        if connected:
            account = await mt5_client.get_account_info()
            if account:
                risk_mgr.set_start_balance(account["balance"])
        else:
            logger.warning("Auto-connect failed — waiting for dashboard login")
    else:
        logger.info("No MT5 creds in .env — waiting for dashboard login")

    # Always-on tasks — run even when MT5 disconnected / bot stopped
    asyncio.create_task(display_loop())
    asyncio.create_task(trade_monitor_task())
    asyncio.create_task(
        coint_monitor.run_periodic(lambda: not _server_alive)
    )

    logger.info("Server ready on http://localhost:8050 — waiting for Start")


# --- Always-on MT5 Trade Monitor ---
# Checks profit targets and timeouts every 3s, even when bot is stopped.
# This is the ONLY place that closes positions — bot_loop only opens them.

_profit_target_map = {f"{c.leg_a}/{c.leg_b}": c.profit_target for c in PAIRS}
_timeout_map = {f"{c.leg_a}/{c.leg_b}": c.max_hold_minutes for c in PAIRS}
_min_hold_map = {f"{c.leg_a}/{c.leg_b}": c.min_hold_seconds for c in PAIRS}


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
                    # Grace period: bid/ask noise right after entry can swing P&L
                    # enough to trip the profit target on the very first monitor
                    # cycle (especially for metals where 0.01 XAG lot moves ~$50
                    # per $1 silver). Wait min_hold_seconds before honoring it.
                    age_seconds = (datetime.now(timezone.utc) - pair.entry_time).total_seconds()
                    min_hold = _min_hold_map.get(pair_name, 20)
                    target = _profit_target_map.get(pair_name, 1.0)
                    if pnl is not None and pnl >= target and age_seconds < min_hold:
                        logger.debug(
                            f"  [MONITOR] {pair_name}: P&L=${pnl:.2f} >= target but "
                            f"still in grace ({age_seconds:.0f}s/{min_hold}s)"
                        )
                    if pnl is not None and pnl >= target and age_seconds >= min_hold:
                        sig = PairSignal(
                            pair_name=pair_name, action="CLOSE", zscore=0,
                            leg_a=pair.leg_a, leg_b=pair.leg_b,
                            leg_a_side="", leg_b_side="",
                            leg_a_lot=0, leg_b_lot=0,
                            reason=f"Profit target hit (P&L=${pnl:.2f} >= ${target:.2f} after {age_seconds:.0f}s)",
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
        pair_name = f"{tracker.cfg.leg_a}/{tracker.cfg.leg_b}"
        coint_state = coint_monitor.get_state(pair_name)
        try:
            state = await tracker.compute_spread(
                timeframe=settings.timeframe,
                lookback=settings.spread_lookback,
                coint_state=coint_state,
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
            "spread_mode": state.spread_mode,
            "spread_value": round(state.spread_value, 6),
            "is_open": is_open,
            "pnl": round(pnl, 2) if pnl is not None else None,
            "in_session": tracker._is_in_session(),
            "cointegrated": coint_state.is_cointegrated if coint_state else None,
            "coint_p": round(coint_state.p_value, 4) if coint_state else None,
            "coint_half_life": (
                round(coint_state.half_life_bars, 1)
                if coint_state and coint_state.half_life_bars != float("inf")
                else None
            ),
            "coint_reason": coint_state.reason if coint_state else "pending",
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
                    "mt5": mt5_client.status(),
                    "cycle": _cycle,
                })
        except Exception as e:
            logger.error(f"Display loop error: {e}")
        await asyncio.sleep(5)


@app.on_event("shutdown")
async def on_shutdown():
    global _bot_running, _server_alive
    _bot_running = False
    _server_alive = False  # signals coint_monitor.run_periodic to exit
    if execution.active_pairs:
        await execution.close_all("shutdown")
    await mt5_client.disconnect()

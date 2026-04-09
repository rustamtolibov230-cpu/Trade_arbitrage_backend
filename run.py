"""
Trade Arbitrage Launcher
========================
Starts the FastAPI backend (which runs the bot + serves the dashboard)
and opens the browser automatically.

Compile to exe:  python build_exe.py
"""

import os
import sys
import time
import threading
import webbrowser
import urllib.request

# --- Fix paths for PyInstaller bundled exe ---
if getattr(sys, 'frozen', False):
    # Running as exe — PyInstaller extracts to _MEIPASS temp dir
    BUNDLE_DIR = sys._MEIPASS
    # But we want to run from the exe's actual directory (for .env, logs, etc.)
    EXE_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))
    EXE_DIR = BUNDLE_DIR

# Add bundle dir to path so imports work
sys.path.insert(0, BUNDLE_DIR)
os.chdir(EXE_DIR)

# Create logs dir if missing
os.makedirs(os.path.join(EXE_DIR, "logs"), exist_ok=True)

HOST = "127.0.0.1"
PORT = 8050
URL = f"http://{HOST}:{PORT}"

_browser_opened = False


def wait_and_open_browser():
    """Wait for server to actually respond, then open browser ONCE."""
    global _browser_opened
    if _browser_opened:
        return
    _browser_opened = True

    for _ in range(30):  # try for up to 15 seconds
        time.sleep(0.5)
        try:
            urllib.request.urlopen(URL, timeout=1)
            print(f"\n  Dashboard ready: {URL}\n")
            webbrowser.open(URL)
            return
        except Exception:
            pass

    print(f"\n  Server did not start. Open manually: {URL}\n")


def main():
    print("=" * 50)
    print("  TRADE ARBITRAGE")
    print("  Pairs Trading Bot + Dashboard")
    print("=" * 50)
    print(f"\n  Server:    {URL}")
    print(f"  Dashboard: {URL}")
    print(f"  Bundle:    {BUNDLE_DIR}")
    print(f"  Work dir:  {EXE_DIR}")
    print("\n  Press Ctrl+C to stop\n")
    print("-" * 50)

    # Open browser in background — only after server is live
    threading.Thread(target=wait_and_open_browser, daemon=True).start()

    # Import app directly (not as string) — this works in both exe and dev mode
    import uvicorn
    from server import app

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()

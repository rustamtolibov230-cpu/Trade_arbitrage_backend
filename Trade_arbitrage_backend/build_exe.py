"""
Build arbitrage_run.exe — BACKEND ONLY (no frontend bundled).

Frontend is standalone: frontend/index.html
Just open it in a browser — it connects to the backend via API.

Usage: python build_exe.py
Output: dist/arbitrage_run.exe + dist/.env (copied)
"""

import subprocess
import sys
import os
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--onefile",
    "--clean",
    "--name", "arbitrage_run",
    # Bundle source code (no frontend)
    "--add-data", "config;config",
    "--add-data", "src;src",
    "--add-data", "server.py;.",
    # Collect entire packages
    "--collect-all", "loguru",
    "--collect-all", "uvicorn",
    "--collect-all", "fastapi",
    "--collect-all", "starlette",
    "--collect-all", "pydantic",
    "--collect-all", "pydantic_settings",
    "--collect-all", "pydantic_core",
    "--collect-all", "websockets",
    "--collect-all", "anyio",
    "--collect-all", "sniffio",
    "--collect-all", "httptools",
    "--collect-all", "dotenv",
    "--collect-all", "python_dotenv",
    # Hidden imports
    "--hidden-import", "MetaTrader5",
    "--hidden-import", "numpy",
    "--hidden-import", "pandas",
    "--hidden-import", "email.mime.multipart",
    "--hidden-import", "email.mime.text",
    "--hidden-import", "uvicorn.lifespan.on",
    "--hidden-import", "uvicorn.lifespan.off",
    "--hidden-import", "uvicorn.protocols.http.auto",
    "--hidden-import", "uvicorn.protocols.websockets.auto",
    "--hidden-import", "uvicorn.loops.auto",
    "--console",
    "run.py",
]

print("Building arbitrage_run.exe (backend only)...")
print()

result = subprocess.run(cmd)
if result.returncode == 0:
    exe_path = os.path.join(SCRIPT_DIR, "dist", "arbitrage_run.exe")
    size_mb = os.path.getsize(exe_path) / (1024 * 1024)

    # Copy .env next to exe
    env_src = os.path.join(SCRIPT_DIR, ".env")
    env_dst = os.path.join(SCRIPT_DIR, "dist", ".env")
    if os.path.exists(env_src):
        shutil.copy2(env_src, env_dst)

    print("\n" + "=" * 50)
    print("  BUILD SUCCESS!")
    print(f"  Backend:  {exe_path} ({size_mb:.1f} MB)")
    print(f"  Frontend: frontend/index.html (separate)")
    print()
    print("  How to run:")
    print("    1. Start backend:  dist\\arbitrage_run.exe")
    print("    2. Open frontend:  frontend\\index.html")
    print("=" * 50)
else:
    print("\nBuild failed!")
    sys.exit(1)

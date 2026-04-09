"""
Build run.exe using PyInstaller.

Usage: python build_exe.py
Output: dist/run.exe
"""

import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--onefile",
    "--clean",
    "--name", "run",
    # Bundle our source code + frontend
    "--add-data", "frontend;frontend",
    "--add-data", "config;config",
    "--add-data", "src;src",
    "--add-data", "server.py;.",
    # Collect entire packages (--hidden-import alone isn't enough)
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
    # Hidden imports for things PyInstaller misses
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

print("Building run.exe...")
print()

result = subprocess.run(cmd)
if result.returncode == 0:
    exe_path = os.path.join(SCRIPT_DIR, "dist", "run.exe")
    size_mb = os.path.getsize(exe_path) / (1024 * 1024)
    print("\n" + "=" * 50)
    print("  BUILD SUCCESS!")
    print(f"  Output: {exe_path}")
    print(f"  Size:   {size_mb:.1f} MB")
    print("=" * 50)
else:
    print("\nBuild failed!")
    sys.exit(1)

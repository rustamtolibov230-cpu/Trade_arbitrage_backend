"""Build backtest.exe — standalone backtest runner.

Usage:  python build_backtest_exe.py
Output: dist/backtest.exe (+ dist/.env copied for MT5 creds)

Double-click the exe → interactive prompts (pair, days, save?).
Or run with CLI flags:  dist\\backtest.exe --pair BTCUSD/ETHUSD --days 60 --save reports
"""

import os
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--onefile",
    "--clean",
    "--name", "backtest",
    # Bundle source so config + src modules resolve at runtime
    "--add-data", "config;config",
    "--add-data", "src;src",
    # Packages with hidden submodules / data files — collect everything
    "--collect-all", "loguru",
    "--collect-all", "statsmodels",
    "--collect-all", "scipy",
    "--collect-all", "patsy",
    "--collect-all", "numpy",
    "--collect-all", "pandas",
    "--collect-all", "pydantic",
    "--collect-all", "pydantic_settings",
    "--collect-all", "pydantic_core",
    "--collect-all", "dotenv",
    "--collect-all", "pyarrow",     # parquet cache (needs data files + C ext)
    # Hidden imports (PyInstaller's static analyzer misses these)
    "--hidden-import", "MetaTrader5",
    "--console",
    "backtest.py",
]

print("Building backtest.exe...")
print()

result = subprocess.run(cmd)
if result.returncode != 0:
    print("\nBuild failed!")
    sys.exit(1)

exe_path = os.path.join(SCRIPT_DIR, "dist", "backtest.exe")
size_mb = os.path.getsize(exe_path) / (1024 * 1024)

# Copy .env so MT5 auto-connects from cached creds
env_src = os.path.join(SCRIPT_DIR, ".env")
env_dst = os.path.join(SCRIPT_DIR, "dist", ".env")
if os.path.exists(env_src):
    shutil.copy2(env_src, env_dst)

print()
print("=" * 60)
print("  BUILD SUCCESS!")
print(f"  Exe:    {exe_path} ({size_mb:.1f} MB)")
print()
print("  How to run:")
print("    Double-click:  dist\\backtest.exe   (interactive prompts)")
print("    CLI:           dist\\backtest.exe --pair BTCUSD/ETHUSD --days 60 --save reports")
print()
print("  First run will fetch historical bars from MT5 and cache them in")
print("  dist/data/backtest_cache/. Subsequent runs are fast (parquet read).")
print("=" * 60)

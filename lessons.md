# Lessons Learned

## 2026-04-09 — Initial Build

### MT5 Filling Mode Must Be Auto-Detected
- **Problem**: All orders rejected with "Unsupported filling mode" — broker uses FOK for crypto, IOC for metals
- **Root cause**: Hardcoded `ORDER_FILLING_IOC` for all symbols
- **Fix**: Read `symbol_info().filling_mode` bitmask and pick the correct mode per symbol
- **Rule**: Never hardcode filling mode. Always auto-detect from symbol info.

### PyInstaller + `uvicorn.run("server:app")` String Import Fails
- **Problem**: Exe crashes with ModuleNotFoundError — PyInstaller can't resolve string-based module imports
- **Fix**: Import `app` directly and pass the object: `uvicorn.run(app, ...)`
- **Rule**: In PyInstaller builds, always pass app objects directly, never use string module paths.

### PyInstaller `--hidden-import` Is Not Enough
- **Problem**: `--hidden-import loguru` didn't bundle loguru — it was not installed in the base Python env
- **Fix**: `pip install` all deps globally (not just in venv), use `--collect-all` for key packages
- **Rule**: Verify deps are installed in the Python that PyInstaller uses.

### Python 3.14 Module-Level Set Mutation Needs `global`
- **Problem**: `ws_clients -= dead` raised "cannot access local variable" in Python 3.14
- **Fix**: Added `global ws_clients` in functions that use `-=` on module-level sets
- **Rule**: Any augmented assignment (`-=`, `+=`) on module-level variables requires `global` in Python 3.14+.

### Z-Score Thresholds Must Match Timeframe
- **Problem**: Z-entry=2.0 on M5 with lookback=100 produced ~1 signal/day — bot appeared broken
- **Fix**: Switched to M1, lookback=50, Z-entry=1.0 → ~10 signals per 50 minutes
- **Rule**: Always check actual Z-score distribution before setting thresholds. Run a quick diagnostic first.

### Bot Should Not Auto-Start
- **Decision**: User wants manual Start button, not auto-trade on launch
- **Why**: Prevents accidental trades when just checking the dashboard

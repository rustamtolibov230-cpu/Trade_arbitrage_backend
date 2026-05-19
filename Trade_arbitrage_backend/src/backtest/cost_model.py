"""Broker cost model — per-symbol spread, commission, contract size.

The live `MT5Client` gives us bar close prices (mid). To simulate realistic
fills we need to translate:
  - BUY fills at   close + spread/2
  - SELL fills at  close - spread/2
  - Commission charged per side per lot (separate from spread)

Contract size converts price moves to USD P&L:
  pnl_usd = price_move * lot * contract_size  (for USD-quoted symbols)

IC Markets Raw Spread figures below are typical-case averages — spreads
widen during news and thin liquidity hours. They are intentionally a bit
conservative (higher than best-case) so backtest results don't overstate.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class SymbolCost:
    """Cost + contract params for one symbol on one broker."""

    symbol: str
    contract_size: float        # units of asset per 1.0 lot
    typical_spread: float       # in quote-currency price units
    commission_per_lot: float   # USD per 1.0 lot, per side
    min_lot: float = 0.01
    lot_step: float = 0.01

    def fill_price(self, close_mid: float, side: str) -> float:
        """Realistic fill: mid adjusted by half-spread in adverse direction."""
        half = self.typical_spread / 2.0
        if side.upper() == "BUY":
            return close_mid + half
        return close_mid - half

    def leg_pnl(
        self,
        side: str,
        entry_price: float,
        exit_price: float,
        lot: float,
    ) -> float:
        """Raw price P&L for one leg (spread already baked into prices)."""
        if side.upper() == "BUY":
            move = exit_price - entry_price
        else:
            move = entry_price - exit_price
        return move * lot * self.contract_size

    def round_trip_commission(self, lot: float) -> float:
        """Commission for open + close (both sides) on this leg."""
        return 2.0 * self.commission_per_lot * lot


class BrokerCostModel:
    """Lookup: symbol → SymbolCost. Supports sensible defaults per broker."""

    def __init__(self, costs: Dict[str, SymbolCost], name: str = "custom"):
        self.costs = costs
        self.name = name

    def get(self, symbol: str) -> Optional[SymbolCost]:
        return self.costs.get(symbol)

    def require(self, symbol: str) -> SymbolCost:
        cost = self.costs.get(symbol)
        if cost is None:
            raise KeyError(
                f"No cost model for {symbol!r} in broker {self.name!r}. "
                f"Add it to src/backtest/cost_model.py."
            )
        return cost


# ---------------------------------------------------------------------------
# IC Markets — Raw Spread account defaults (typical live spreads)
#
# Sources: IC Markets public typical-spread tables, forum data, and aggregated
# user reports. Intentionally chosen on the pessimistic side of typical.
# Commission model: cTrader/Raw Spread charges a flat fee per side per lot.
# ---------------------------------------------------------------------------
IC_MARKETS_RAW = BrokerCostModel(
    name="IC Markets Raw Spread",
    costs={
        # Crypto — 1 lot = 1 unit of the base asset on IC Markets.
        # Typical spreads are highly variable; values below are realistic
        # daytime averages on a calm market.
        "BTCUSD": SymbolCost(
            symbol="BTCUSD",
            contract_size=1.0,
            typical_spread=8.00,         # $8 spread on BTC is calm-market typical
            commission_per_lot=3.50,     # $3.50 per side per lot
            min_lot=0.01,
            lot_step=0.01,
        ),
        "ETHUSD": SymbolCost(
            symbol="ETHUSD",
            contract_size=1.0,
            typical_spread=1.20,         # ~$1.20 on ETH
            commission_per_lot=3.50,
            min_lot=0.01,
            lot_step=0.01,
        ),
        # Metals — 1 lot XAU = 100 oz, 1 lot XAG = 5000 oz on IC Markets.
        # Spreads are in quote currency (USD per oz).
        "XAUUSD": SymbolCost(
            symbol="XAUUSD",
            contract_size=100.0,
            typical_spread=0.12,         # ~12 cents per oz
            commission_per_lot=3.50,     # Raw Spread: ~$3.50 per side per lot
            min_lot=0.01,
            lot_step=0.01,
        ),
        "XAGUSD": SymbolCost(
            symbol="XAGUSD",
            contract_size=5000.0,
            typical_spread=0.020,        # ~2 cents per oz
            commission_per_lot=3.50,
            min_lot=0.01,
            lot_step=0.01,
        ),
        # --- Energy ---
        # WTI / Brent crude on BlackBull: usually 1 lot = 1000 barrels.
        # If your broker uses 100, halve contract_size and double targets.
        "USOIL": SymbolCost(
            symbol="USOIL",
            contract_size=1000.0,
            typical_spread=0.03,         # ~3 cents per barrel raw spread
            commission_per_lot=3.50,
            min_lot=0.01,
            lot_step=0.01,
        ),
        "UKOIL": SymbolCost(
            symbol="UKOIL",
            contract_size=1000.0,
            typical_spread=0.03,
            commission_per_lot=3.50,
            min_lot=0.01,
            lot_step=0.01,
        ),
        # --- Indices ---
        # 1 lot ~ $1 per index point on most retail brokers (BlackBull included).
        # Spreads in *index points*.
        "NAS100": SymbolCost(
            symbol="NAS100",
            contract_size=1.0,
            typical_spread=2.0,          # 2 points typical
            commission_per_lot=0.0,      # most CFD indices spread-only
            min_lot=0.1,
            lot_step=0.1,
        ),
        "SPX500": SymbolCost(
            symbol="SPX500",
            contract_size=1.0,
            typical_spread=0.7,          # 0.7 points typical
            commission_per_lot=0.0,
            min_lot=0.1,
            lot_step=0.1,
        ),
        # --- FX ---
        # Standard lot = 100,000 base currency. Spread in price units
        # (price digits = 0.00001 for 5-decimal pairs).
        "AUDUSD": SymbolCost(
            symbol="AUDUSD",
            contract_size=100000.0,
            typical_spread=0.00015,      # ~1.5 pips raw
            commission_per_lot=3.50,
            min_lot=0.01,
            lot_step=0.01,
        ),
        "NZDUSD": SymbolCost(
            symbol="NZDUSD",
            contract_size=100000.0,
            typical_spread=0.00020,      # ~2 pips raw
            commission_per_lot=3.50,
            min_lot=0.01,
            lot_step=0.01,
        ),
    },
)


def round_trip_cost_for_pair(
    cost_model: BrokerCostModel,
    leg_a_symbol: str,
    leg_b_symbol: str,
    leg_a_lot: float,
    leg_b_lot: float,
) -> Dict[str, float]:
    """Break down total round-trip cost (spread + commission, both legs).

    Useful for sanity-checking whether the profit target is even reachable.
    """
    cost_a = cost_model.require(leg_a_symbol)
    cost_b = cost_model.require(leg_b_symbol)

    # Spread cost = full spread * lot * contract_size (half in + half out)
    spread_cost_a = cost_a.typical_spread * leg_a_lot * cost_a.contract_size
    spread_cost_b = cost_b.typical_spread * leg_b_lot * cost_b.contract_size

    comm_a = cost_a.round_trip_commission(leg_a_lot)
    comm_b = cost_b.round_trip_commission(leg_b_lot)

    return {
        "spread_cost_a": spread_cost_a,
        "spread_cost_b": spread_cost_b,
        "commission_a": comm_a,
        "commission_b": comm_b,
        "total": spread_cost_a + spread_cost_b + comm_a + comm_b,
    }


def validate_profit_targets(
    pairs,
    cost_model: BrokerCostModel,
    safety_factor: float = 1.5,
):
    """Verify every pair's profit_target exceeds safety_factor × round-trip cost.

    Returns (all_ok, results). Each result dict has:
        pair, ok, target, cost, ratio, min_target, components, error
    Callers should log each result (use ``format_cost_check_line``) and refuse
    to start the bot if ``all_ok`` is False.
    """
    results = []
    all_ok = True
    for cfg in pairs:
        pair_name = f"{cfg.leg_a}/{cfg.leg_b}"
        try:
            components = round_trip_cost_for_pair(
                cost_model, cfg.leg_a, cfg.leg_b, cfg.leg_a_lot, cfg.leg_b_lot
            )
        except KeyError as e:
            results.append({
                "pair": pair_name,
                "ok": False,
                "target": cfg.profit_target,
                "error": f"no cost model entry — {e}",
            })
            all_ok = False
            continue

        total = components["total"]
        target = cfg.profit_target
        ratio = target / total if total > 0 else float("inf")
        ok = ratio >= safety_factor
        if not ok:
            all_ok = False
        results.append({
            "pair": pair_name,
            "ok": ok,
            "target": target,
            "cost": total,
            "ratio": ratio,
            "min_target": total * safety_factor,
            "components": components,
        })
    return all_ok, results


def check_trade_viability(
    cost_model: BrokerCostModel,
    leg_a_symbol: str,
    leg_b_symbol: str,
    leg_a_lot: float,
    leg_b_lot: float,
    profit_target: float,
    safety_factor: float = 1.5,
) -> tuple[bool, float, float]:
    """Runtime cost check for a specific lot combo (post-beta sizing).

    The startup gate (validate_profit_targets) uses the *configured* lots; once
    OLS-beta sizing dynamically scales leg_b, real costs can drift. Call this
    right before opening a trade with the lots that will actually be sent.

    Returns (ok, total_cost, ratio).
    """
    try:
        costs = round_trip_cost_for_pair(
            cost_model, leg_a_symbol, leg_b_symbol, leg_a_lot, leg_b_lot
        )
    except KeyError:
        return False, 0.0, 0.0
    total = costs["total"]
    if total <= 0:
        return True, 0.0, float("inf")
    ratio = profit_target / total
    return ratio >= safety_factor, total, ratio


def format_cost_check_line(result: dict, safety_factor: float = 1.5) -> str:
    """Pretty one-line summary for a single validate_profit_targets result."""
    if result.get("error"):
        return (
            f"[COST] {result['pair']} (target ${result['target']:.2f}): "
            f"{result['error']}"
        )
    status = "PASS" if result["ok"] else "FAIL"
    c = result["components"]
    line = (
        f"[COST] {result['pair']} [{status}]: "
        f"target=${result['target']:.2f}, "
        f"round-trip=${result['cost']:.2f} "
        f"(A sprd ${c['spread_cost_a']:.2f}, B sprd ${c['spread_cost_b']:.2f}, "
        f"A comm ${c['commission_a']:.2f}, B comm ${c['commission_b']:.2f}) "
        f"| ratio={result['ratio']:.2f}x (min {safety_factor}x)"
    )
    if not result["ok"]:
        line += (
            f" -- raise target to >=${result['min_target']:.2f} "
            f"or reduce lots"
        )
    return line

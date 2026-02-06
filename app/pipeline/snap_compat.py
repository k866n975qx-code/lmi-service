"""V4/V5 snapshot compatibility accessors.

Downstream consumers (periods, diff, alerts, charts, ai_insights) use these
helpers to read daily snapshots without caring whether the source is v4 or v5.

V5 key changes:
  totals            → portfolio.totals
  income            → portfolio.income  (+ income_stability, income_growth)
  portfolio_rollups → portfolio          (performance, risk restructured)
  goal_progress     → goals.baseline
  goal_progress_net → goals.net_of_interest
  goal_tiers        → goals  (tiers[], current_state)
  goal_pace         → goals.pace
  margin_stress     → margin  (current, stress restructured)
  margin_guidance   → margin.guidance
  dividends_upcoming→ dividends.upcoming_this_month
  coverage          → meta.data_quality
  ultimate          → analytics  (nested: distribution/risk/performance)
  holdings flat     → holdings nested (cost/valuation/income)
  as_of/as_of_date_local → timestamps.portfolio_data_as_of_local
"""
from __future__ import annotations
from datetime import date


# ── Portfolio-level sections ──────────────────────────────────────────────────


def get_totals(snap: dict) -> dict:
    return snap.get("totals") or (snap.get("portfolio") or {}).get("totals") or {}


def get_income(snap: dict) -> dict:
    return snap.get("income") or (snap.get("portfolio") or {}).get("income") or {}


def get_perf(snap: dict) -> dict:
    v4 = (snap.get("portfolio_rollups") or {}).get("performance")
    if v4:
        return v4
    return (snap.get("portfolio") or {}).get("performance") or {}


def get_risk_flat(snap: dict) -> dict:
    """Return flat risk dict (all metrics at top level) from v4 or v5."""
    v4 = (snap.get("portfolio_rollups") or {}).get("risk")
    if v4:
        return v4
    pr = (snap.get("portfolio") or {}).get("risk") or {}
    flat: dict = {}
    for key in ("volatility", "ratios", "drawdown", "var"):
        sub = pr.get(key)
        if isinstance(sub, dict):
            flat.update(sub)
    for key in ("tail_risk", "beta_portfolio", "portfolio_risk_quality", "income_stability_score"):
        if pr.get(key) is not None:
            flat[key] = pr[key]
    return flat


def get_rollups(snap: dict) -> dict:
    """Return a portfolio_rollups-compatible dict from v4 or v5."""
    v4 = snap.get("portfolio_rollups")
    if v4:
        return v4
    portfolio = snap.get("portfolio") or {}
    inc = portfolio.get("income") or {}
    attr = portfolio.get("attribution") or {}
    perf_block = portfolio.get("performance") or {}
    return {
        "performance": perf_block,
        "risk": get_risk_flat(snap),
        "income_stability": inc.get("income_stability"),
        "income_growth": inc.get("income_growth"),
        "tail_risk": (portfolio.get("risk") or {}).get("tail_risk"),
        "vs_benchmark": perf_block.get("vs_benchmark"),
        "return_attribution_1m": attr.get("1m"),
        "return_attribution_3m": attr.get("3m"),
        "return_attribution_6m": attr.get("6m"),
        "return_attribution_12m": attr.get("12m"),
    }


# ── Timestamps ────────────────────────────────────────────────────────────────


def get_as_of(snap: dict) -> str:
    """Get as_of date from v4 or v5 snapshot."""
    # v5: check timestamps first
    timestamps = snap.get("timestamps") or {}
    if timestamps.get("portfolio_data_as_of_local"):
        return timestamps.get("portfolio_data_as_of_local")
    # v4: direct as_of or as_of_date_local field
    v4_as_of = snap.get("as_of") or snap.get("as_of_date_local")
    if v4_as_of:
        if isinstance(v4_as_of, str) and len(v4_as_of) >= 10:
            return v4_as_of[:10]
        return str(v4_as_of)[:10]
    return ""


# ── Goals ─────────────────────────────────────────────────────────────────────


def get_goal_progress(snap: dict) -> dict:
    return snap.get("goal_progress") or (snap.get("goals") or {}).get("baseline") or {}


def get_goal_progress_net(snap: dict) -> dict:
    return snap.get("goal_progress_net") or (snap.get("goals") or {}).get("net_of_interest") or {}


def get_goal_tiers(snap: dict) -> dict:
    v4 = snap.get("goal_tiers")
    if v4:
        return v4
    goals = snap.get("goals") or {}
    return {
        "tiers": goals.get("tiers") or [],
        "current_state": goals.get("current_state"),
        "provenance": goals.get("tiers_provenance"),
    }


def get_goal_pace(snap: dict) -> dict:
    return snap.get("goal_pace") or (snap.get("goals") or {}).get("pace") or {}


# ── Margin ────────────────────────────────────────────────────────────────────


def get_margin_stress(snap: dict) -> dict:
    """Return a v4-compatible margin_stress dict from v4 or v5."""
    v4 = snap.get("margin_stress")
    if v4:
        return v4
    margin = snap.get("margin") or {}
    stress = margin.get("stress") or {}
    result: dict = {}
    if margin.get("current"):
        result["current"] = margin["current"]
    if margin.get("efficiency"):
        result["efficiency"] = margin["efficiency"]
    if margin.get("history_90d"):
        result["historical_trends_90d"] = margin["history_90d"]
    scenarios: dict = {}
    mc = stress.get("margin_call_risk")
    if mc:
        buf = mc.get("buffer_to_margin_call_pct")
        scenarios["margin_call_distance"] = {
            "portfolio_decline_to_call_pct": -abs(buf) if isinstance(buf, (int, float)) else buf,
            "dollar_decline_to_call": mc.get("dollar_decline_to_call"),
            "days_at_current_volatility": mc.get("days_at_current_volatility"),
            "buffer_status": mc.get("buffer_status"),
        }
    rate_arr = stress.get("rate_shock_scenarios") or []
    if rate_arr:
        rate_dict: dict = {}
        for item in rate_arr:
            scenario = item.get("scenario", "")
            label = "rate_plus_" + scenario.replace("+", "").strip()
            entry = {k: v for k, v in item.items() if k != "scenario"}
            rate_dict[label] = entry
        scenarios["interest_rate_shock"] = rate_dict
    if scenarios:
        result["stress_scenarios"] = scenarios
    return result


def get_margin_guidance(snap: dict) -> dict:
    """Return a v4-compatible margin_guidance dict from v4 or v5."""
    v4 = snap.get("margin_guidance")
    if v4:
        return v4
    g = (snap.get("margin") or {}).get("guidance") or {}
    return {
        "selected_mode": g.get("recommended_mode"),
        "modes": g.get("modes"),
        "rates": g.get("rates"),
    }


# ── Dividends ─────────────────────────────────────────────────────────────────


def get_dividends(snap: dict) -> dict:
    """Return v4-compatible dividends dict."""
    v4 = snap.get("dividends") or {}
    if v4.get("realized_mtd") or v4.get("windows"):
        return v4
    realized = v4.get("realized") or {}
    result = dict(v4)
    if realized.get("mtd"):
        result["realized_mtd"] = realized["mtd"]
    windows: dict = {}
    for key in ("30d", "ytd", "qtd"):
        if realized.get(key):
            windows[key] = realized[key]
    if windows:
        result["windows"] = windows
    return result


def get_dividends_upcoming(snap: dict) -> dict:
    v4 = snap.get("dividends_upcoming")
    if v4:
        return v4
    return (snap.get("dividends") or {}).get("upcoming_this_month") or {}


# ── Coverage / data quality ───────────────────────────────────────────────────


def get_coverage(snap: dict) -> dict:
    v4 = snap.get("coverage")
    if isinstance(v4, dict):
        return v4
    return (snap.get("meta") or {}).get("data_quality") or {}


# ── Macro ─────────────────────────────────────────────────────────────────────


def get_macro_snapshot(snap: dict) -> dict:
    return (snap.get("macro") or {}).get("snapshot") or {}


# ── Date ──────────────────────────────────────────────────────────────────────


def get_as_of(snap: dict) -> str | None:
    v4 = snap.get("as_of_date_local") or snap.get("as_of")
    if v4:
        return v4
    return (snap.get("timestamps") or {}).get("portfolio_data_as_of_local")


# ── Holdings ──────────────────────────────────────────────────────────────────


def get_holding_ultimate(h: dict) -> dict:
    """Get flat ultimate/analytics dict from a v4 or v5 holding."""
    v4 = h.get("ultimate")
    if v4:
        return v4
    analytics = h.get("analytics") or {}
    flat: dict = {}
    for key in ("distribution", "risk", "performance"):
        sub = analytics.get(key)
        if isinstance(sub, dict):
            flat.update(sub)
    return flat


def normalize_holding(h: dict) -> dict:
    """Normalize a v5 holding to flat v4-like access.

    V5 nests cost/valuation/income/analytics under sub-keys.
    This returns a dict with all fields at top level for backward compat.
    """
    if h.get("weight_pct") is not None or h.get("market_value") is not None:
        return h
    flat: dict = {"symbol": h.get("symbol"), "shares": h.get("shares"), "trades": h.get("trades_count")}
    for section in ("cost", "valuation", "income"):
        sub = h.get(section) or {}
        flat.update(sub)
    if "portfolio_weight_pct" in flat and "weight_pct" not in flat:
        flat["weight_pct"] = flat["portfolio_weight_pct"]
    if "avg_cost_per_share" in flat and "avg_cost" not in flat:
        flat["avg_cost"] = flat["avg_cost_per_share"]
    flat["ultimate"] = get_holding_ultimate(h)
    flat["dividend_reliability"] = h.get("reliability") or h.get("dividend_reliability")
    return flat


def get_holdings_flat(snap: dict) -> list[dict]:
    """Get holdings list with each holding normalized to flat v4 format."""
    return [normalize_holding(h) for h in (snap.get("holdings") or [])]

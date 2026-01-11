from __future__ import annotations

import json
import sqlite3
from datetime import date

HOLDINGS_CHANGE_MIN_WEIGHT_PCT = 0.10
HOLDINGS_CHANGE_MIN_MARKET_VALUE = 25.0
_DIFF_TOL_MONEY = 0.05
_DIFF_TOL_PCT = 0.1
_HOLDING_ULTIMATE_FIELDS = {"sortino_1y", "sortino_6m", "sortino_3m", "sortino_1m"}


def _load_daily_snapshot(conn: sqlite3.Connection, as_of_date_local: str):
    cur = conn.cursor()
    row = cur.execute(
        "SELECT payload_json FROM snapshot_daily_current WHERE as_of_date_local=?",
        (as_of_date_local,),
    ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


def _delta(left, right, precision=3):
    if left is None or right is None:
        return None
    try:
        return round(float(right) - float(left), precision)
    except (TypeError, ValueError):
        return None


def _section_diff(left: dict, right: dict, keys: list[str], precision=3):
    out = {}
    for key in keys:
        out[key] = {
            "left": left.get(key) if left else None,
            "right": right.get(key) if right else None,
            "delta": _delta(left.get(key) if left else None, right.get(key) if right else None, precision),
        }
    return out


def _is_number(val):
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def _safe_divide(a, b):
    if a is None or b in (None, 0, 0.0):
        return None
    return float(a / b)


def _format_money(val):
    if val is None:
        return "n/a"
    sign = "-" if val < 0 else "+"
    return f"{sign}${abs(val):.2f}"


def _format_pct(val):
    if val is None:
        return "n/a"
    sign = "-" if val < 0 else "+"
    return f"{sign}{abs(val):.2f}%"


def _format_number(val, precision=2):
    if val is None:
        return "n/a"
    sign = "-" if val < 0 else "+"
    return f"{sign}{abs(val):.{precision}f}"


def _annualized_return_pct(simple_return_pct, days_apart):
    if simple_return_pct is None or not days_apart:
        return None
    try:
        base = 1.0 + (float(simple_return_pct) / 100.0)
        if base <= 0:
            return None
        annualized = (base ** (365.0 / float(days_apart)) - 1.0) * 100.0
        return round(annualized, 2)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _numeric_tree(left, right, precision=3):
    if _is_number(left) or _is_number(right):
        return {
            "left": left if _is_number(left) else None,
            "right": right if _is_number(right) else None,
            "delta": _delta(left if _is_number(left) else None, right if _is_number(right) else None, precision),
        }
    if isinstance(left, dict) or isinstance(right, dict):
        out = {}
        keys = set(left.keys() if isinstance(left, dict) else []) | set(right.keys() if isinstance(right, dict) else [])
        for key in keys:
            sub = _numeric_tree(
                left.get(key) if isinstance(left, dict) else None,
                right.get(key) if isinstance(right, dict) else None,
                precision,
            )
            if sub:
                out[key] = sub
        return out or None
    return None


def _holdings_map(snapshot: dict):
    holdings = snapshot.get("holdings") or []
    out = {}
    for holding in holdings:
        sym = holding.get("symbol")
        if not sym:
            continue
        out[sym] = holding
    return out


def _holding_field_value(holding: dict, field: str):
    if not holding:
        return None
    val = holding.get(field)
    if val is None and field in _HOLDING_ULTIMATE_FIELDS:
        val = (holding.get("ultimate") or {}).get(field)
    return val


def _holding_fields_diff(left: dict, right: dict, fields: list[str], missing_note: str | None = None):
    out = {}
    for field in fields:
        lval = _holding_field_value(left, field)
        rval = _holding_field_value(right, field)
        precision = 6 if field == "shares" else 3
        if _is_number(lval) or _is_number(rval):
            if missing_note and lval is None and _is_number(rval):
                lval = 0.0
                delta = _delta(lval, rval, precision)
                out[field] = {"left": lval, "right": rval, "delta": delta, "note": missing_note}
                continue
            if missing_note and rval is None and _is_number(lval):
                rval = 0.0
                delta = _delta(lval, rval, precision)
                out[field] = {"left": lval, "right": rval, "delta": delta, "note": missing_note}
                continue
            delta = _delta(lval, rval, precision)
            out[field] = {"left": lval, "right": rval, "delta": delta}
    return out


def _holdings_weight_shift(left: dict, right: dict):
    left_map = _holdings_map(left)
    right_map = _holdings_map(right)
    symbols = set(left_map.keys()) | set(right_map.keys())
    total = 0.0
    for sym in symbols:
        lval = left_map.get(sym, {}).get("weight_pct") or 0.0
        rval = right_map.get(sym, {}).get("weight_pct") or 0.0
        if _is_number(lval) and _is_number(rval):
            total += abs(rval - lval)
    return round(total, 3)


def _holding_income_delta(left: dict, right: dict):
    lval = left.get("projected_monthly_dividend")
    rval = right.get("projected_monthly_dividend")
    return _delta(lval, rval, 3)


def _holdings_diff(left: dict, right: dict, min_weight_delta: float | None = None, min_mv_delta: float | None = None):
    left_map = _holdings_map(left)
    right_map = _holdings_map(right)
    symbols = sorted(set(left_map.keys()) | set(right_map.keys()))
    added = []
    removed = []
    changed = []
    fields = [
        "shares",
        "market_value",
        "weight_pct",
        "last_price",
        "unrealized_pnl",
        "sortino_1y",
        "sortino_6m",
        "sortino_3m",
        "sortino_1m",
    ]
    for sym in symbols:
        lval = left_map.get(sym, {})
        rval = right_map.get(sym, {})
        if sym not in left_map:
            added.append({"symbol": sym, "fields": _holding_fields_diff({}, rval, fields, missing_note="not owned on left")})
        elif sym not in right_map:
            removed.append({"symbol": sym, "fields": _holding_fields_diff(lval, {}, fields, missing_note="not owned on right")})
        else:
            fields_diff = _holding_fields_diff(lval, rval, fields)
            if any(v.get("delta") not in (None, 0.0) for v in fields_diff.values()):
                shares_delta = fields_diff.get("shares", {}).get("delta")
                mv_delta = fields_diff.get("market_value", {}).get("delta")
                shares_delta = 0.0 if shares_delta is not None and abs(shares_delta) < 1e-3 else shares_delta
                mv_delta = 0.0 if mv_delta is not None and abs(mv_delta) < 0.01 else mv_delta
                if shares_delta not in (None, 0.0) and mv_delta not in (None, 0.0):
                    change_type = "mixed"
                elif shares_delta not in (None, 0.0):
                    change_type = "size_change"
                else:
                    change_type = "price_move"
                include = True
                if min_weight_delta is not None or min_mv_delta is not None:
                    weight_delta = fields_diff.get("weight_pct", {}).get("delta")
                    mv_delta_val = fields_diff.get("market_value", {}).get("delta")
                    weight_ok = False
                    mv_ok = False
                    if min_weight_delta is not None and weight_delta is not None:
                        weight_ok = abs(weight_delta) >= min_weight_delta
                    if min_mv_delta is not None and mv_delta_val is not None:
                        mv_ok = abs(mv_delta_val) >= min_mv_delta
                    if not (weight_ok or mv_ok):
                        include = False
                if include:
                    changed.append(
                        {
                            "symbol": sym,
                            "change_type": change_type,
                            "impact_on_income_monthly": _holding_income_delta(lval, rval),
                            "fields": fields_diff,
                        }
                    )
    changed.sort(key=lambda d: abs(d["fields"].get("market_value", {}).get("delta") or 0.0), reverse=True)
    return {"added": added, "removed": removed, "changed": changed}


def _summary_block(left: dict, right: dict, holdings_diff: dict, days_apart: int | None = None, period_type: str | None = None, range_metrics: dict | None = None):
    left_totals = left.get("totals", {}) if left else {}
    right_totals = right.get("totals", {}) if right else {}
    mv_left = left_totals.get("market_value")
    mv_right = right_totals.get("market_value")
    mv_delta = _delta(mv_left, mv_right, 2)
    mv_pct = _safe_divide(mv_delta, mv_left) * 100 if mv_left else None

    unreal_left = left_totals.get("unrealized_pnl")
    unreal_right = right_totals.get("unrealized_pnl")
    unreal_delta = _delta(unreal_left, unreal_right, 2)

    income_left = (left.get("income") or {}).get("forward_12m_total")
    income_right = (right.get("income") or {}).get("forward_12m_total")
    income_delta = _delta(income_left, income_right, 2)

    if mv_delta is None or unreal_delta is None:
        direction = "mixed"
    else:
        if abs(mv_delta) < 1 and abs(unreal_delta) < 1:
            direction = "neutral"
        elif mv_delta > 0 and unreal_delta > 0:
            direction = "positive"
        elif mv_delta < 0 and unreal_delta < 0:
            direction = "negative"
        else:
            direction = "mixed"

    weight_shift = _holdings_weight_shift(left, right)
    if weight_shift >= 1.0:
        primary_driver = "allocation_change"
    elif income_delta is not None and abs(income_delta) >= 0.01:
        primary_driver = "income_change"
    else:
        primary_driver = "price_move"

    if days_apart is None:
        days_apart = 0
        try:
            left_date = left.get("as_of")
            right_date = right.get("as_of")
            if left_date and right_date:
                days_apart = abs((date.fromisoformat(right_date) - date.fromisoformat(left_date)).days)
        except Exception:
            days_apart = 0

    if mv_delta is None:
        headline = "Mixed change with incomplete data."
    else:
        magnitude = "small" if abs(mv_delta) < 10 else "modest" if abs(mv_delta) < 100 else "large"
        move = "drawdown" if mv_delta < 0 else "gain" if mv_delta > 0 else "flat move"
        income_desc = "flat income" if income_delta is None or abs(income_delta) < 0.01 else ("higher income" if income_delta > 0 else "lower income")
        if days_apart:
            day_word = "day" if days_apart == 1 else "days"
            days_phrase = f"Over {days_apart} {day_word}, "
        else:
            days_phrase = ""
        headline = f"{days_phrase}{magnitude} {move} with {income_desc}."

    highlights = []
    if mv_delta is not None:
        if days_apart:
            day_word = "day" if days_apart == 1 else "days"
            days_tag = f" vs {days_apart} {day_word} ago"
        else:
            days_tag = ""
        highlights.append(f"Market value {_format_money(mv_delta)} ({_format_pct(mv_pct)}){days_tag}")
    if income_delta is not None:
        highlights.append(f"Forward 12m income {_format_money(income_delta)}")
    if period_type in (None, "daily"):
        twr_1m = (right.get("portfolio_rollups") or {}).get("performance", {}).get("twr_1m_pct")
        if _is_number(twr_1m):
            highlights.append(f"1M TWR {_format_pct(twr_1m)}")
    added = len(holdings_diff.get("added") or [])
    removed = len(holdings_diff.get("removed") or [])
    if added == 0 and removed == 0:
        highlights.append("No holdings added or removed")
    else:
        highlights.append(f"Holdings added {added}, removed {removed}")

    if period_type and period_type != "daily":
        perf = (right.get("portfolio_rollups") or {}).get("performance", {})
        risk = (right.get("portfolio_rollups") or {}).get("risk", {})
        if period_type == "weekly":
            per_day = (range_metrics or {}).get("per_day_return_pct")
            if _is_number(per_day):
                highlights.append(f"Per-day return {_format_pct(per_day)}")
            vol_30d = risk.get("vol_30d_pct")
            if _is_number(vol_30d):
                highlights.append(f"Vol 30d {_format_pct(vol_30d)}")
        elif period_type == "monthly":
            annualized = (range_metrics or {}).get("annualized_return_pct")
            if _is_number(annualized):
                highlights.append(f"Annualized return {_format_pct(annualized)}")
            vol_90d = risk.get("vol_90d_pct")
            if _is_number(vol_90d):
                highlights.append(f"Vol 90d {_format_pct(vol_90d)}")
        elif period_type == "quarterly":
            annualized = (range_metrics or {}).get("annualized_return_pct")
            if _is_number(annualized):
                highlights.append(f"Annualized return {_format_pct(annualized)}")
            sharpe = risk.get("sharpe_1y")
            sortino = risk.get("sortino_1y")
            calmar = risk.get("calmar_1y")
            if _is_number(sharpe):
                highlights.append(f"Sharpe 1y {_format_number(sharpe)}")
            if _is_number(sortino):
                highlights.append(f"Sortino 1y {_format_number(sortino)}")
            if _is_number(calmar):
                highlights.append(f"Calmar 1y {_format_number(calmar)}")
        elif period_type == "yearly":
            twr_12m = perf.get("twr_12m_pct")
            if _is_number(twr_12m):
                highlights.append(f"TWR 12m {_format_pct(twr_12m)}")
            sharpe = risk.get("sharpe_1y")
            sortino = risk.get("sortino_1y")
            calmar = risk.get("calmar_1y")
            if _is_number(sharpe):
                highlights.append(f"Sharpe 1y {_format_number(sharpe)}")
            if _is_number(sortino):
                highlights.append(f"Sortino 1y {_format_number(sortino)}")
            if _is_number(calmar):
                highlights.append(f"Calmar 1y {_format_number(calmar)}")

    return {
        "direction": direction,
        "primary_driver": primary_driver,
        "headline": headline,
        "highlights": highlights,
    }


def build_daily_diff(
    left: dict,
    right: dict,
    left_date: str,
    right_date: str,
    left_ref: str | None = None,
    right_ref: str | None = None,
    period_type: str = "daily",
    calendar_aligned: bool | None = None,
    period_start: str | None = None,
    period_end: str | None = None,
    dividends_period_totals: tuple[float | None, float | None] | None = None,
    dividends_mtd_totals: tuple[float | None, float | None] | None = None,
):
    """Builds a diff payload. For period diffs, pass period_type plus period bounds and dividend totals."""
    totals = _section_diff(
        left.get("totals", {}),
        right.get("totals", {}),
        ["market_value", "net_liquidation_value", "cost_basis", "unrealized_pnl", "unrealized_pct", "margin_loan_balance", "margin_to_portfolio_pct"],
    )
    income = _section_diff(
        left.get("income", {}),
        right.get("income", {}),
        ["projected_monthly_income", "forward_12m_total", "portfolio_current_yield_pct", "portfolio_yield_on_cost_pct"],
    )
    rollup_perf = _section_diff(
        left.get("portfolio_rollups", {}).get("performance", {}),
        right.get("portfolio_rollups", {}).get("performance", {}),
        ["twr_1m_pct", "twr_3m_pct", "twr_6m_pct", "twr_12m_pct"],
    )
    rollup_risk = _section_diff(
        left.get("portfolio_rollups", {}).get("risk", {}),
        right.get("portfolio_rollups", {}).get("risk", {}),
        [
            "vol_30d_pct",
            "vol_90d_pct",
            "sharpe_1y",
            "sortino_1y",
            "sortino_6m",
            "sortino_3m",
            "sortino_1m",
            "sortino_sharpe_ratio",
            "sortino_sharpe_divergence",
            "information_ratio_1y",
            "tracking_error_1y_pct",
            "ulcer_index_1y",
            "omega_ratio_1y",
            "pain_adjusted_return",
            "income_stability_score",
            "var_90_1d_pct",
            "calmar_1y",
            "max_drawdown_1y_pct",
            "var_95_1d_pct",
            "var_99_1d_pct",
            "var_95_1w_pct",
            "var_95_1m_pct",
            "cvar_90_1d_pct",
            "cvar_95_1d_pct",
            "cvar_99_1d_pct",
            "cvar_95_1w_pct",
            "cvar_95_1m_pct",
        ],
    )

    rollup_income_stability = _numeric_tree(
        left.get("portfolio_rollups", {}).get("income_stability", {}),
        right.get("portfolio_rollups", {}).get("income_stability", {}),
    )
    rollup_income_growth = _numeric_tree(
        left.get("portfolio_rollups", {}).get("income_growth", {}),
        right.get("portfolio_rollups", {}).get("income_growth", {}),
    )
    rollup_tail_risk = _numeric_tree(
        left.get("portfolio_rollups", {}).get("tail_risk", {}),
        right.get("portfolio_rollups", {}).get("tail_risk", {}),
    )
    rollup_vs_benchmark = _numeric_tree(
        left.get("portfolio_rollups", {}).get("vs_benchmark", {}),
        right.get("portfolio_rollups", {}).get("vs_benchmark", {}),
    )
    rollup_return_attribution_1m = _numeric_tree(
        left.get("portfolio_rollups", {}).get("return_attribution_1m", {}),
        right.get("portfolio_rollups", {}).get("return_attribution_1m", {}),
    )
    rollup_return_attribution_3m = _numeric_tree(
        left.get("portfolio_rollups", {}).get("return_attribution_3m", {}),
        right.get("portfolio_rollups", {}).get("return_attribution_3m", {}),
    )
    rollup_return_attribution_6m = _numeric_tree(
        left.get("portfolio_rollups", {}).get("return_attribution_6m", {}),
        right.get("portfolio_rollups", {}).get("return_attribution_6m", {}),
    )
    rollup_return_attribution_12m = _numeric_tree(
        left.get("portfolio_rollups", {}).get("return_attribution_12m", {}),
        right.get("portfolio_rollups", {}).get("return_attribution_12m", {}),
    )

    goal_progress = _section_diff(
        left.get("goal_progress", {}) or {},
        right.get("goal_progress", {}) or {},
        ["progress_pct", "current_projected_monthly", "months_to_goal"],
    )
    goal_progress_net = _section_diff(
        left.get("goal_progress_net", {}) or {},
        right.get("goal_progress_net", {}) or {},
        ["progress_pct", "current_projected_monthly_net"],
    )

    margin_left = {
        "margin_loan_balance": (left.get("totals") or {}).get("margin_loan_balance"),
        "margin_to_portfolio_pct": (left.get("totals") or {}).get("margin_to_portfolio_pct"),
        "ltv_pct": (left.get("totals") or {}).get("margin_to_portfolio_pct"),
    }
    margin_right = {
        "margin_loan_balance": (right.get("totals") or {}).get("margin_loan_balance"),
        "margin_to_portfolio_pct": (right.get("totals") or {}).get("margin_to_portfolio_pct"),
        "ltv_pct": (right.get("totals") or {}).get("margin_to_portfolio_pct"),
    }
    margin = _section_diff(margin_left, margin_right, ["margin_loan_balance", "margin_to_portfolio_pct", "ltv_pct"])
    margin_stress = _numeric_tree(left.get("margin_stress", {}), right.get("margin_stress", {}))

    dividends_compact = _section_diff(
        {
            "forward_12m_total": (left.get("income") or {}).get("forward_12m_total"),
            "realized_mtd_total": (left.get("dividends") or {}).get("realized_mtd", {}).get("total_dividends"),
            "next_30d_total": (left.get("dividends_upcoming") or {}).get("projected"),
        },
        {
            "forward_12m_total": (right.get("income") or {}).get("forward_12m_total"),
            "realized_mtd_total": (right.get("dividends") or {}).get("realized_mtd", {}).get("total_dividends"),
            "next_30d_total": (right.get("dividends_upcoming") or {}).get("projected"),
        },
        ["forward_12m_total", "realized_mtd_total", "next_30d_total"],
    )

    coverage_left = left.get("coverage") or {}
    coverage_right = right.get("coverage") or {}
    coverage = _section_diff(coverage_left, coverage_right, ["derived_pct", "missing_pct"])
    note = "Data quality unchanged."
    if coverage.get("derived_pct", {}).get("delta") is not None or coverage.get("missing_pct", {}).get("delta") is not None:
        derived_delta = coverage.get("derived_pct", {}).get("delta") or 0.0
        missing_delta = coverage.get("missing_pct", {}).get("delta") or 0.0
        if derived_delta > 0 and missing_delta < 0:
            note = "Data quality slightly improved (fewer missing fields)."
        elif derived_delta < 0 and missing_delta > 0:
            note = "Data quality worsened (more missing fields)."
    coverage["note"] = note

    min_weight_delta = None
    min_mv_delta = None
    if period_type in ("quarterly", "yearly"):
        min_weight_delta = HOLDINGS_CHANGE_MIN_WEIGHT_PCT
        min_mv_delta = HOLDINGS_CHANGE_MIN_MARKET_VALUE
    holdings_diff = _holdings_diff(left, right, min_weight_delta=min_weight_delta, min_mv_delta=min_mv_delta)

    days_apart = None
    try:
        days_apart = abs((date.fromisoformat(right_date) - date.fromisoformat(left_date)).days)
    except Exception:
        days_apart = None
    comparison = {
        "left_date": left_date,
        "right_date": right_date,
        "days_apart": days_apart,
        "mode": "adjacent" if days_apart == 1 else "non_adjacent",
    }

    range_metrics = None
    if days_apart and days_apart > 0:
        left_mv = (left.get("totals") or {}).get("market_value")
        right_mv = (right.get("totals") or {}).get("market_value")
        if _is_number(left_mv) and _is_number(right_mv) and left_mv:
            simple_pnl = right_mv - left_mv
            simple_return_pct = (right_mv / left_mv - 1.0) * 100
            range_metrics = {
                "simple_return_pct": round(simple_return_pct, 2),
                "simple_pnl": round(simple_pnl, 2),
            }
            if period_type in ("monthly", "quarterly", "yearly"):
                range_metrics["annualized_return_pct"] = _annualized_return_pct(simple_return_pct, days_apart)
                range_metrics["per_day_return_pct"] = None
            else:
                range_metrics["per_day_return_pct"] = round(simple_return_pct / days_apart, 2)

    include_period_fields = period_type in ("weekly", "monthly", "quarterly", "yearly")
    if include_period_fields:
        comparison["period_type"] = period_type
        comparison["period_start"] = period_start or left_date
        comparison["period_end"] = period_end or right_date
        comparison["calendar_aligned"] = bool(calendar_aligned)

    realized_period_left = None
    realized_period_right = None
    if include_period_fields and dividends_period_totals:
        realized_period_left, realized_period_right = dividends_period_totals

    realized_mtd_left = None
    realized_mtd_right = None
    include_realized_mtd = bool(period_type == "monthly" and calendar_aligned)
    if include_realized_mtd:
        if dividends_mtd_totals:
            realized_mtd_left, realized_mtd_right = dividends_mtd_totals
        else:
            realized_mtd_left = (left.get("dividends") or {}).get("realized_mtd", {}).get("total_dividends")
            realized_mtd_right = (right.get("dividends") or {}).get("realized_mtd", {}).get("total_dividends")

    if include_period_fields:
        dividends_compact = _section_diff(
            {
                "forward_12m_total": (left.get("income") or {}).get("forward_12m_total"),
                "realized_period_total": realized_period_left,
                "realized_mtd_total": realized_mtd_left if include_realized_mtd else None,
                "next_30d_total": (left.get("dividends_upcoming") or {}).get("projected"),
            },
            {
                "forward_12m_total": (right.get("income") or {}).get("forward_12m_total"),
                "realized_period_total": realized_period_right,
                "realized_mtd_total": realized_mtd_right if include_realized_mtd else None,
                "next_30d_total": (right.get("dividends_upcoming") or {}).get("projected"),
            },
            ["forward_12m_total", "realized_period_total", "realized_mtd_total", "next_30d_total"],
        )

    diff = {
        "comparison": comparison,
        "left": {"date": left_date},
        "right": {"date": right_date},
        "summary": _summary_block(left, right, holdings_diff, days_apart, period_type=period_type, range_metrics=range_metrics),
        "range_metrics": range_metrics,
        "portfolio_metrics": {
            "totals": totals,
            "income": income,
            "rollups": {
                "performance": rollup_perf,
                "risk": rollup_risk,
                "income_stability": rollup_income_stability,
                "income_growth": rollup_income_growth,
                "tail_risk": rollup_tail_risk,
                "vs_benchmark": rollup_vs_benchmark,
                "return_attribution_1m": rollup_return_attribution_1m,
                "return_attribution_3m": rollup_return_attribution_3m,
                "return_attribution_6m": rollup_return_attribution_6m,
                "return_attribution_12m": rollup_return_attribution_12m,
            },
            "goal_progress": goal_progress,
            "goal_progress_net": goal_progress_net,
            "margin": margin,
            "margin_stress": margin_stress,
        },
        "dividends": dividends_compact,
        "coverage": coverage,
        "macro": {
            "snapshot": _numeric_tree(left.get("macro", {}).get("snapshot", {}), right.get("macro", {}).get("snapshot", {})),
            "trends": _numeric_tree(left.get("macro", {}).get("trends", {}), right.get("macro", {}).get("trends", {})),
        },
        "holdings": holdings_diff,
    }
    diff["summary"]["period_type"] = period_type

    if left_ref:
        diff["left"]["file"] = left_ref
    if right_ref:
        diff["right"]["file"] = right_ref

    validate_diff_payload(diff, left, right)
    return diff


def diff_daily_from_db(conn: sqlite3.Connection, left_date: str, right_date: str):
    left = _load_daily_snapshot(conn, left_date)
    right = _load_daily_snapshot(conn, right_date)
    if not left or not right:
        missing = []
        if not left:
            missing.append(left_date)
        if not right:
            missing.append(right_date)
        raise ValueError(f"missing daily snapshot(s): {', '.join(missing)}")
    return build_daily_diff(left, right, left_date, right_date)


def validate_diff_payload(diff: dict, left: dict, right: dict):
    totals = (diff.get("portfolio_metrics") or {}).get("totals") or {}
    range_metrics = diff.get("range_metrics") or {}
    mv_delta = (totals.get("market_value") or {}).get("delta")
    simple_pnl = range_metrics.get("simple_pnl")
    if _is_number(mv_delta) and _is_number(simple_pnl):
        if abs(mv_delta - simple_pnl) > _DIFF_TOL_MONEY:
            raise ValueError("diff_invariant_failed: market_value_delta_vs_simple_pnl")

    margin_delta = (totals.get("margin_loan_balance") or {}).get("delta")
    net_delta = (totals.get("net_liquidation_value") or {}).get("delta")
    if _is_number(margin_delta) and abs(margin_delta) <= _DIFF_TOL_MONEY:
        if _is_number(net_delta) and _is_number(mv_delta):
            if abs(net_delta - mv_delta) > _DIFF_TOL_MONEY:
                raise ValueError("diff_invariant_failed: net_liquidation_delta_vs_market_value_delta")

    for side, snap in (("left", left), ("right", right)):
        holdings = snap.get("holdings") or []
        if not holdings:
            continue
        weights = []
        missing_weights = False
        for h in holdings:
            val = h.get("weight_pct")
            if not _is_number(val):
                missing_weights = True
                break
            weights.append(float(val))
        if missing_weights or not weights:
            continue
        total_weight = sum(weights)
        if abs(total_weight - 100.0) > _DIFF_TOL_PCT:
            raise ValueError(f"diff_invariant_failed: weight_sum_{side}")

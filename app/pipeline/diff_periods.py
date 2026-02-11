from __future__ import annotations

import calendar
import sqlite3
from datetime import date

from .diff_daily import build_daily_diff, _rollups
from .periods import _period_bounds
from .snapshot_views import assemble_period_snapshot
from ..config import settings

_PERIOD_MAP = {
    "weekly": "WEEK",
    "monthly": "MONTH",
    "quarterly": "QUARTER",
    "yearly": "YEAR",
}


def _load_period_snapshot(conn: sqlite3.Connection, period_type: str, start_date: str, end_date: str):
    """Load period snapshot from flat tables (assembled to V5 period shape)."""
    return assemble_period_snapshot(conn, period_type, end_date, period_start_date=start_date, rolling=False)


def _last_interval_holdings(period_snap: dict):
    intervals = period_snap.get("intervals") or []
    if not intervals:
        return []
    period = period_snap.get("period") or {}
    end_date = period.get("end_date")
    if end_date:
        for interval in intervals:
            if interval.get("end_date") == end_date and interval.get("holdings"):
                return interval.get("holdings") or []
    return (intervals[-1].get("holdings") or []) if intervals else []


def _last_interval_totals(period_snap: dict):
    intervals = period_snap.get("intervals") or []
    if not intervals:
        return {}
    period = period_snap.get("period") or {}
    end_date = period.get("end_date")
    if end_date:
        for interval in intervals:
            if interval.get("end_date") == end_date and interval.get("totals"):
                return interval.get("totals") or {}
    return (intervals[-1].get("totals") or {}) if intervals else {}


def _last_interval_margin(period_snap: dict):
    intervals = period_snap.get("intervals") or []
    if not intervals:
        return {}
    period = period_snap.get("period") or {}
    end_date = period.get("end_date")
    if end_date:
        for interval in intervals:
            if interval.get("end_date") == end_date and interval.get("margin") is not None:
                return interval.get("margin") or {}
    return (intervals[-1].get("margin") or {}) if intervals else {}


def _last_interval_daily(period_snap: dict):
    intervals = period_snap.get("intervals") or []
    if not intervals:
        return period_snap.get("daily_snapshot") or {}
    period = period_snap.get("period") or {}
    end_date = period.get("end_date")
    if end_date:
        for interval in intervals:
            if interval.get("end_date") == end_date and interval.get("daily_snapshot"):
                return interval.get("daily_snapshot") or {}
    for interval in reversed(intervals):
        if interval.get("daily_snapshot"):
            return interval.get("daily_snapshot") or {}
    return period_snap.get("daily_snapshot") or {}


def _period_to_daily_like(period_snap: dict):
    period = period_snap.get("period") or {}
    end_date = period.get("end_date") or period_snap.get("as_of")
    summary = period_snap.get("period_summary") or {}
    interval_totals = _last_interval_totals(period_snap)
    interval_margin = _last_interval_margin(period_snap)
    interval_daily = _last_interval_daily(period_snap)

    totals_end = (summary.get("totals") or {}).get("end") or {}
    margin_end = (summary.get("margin") or {}).get("end") or interval_margin or {}
    income_end = (summary.get("income") or {}).get("end") or {}
    risk_end = (summary.get("risk") or {}).get("end") or {}
    goal_end = (summary.get("goal_progress") or {}).get("end") or {}
    goal_net_end = (summary.get("goal_progress_net") or {}).get("end") or {}
    dividends_end = (summary.get("dividends") or {}).get("end") or {}

    twr_windows = (summary.get("performance") or {}).get("twr_windows") or {}
    macro_end = (summary.get("macro") or {}).get("end") or {}
    macro_avg = (summary.get("macro") or {}).get("avg") or {}

    totals = {
        "market_value": totals_end.get("total_market_value") or totals_end.get("market_value"),
        "net_liquidation_value": totals_end.get("net_liquidation_value") or interval_totals.get("net_liquidation_value"),
        "cost_basis": totals_end.get("cost_basis") or interval_totals.get("cost_basis"),
        "unrealized_pnl": totals_end.get("unrealized_pnl"),
        "unrealized_pct": totals_end.get("unrealized_pct"),
        "margin_loan_balance": margin_end.get("margin_loan_balance"),
        "margin_to_portfolio_pct": margin_end.get("margin_to_portfolio_pct") or margin_end.get("ltv_pct"),
    }

    rollups_daily = _rollups(interval_daily) if interval_daily else {}
    portfolio_rollups = {
        "performance": {
            "twr_1m_pct": twr_windows.get("twr_1m_pct_end"),
            "twr_3m_pct": twr_windows.get("twr_3m_pct_end"),
            "twr_6m_pct": twr_windows.get("twr_6m_pct_end"),
            "twr_12m_pct": twr_windows.get("twr_12m_pct_end"),
        },
        "risk": {
            "vol_30d_pct": risk_end.get("vol_30d_pct"),
            "vol_90d_pct": risk_end.get("vol_90d_pct"),
            "sharpe_1y": risk_end.get("sharpe_1y"),
            "sortino_1y": risk_end.get("sortino_1y"),
            "sortino_6m": risk_end.get("sortino_6m"),
            "sortino_3m": risk_end.get("sortino_3m"),
            "sortino_1m": risk_end.get("sortino_1m"),
            "sortino_sharpe_ratio": risk_end.get("sortino_sharpe_ratio"),
            "sortino_sharpe_divergence": risk_end.get("sortino_sharpe_divergence"),
            "calmar_1y": risk_end.get("calmar_1y"),
            "max_drawdown_1y_pct": risk_end.get("max_drawdown_1y_pct"),
            "information_ratio_1y": risk_end.get("information_ratio_1y"),
            "tracking_error_1y_pct": risk_end.get("tracking_error_1y_pct"),
            "ulcer_index_1y": risk_end.get("ulcer_index_1y"),
            "omega_ratio_1y": risk_end.get("omega_ratio_1y"),
            "pain_adjusted_return": risk_end.get("pain_adjusted_return"),
            "income_stability_score": risk_end.get("income_stability_score"),
            "var_90_1d_pct": risk_end.get("var_90_1d_pct"),
            "var_95_1d_pct": risk_end.get("var_95_1d_pct"),
            "var_99_1d_pct": risk_end.get("var_99_1d_pct"),
            "var_95_1w_pct": risk_end.get("var_95_1w_pct"),
            "var_95_1m_pct": risk_end.get("var_95_1m_pct"),
            "cvar_90_1d_pct": risk_end.get("cvar_90_1d_pct"),
            "cvar_95_1d_pct": risk_end.get("cvar_95_1d_pct"),
            "cvar_99_1d_pct": risk_end.get("cvar_99_1d_pct"),
            "cvar_95_1w_pct": risk_end.get("cvar_95_1w_pct"),
            "cvar_95_1m_pct": risk_end.get("cvar_95_1m_pct"),
        },
        "income_stability": rollups_daily.get("income_stability"),
        "income_growth": rollups_daily.get("income_growth"),
        "tail_risk": rollups_daily.get("tail_risk"),
        "vs_benchmark": rollups_daily.get("vs_benchmark"),
        "return_attribution_1m": rollups_daily.get("return_attribution_1m"),
        "return_attribution_3m": rollups_daily.get("return_attribution_3m"),
        "return_attribution_6m": rollups_daily.get("return_attribution_6m"),
        "return_attribution_12m": rollups_daily.get("return_attribution_12m"),
    }

    realized_mtd_total = None
    next_30d_total = None
    if interval_daily:
        realized_mtd_total = (interval_daily.get("dividends") or {}).get("realized_mtd", {}).get("total_dividends")
        next_30d_total = (interval_daily.get("dividends_upcoming") or {}).get("projected")
    if realized_mtd_total is None:
        realized_mtd_total = dividends_end.get("period_dividends_received_ytd")
    if next_30d_total is None:
        next_30d_total = dividends_end.get("dividends_upcoming_30d")

    dividends = {"realized_mtd": {"total_dividends": realized_mtd_total}}
    dividends_upcoming = {"projected": next_30d_total}

    summary_block = period_snap.get("summary") or {}
    missing_pct = None
    coverage_pct = period.get("coverage_pct")
    if isinstance(coverage_pct, (int, float)):
        missing_pct = round(100.0 - float(coverage_pct), 2)
    else:
        expected = period.get("expected_days")
        observed = period.get("observed_days")
        if isinstance(expected, (int, float)) and isinstance(observed, (int, float)) and expected:
            missing_pct = round((1.0 - (float(observed) / float(expected))) * 100, 2)
    coverage = {
        "derived_pct": summary_block.get("derived_pct"),
        "pulled_pct": summary_block.get("pulled_pct"),
        "missing_pct": missing_pct,
    }

    return {
        "as_of": end_date,
        "totals": totals,
        "income": income_end,
        "portfolio_rollups": portfolio_rollups,
        "goal_progress": goal_end,
        "goal_progress_net": goal_net_end,
        "dividends": dividends,
        "dividends_upcoming": dividends_upcoming,
        "margin_stress": (interval_daily or {}).get("margin_stress"),
        "coverage": coverage,
        "macro": {"snapshot": macro_end, "trends": macro_avg},
        "holdings": _last_interval_holdings(period_snap),
    }


def _calendar_aligned(period_type: str, start_date: date, end_date: date, weekly_aligned: bool) -> bool:
    if period_type == "weekly":
        return bool(weekly_aligned)
    if period_type == "monthly":
        if start_date.day != 1:
            return False
        last_day = calendar.monthrange(start_date.year, start_date.month)[1]
        return end_date.year == start_date.year and end_date.month == start_date.month and end_date.day == last_day
    if period_type == "quarterly":
        if start_date.day != 1:
            return False
        q_start_month = ((start_date.month - 1) // 3) * 3 + 1
        if start_date.month != q_start_month:
            return False
        q_end_month = q_start_month + 2
        last_day = calendar.monthrange(start_date.year, q_end_month)[1]
        return end_date.year == start_date.year and end_date.month == q_end_month and end_date.day == last_day
    if period_type == "yearly":
        return start_date.month == 1 and start_date.day == 1 and end_date.month == 12 and end_date.day == 31 and end_date.year == start_date.year
    return False


def _sum_dividends_for_period(conn: sqlite3.Connection, start_date: date, end_date: date) -> float | None:
    if not start_date or not end_date:
        return None
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT amount
        FROM investment_transactions
        WHERE transaction_type='dividend' AND date BETWEEN ? AND ?
        """,
        (start_date.isoformat(), end_date.isoformat()),
    ).fetchall()
    total = 0.0
    seen = False
    for (amount,) in rows:
        try:
            total += abs(float(amount))
            seen = True
        except (TypeError, ValueError):
            continue
    return round(total, 2) if seen else None


def diff_periods_from_db(conn: sqlite3.Connection, snapshot_type: str, left_as_of: str, right_as_of: str):
    if snapshot_type not in _PERIOD_MAP:
        raise ValueError("snapshot_type must be weekly|monthly|quarterly|yearly")
    try:
        left_date = date.fromisoformat(left_as_of)
        right_date = date.fromisoformat(right_as_of)
    except ValueError:
        raise ValueError("as_of must be YYYY-MM-DD")

    left_start, left_end = _period_bounds(snapshot_type, left_date)
    right_start, right_end = _period_bounds(snapshot_type, right_date)
    period_type = _PERIOD_MAP[snapshot_type]

    left_snap = _load_period_snapshot(conn, period_type, left_start.isoformat(), left_end.isoformat())
    right_snap = _load_period_snapshot(conn, period_type, right_start.isoformat(), right_end.isoformat())
    if not left_snap or not right_snap:
        missing = []
        if not left_snap:
            missing.append(left_end.isoformat())
        if not right_snap:
            missing.append(right_end.isoformat())
        raise ValueError(f"missing period snapshot(s): {', '.join(missing)}")

    left_like = _period_to_daily_like(left_snap)
    right_like = _period_to_daily_like(right_snap)

    left_period = left_snap.get("period") or {}
    right_period = right_snap.get("period") or {}
    left_start = date.fromisoformat(left_period.get("start_date")) if left_period.get("start_date") else left_start
    left_end_date = date.fromisoformat(left_period.get("end_date")) if left_period.get("end_date") else left_end
    right_start = date.fromisoformat(right_period.get("start_date")) if right_period.get("start_date") else right_start
    right_end_date = date.fromisoformat(right_period.get("end_date")) if right_period.get("end_date") else right_end

    realized_left = _sum_dividends_for_period(conn, left_start, left_end_date)
    realized_right = _sum_dividends_for_period(conn, right_start, right_end_date)
    mtd_left = (left_like.get("dividends") or {}).get("realized_mtd", {}).get("total_dividends")
    mtd_right = (right_like.get("dividends") or {}).get("realized_mtd", {}).get("total_dividends")

    weekly_aligned = bool(getattr(settings, "weekly_calendar_aligned", False))
    calendar_aligned = _calendar_aligned(snapshot_type, left_end, right_end, weekly_aligned)

    diff = build_daily_diff(
        left_like,
        right_like,
        left_end.isoformat(),
        right_end.isoformat(),
        period_type=snapshot_type,
        calendar_aligned=calendar_aligned,
        period_start=left_end.isoformat(),
        period_end=right_end.isoformat(),
        dividends_period_totals=(realized_left, realized_right),
        dividends_mtd_totals=(mtd_left, mtd_right),
    )
    diff["summary"]["period_type"] = snapshot_type
    return diff

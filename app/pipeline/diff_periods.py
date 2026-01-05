from __future__ import annotations

import json
import sqlite3
from datetime import date

from .diff_daily import build_daily_diff
from .periods import _period_bounds

_PERIOD_MAP = {
    "weekly": "WEEK",
    "monthly": "MONTH",
    "quarterly": "QUARTER",
    "yearly": "YEAR",
}


def _load_period_snapshot(conn: sqlite3.Connection, period_type: str, start_date: str, end_date: str):
    cur = conn.cursor()
    row = cur.execute(
        "SELECT payload_json FROM snapshots WHERE period_type=? AND period_start_date=? AND period_end_date=?",
        (period_type, start_date, end_date),
    ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


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


def _period_to_daily_like(period_snap: dict):
    period = period_snap.get("period") or {}
    end_date = period.get("end_date") or period_snap.get("as_of")
    summary = period_snap.get("period_summary") or {}
    interval_totals = _last_interval_totals(period_snap)
    interval_margin = _last_interval_margin(period_snap)

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
            "calmar_1y": risk_end.get("calmar_1y"),
            "max_drawdown_1y_pct": risk_end.get("max_drawdown_1y_pct"),
        },
    }

    dividends = {
        "realized_mtd": {
            "total_dividends": dividends_end.get("period_dividends_received_ytd"),
        }
    }
    dividends_upcoming = {
        "projected": dividends_end.get("dividends_upcoming_30d"),
    }

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
        "coverage": coverage,
        "macro": {"snapshot": macro_end, "trends": macro_avg},
        "holdings": _last_interval_holdings(period_snap),
    }


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
    diff = build_daily_diff(left_like, right_like, left_end.isoformat(), right_end.isoformat())
    diff["summary"]["period_type"] = snapshot_type
    return diff

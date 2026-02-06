from __future__ import annotations

import json
import sqlite3
import statistics
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

from .snapshots import (
    DIVIDEND_CUT_THRESHOLD,
    _build_pay_history,
    _load_dividend_transactions,
    _load_first_acquired_dates,
    _load_provider_dividends,
    _month_keys,
    _months_between,
    _monthly_income_totals,
)


_V3_INTERVAL_FIELDS = {
    "income_stability",
    "income_growth",
    "tail_risk",
    "vs_benchmark",
    "margin_stress",
    "return_attribution_1m",
    "return_attribution_3m",
    "return_attribution_6m",
    "return_attribution_12m",
}

_V3_RISK_KEYS = {
    "information_ratio_1y",
    "tracking_error_1y_pct",
    "ulcer_index_1y",
    "omega_ratio_1y",
    "pain_adjusted_return",
    "income_stability_score",
    "var_90_1d_pct",
    "var_99_1d_pct",
    "var_95_1w_pct",
    "var_95_1m_pct",
    "cvar_90_1d_pct",
    "cvar_99_1d_pct",
    "cvar_95_1w_pct",
    "cvar_95_1m_pct",
    "portfolio_risk_quality",
}

_V3_TOKENS = _V3_INTERVAL_FIELDS | _V3_RISK_KEYS | {
    "dividend_reliability",
    "income_stability",
    "income_growth",
    "tail_risk",
    "vs_benchmark",
}


def replace_nulls_with_reasons(payload: dict, *, kind: str, conn: sqlite3.Connection | None = None) -> dict:
    context = _NullReasonContext(payload, kind, conn)
    return _replace_nulls(payload, [], payload, context)


class _NullReasonContext:
    def __init__(self, payload: dict, kind: str, conn: sqlite3.Connection | None):
        self.payload = payload
        self.kind = kind
        self.conn = conn
        self.schema_by_date = _load_schema_versions(conn)
        self.interval_end_dates = {}
        self.period_start_date = None
        self.left_date = None
        self.right_date = None
        self.left_schema = None
        self.right_schema = None
        self.period_type = None

        if kind == "period":
            period = payload.get("period") or {}
            self.period_start_date = period.get("start_date")
            intervals = payload.get("intervals") or []
            self.interval_end_dates = {idx: interval.get("end_date") for idx, interval in enumerate(intervals)}
        elif kind == "diff":
            comparison = payload.get("comparison") or {}
            self.left_date = comparison.get("left_date")
            self.right_date = comparison.get("right_date")
            self.period_type = comparison.get("period_type")
            self.left_schema = self.schema_by_date.get(self.left_date)
            self.right_schema = self.schema_by_date.get(self.right_date)

        self.dividend_reasons_by_symbol = {}
        self.portfolio_income_reasons = {}
        self.as_of_date = _as_of_date(payload, kind)
        self.symbols = _extract_symbols(payload, kind) if kind in ("daily", "period") else []
        if conn and self.as_of_date and self.symbols:
            self.dividend_reasons_by_symbol = _dividend_reliability_reasons(conn, self.symbols, self.as_of_date)
        if conn and self.as_of_date:
            self.portfolio_income_reasons = _portfolio_income_reasons(conn, self.as_of_date)


def _replace_nulls(obj: Any, path: list[Any], root: dict, context: _NullReasonContext) -> Any:
    if isinstance(obj, dict):
        return {key: _replace_nulls(val, path + [key], root, context) for key, val in obj.items()}
    if isinstance(obj, list):
        return [_replace_nulls(val, path + [idx], root, context) for idx, val in enumerate(obj)]
    if obj is None:
        return _reason_for_null(path, root, context)
    if _is_zero(obj):
        reason = _reason_for_zero(path, root, context)
        if reason:
            return reason
    return obj


def _is_zero(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and float(val) == 0.0


def _reason_for_null(path: list[Any], root: dict, context: _NullReasonContext) -> str:
    # V5 renamed dividend_reliability → reliability
    if "dividend_reliability" in path or "reliability" in path:
        symbol = _find_holding_symbol(root, path)
        field = path[-1] if path else None
        if symbol and field:
            reason = context.dividend_reasons_by_symbol.get(symbol, {}).get(field)
            if reason:
                return reason
            # Fallback for reliability fields with no specific reason
            if field in ("last_increase_date", "last_decrease_date"):
                return f"not computed for {symbol} — insufficient data or no qualifying event"
            if "growth" in field or "volatility" in field:
                return f"not computed for {symbol} — insufficient dividend history"

    if "income_growth" in path or "income_stability" in path:
        field = path[-1] if path else None
        if field in context.portfolio_income_reasons:
            return context.portfolio_income_reasons[field]

    if context.kind == "period":
        if len(path) >= 3 and path[0] == "intervals" and isinstance(path[1], int):
            idx = path[1]
            end_date = context.interval_end_dates.get(idx)
            schema = context.schema_by_date.get(end_date)
            if path[2] in _V3_INTERVAL_FIELDS and _is_pre_v3(schema):
                return f"daily snapshot {end_date} schema {schema or 'unknown'} missing v3 field"
            if path[2] == "risk" and len(path) >= 4 and path[3] in _V3_RISK_KEYS and _is_pre_v3(schema):
                return f"daily snapshot {end_date} schema {schema or 'unknown'} missing v3 metric"
        if len(path) >= 4 and path[:3] == ["period_summary", "risk", "start"]:
            schema = context.schema_by_date.get(context.period_start_date)
            if path[3] in _V3_RISK_KEYS and _is_pre_v3(schema):
                return f"period start snapshot {context.period_start_date} schema {schema or 'unknown'} missing v3 metric"

    if context.kind == "diff":
        if len(path) >= 2 and path[0] == "dividends" and path[1] == "realized_mtd_total" and context.period_type:
            return "period snapshots do not include realized_mtd totals"
        if path and path[-1] in ("left", "right", "delta"):
            if _path_contains_token(path, _V3_TOKENS):
                return _diff_reason(path[-1], context)
            return _diff_fallback_reason(path[-1], context)

    # V5: better fallbacks for common field groups
    if "income_stability" in path or "income_growth" in path:
        return "insufficient dividend history to compute metric"
    if "margin" in path:
        field = path[-1] if path else ""
        if "coverage" in str(field) or "ltv" in str(field):
            return "cannot compute — margin or income data missing"
        if "trend" in str(field):
            return "insufficient history for trend calculation"
        return "margin data not available for this date"
    if "goals" in path:
        return "goal data not available for this date"
    if "projected_vs_received" in path:
        return "insufficient income history for projection comparison"
    return "value not available in source data"


def _reason_for_zero(path: list[Any], root: dict, context: _NullReasonContext) -> str | None:
    if "dividend_reliability" in path or "reliability" in path:
        symbol = _find_holding_symbol(root, path)
        field = path[-1] if path else None
        if symbol and field:
            reason = context.dividend_reasons_by_symbol.get(symbol, {}).get(field)
            if reason:
                return reason

    if "income_growth" in path or "income_stability" in path:
        field = path[-1] if path else None
        if field in context.portfolio_income_reasons:
            return context.portfolio_income_reasons[field]

    return None


def _diff_reason(side: str, context: _NullReasonContext) -> str:
    if side == "left":
        if _is_pre_v3(context.left_schema):
            return f"left snapshot {context.left_date} schema {context.left_schema or 'unknown'} missing v3 metric"
        return "left value missing in source data"
    if side == "right":
        if _is_pre_v3(context.right_schema):
            return f"right snapshot {context.right_date} schema {context.right_schema or 'unknown'} missing v3 metric"
        return "right value missing in source data"
    if _is_pre_v3(context.left_schema) and _is_pre_v3(context.right_schema):
        return "delta requires left and right values; both missing (schema < 3.0)"
    if _is_pre_v3(context.left_schema):
        return "delta requires left value; left snapshot schema < 3.0 missing metric"
    if _is_pre_v3(context.right_schema):
        return "delta requires right value; right snapshot schema < 3.0 missing metric"
    return "delta requires left and right values; one side missing"


def _diff_fallback_reason(side: str, context: _NullReasonContext) -> str:
    if side == "left":
        return "left value missing in source data"
    if side == "right":
        return "right value missing in source data"
    return "delta requires left and right values; one side missing"


def _path_contains_token(path: list[Any], tokens: set[str]) -> bool:
    for part in path:
        if isinstance(part, str) and part in tokens:
            return True
    return False


def _load_schema_versions(conn: sqlite3.Connection | None) -> dict[str, str | None]:
    if conn is None:
        return {}
    cur = conn.cursor()
    rows = cur.execute("SELECT as_of_date_local, payload_json FROM snapshot_daily_current").fetchall()
    out = {}
    for as_of_date, payload in rows:
        schema_version = None
        try:
            meta = json.loads(payload).get("meta", {})
            schema_version = meta.get("schema_version")
        except Exception:
            schema_version = None
        out[str(as_of_date)] = schema_version
    return out


def _is_pre_v3(schema: str | None) -> bool:
    if not schema:
        return True
    try:
        return float(schema) < 3.0
    except (TypeError, ValueError):
        return True


def _as_of_date(payload: dict, kind: str) -> date | None:
    if kind == "daily":
        # V5: timestamps.portfolio_data_as_of_local
        ts = payload.get("timestamps") or {}
        v5_date = ts.get("portfolio_data_as_of_local")
        return _parse_date(v5_date or payload.get("as_of_date_local") or payload.get("as_of"))
    if kind == "period":
        period = payload.get("period") or {}
        return _parse_date(period.get("end_date") or payload.get("as_of"))
    if kind == "diff":
        comparison = payload.get("comparison") or {}
        return _parse_date(comparison.get("right_date"))
    return None


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _extract_symbols(payload: dict, kind: str) -> list[str]:
    holdings = payload.get("holdings") or []
    if not isinstance(holdings, list):
        holdings = []
    if not holdings and kind == "period":
        intervals = payload.get("intervals") or []
        for interval in reversed(intervals):
            daily = interval.get("daily_snapshot")
            if isinstance(daily, dict) and daily.get("holdings"):
                holdings = daily.get("holdings") or []
                break
        if not holdings and intervals:
            holdings = intervals[-1].get("holdings") or []
    symbols = []
    for holding in holdings:
        sym = holding.get("symbol") or holding.get("ticker")
        if sym:
            symbols.append(str(sym).upper())
    return list(dict.fromkeys(symbols))


def _find_holding_symbol(root: dict, path: list[Any]) -> str | None:
    for idx in range(len(path) - 1, 0, -1):
        if path[idx - 1] == "holdings" and isinstance(path[idx], int):
            holding = _get_value_at_path(root, path[: idx + 1])
            if isinstance(holding, dict):
                sym = holding.get("symbol") or holding.get("ticker")
                return str(sym).upper() if sym else None
    return None


def _get_value_at_path(root: dict, path: list[Any]) -> Any:
    current = root
    for part in path:
        if isinstance(part, int):
            if not isinstance(current, list) or part >= len(current):
                return None
            current = current[part]
        else:
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
    return current


def _dividend_reliability_reasons(
    conn: sqlite3.Connection, symbols: list[str], as_of_date: date
) -> dict[str, dict[str, str]]:
    div_tx = _load_dividend_transactions(conn)
    provider_divs = _load_provider_dividends(conn)
    pay_history = _build_pay_history(div_tx)
    acquired_dates = _load_first_acquired_dates(conn)
    cutoff = as_of_date - timedelta(days=365)

    out: dict[str, dict[str, str]] = {}
    for sym in symbols:
        reasons = {}
        first_acquired = acquired_dates.get(sym)
        window_start = cutoff
        if first_acquired and first_acquired <= as_of_date and first_acquired > cutoff:
            window_start = first_acquired
        window_months = max(1, _months_between(window_start, as_of_date))

        pay_events = [ev for ev in pay_history.get(sym, []) if ev.get("date") and ev.get("date") >= window_start]
        pay_dates = sorted({ev.get("date") for ev in pay_events if ev.get("date")})

        provider_events = provider_divs.get(sym, [])
        # Use full 365-day cutoff for provider ex-dates — the stock's dividend
        # schedule is independent of when the position was acquired.
        ex_dates = sorted(
            {
                ev.get("ex_date")
                for ev in provider_events
                if ev.get("ex_date") and cutoff <= ev.get("ex_date") <= as_of_date
            }
        )
        hist_dates = ex_dates or pay_dates
        hist_count = len(hist_dates)

        totals_12 = _monthly_income_totals(div_tx, as_of_date, window_months, symbol=sym)

        # Fall back to provider ex-date amounts when no actual received dividends
        if not any(t > 0 for t in totals_12) and provider_events:
            prov_by_month: dict[tuple[int, int], float] = defaultdict(float)
            for ev in provider_events:
                ex = ev.get("ex_date")
                amt = ev.get("amount")
                if ex and isinstance(amt, (int, float)) and cutoff <= ex <= as_of_date:
                    prov_by_month[(ex.year, ex.month)] += float(amt)
            if prov_by_month:
                totals_12 = [prov_by_month.get(key, 0.0) for key in _month_keys(as_of_date, 12)]

        totals_6 = totals_12[-6:] if len(totals_12) >= 6 else totals_12
        start_6 = totals_6[0] if totals_6 else 0.0
        end_6 = totals_6[-1] if totals_6 else 0.0
        mean_6 = statistics.mean(totals_6) if totals_6 else 0.0

        if hist_count < 2:
            if ex_dates:
                base = f"insufficient dividend history (<2 ex-dates since {window_start.isoformat()}; found {hist_count})"
            else:
                base = f"insufficient dividend payment history (<2 dates since {window_start.isoformat()}; found {hist_count})"
            reasons["payment_frequency_actual"] = base
            reasons["payment_frequency_expected"] = base
            reasons["avg_days_between_payments"] = base
            reasons["payment_timing_consistency"] = base
            reasons["dividend_cuts_12m"] = base
            reasons["missed_payments_12m"] = base
        elif not pay_dates:
            reasons["missed_payments_12m"] = f"no dividend payments recorded since {window_start.isoformat()}"

        if start_6 <= 0 or end_6 <= 0:
            reasons["dividend_growth_rate_6m_pct"] = (
                f"requires positive dividends at start/end of 6m window (start={start_6:.2f}, end={end_6:.2f})"
            )

        if mean_6 <= 0:
            reasons["dividend_volatility_6m_pct"] = "mean dividends over last 6m is 0; volatility undefined"

        events = []
        for ev in provider_divs.get(sym, []):
            ex_date = ev.get("ex_date")
            amt = ev.get("amount")
            if ex_date and isinstance(amt, (int, float)) and ex_date >= cutoff:
                events.append((ex_date, float(amt)))
        events.sort(key=lambda item: item[0])
        if not events and pay_events:
            for ev in pay_events:
                dt = ev.get("date")
                amt = ev.get("amount")
                if dt and isinstance(amt, (int, float)) and dt >= window_start:
                    events.append((dt, float(amt)))
            events.sort(key=lambda item: item[0])

        has_increase = False
        has_decrease = False
        for prev, curr in zip(events[:-1], events[1:]):
            prev_amt = prev[1]
            curr_amt = curr[1]
            if prev_amt <= 0:
                continue
            change = (curr_amt - prev_amt) / prev_amt
            if change <= -DIVIDEND_CUT_THRESHOLD:
                has_decrease = True
            elif change >= DIVIDEND_CUT_THRESHOLD:
                has_increase = True

        if not events:
            reasons["last_increase_date"] = "no dividend events in last 12m"
            reasons["last_decrease_date"] = "no dividend events in last 12m"
        else:
            if not has_increase:
                reasons["last_increase_date"] = f"no increase >= {int(DIVIDEND_CUT_THRESHOLD * 100)}% detected in last 12m"
            if not has_decrease:
                reasons["last_decrease_date"] = f"no decrease >= {int(DIVIDEND_CUT_THRESHOLD * 100)}% detected in last 12m"

        out[sym] = reasons
    return out


def _portfolio_income_reasons(conn: sqlite3.Connection, as_of_date: date) -> dict[str, str]:
    div_tx = _load_dividend_transactions(conn)
    totals_24 = _monthly_income_totals(div_tx, as_of_date, 24)
    totals_12 = totals_24[-12:] if len(totals_24) >= 12 else totals_24
    totals_6 = totals_12[-6:] if len(totals_12) >= 6 else totals_12

    start_6 = totals_6[0] if totals_6 else 0.0
    end_6 = totals_6[-1] if totals_6 else 0.0
    start_12 = totals_12[0] if totals_12 else 0.0
    end_12 = totals_12[-1] if totals_12 else 0.0
    reasons = {}

    if not div_tx:
        reasons["dividend_cut_count_12m"] = "no dividend payments recorded yet"
        reasons["missed_payment_count_12m"] = "no dividend payments recorded yet"

    if len(totals_12) < 6:
        reasons["qoq_pct"] = "insufficient dividend history for 2 quarters (need 6 months)"
    else:
        prev_q = sum(totals_12[-6:-3])
        if prev_q <= 0:
            reasons["qoq_pct"] = "prior quarter dividend total is 0; qoq undefined"

    if len(totals_24) < 24:
        reasons["yoy_pct"] = "insufficient dividend history for year-over-year (need 24 months)"
    else:
        prev_12 = sum(totals_24[-24:-12])
        if prev_12 <= 0:
            reasons["yoy_pct"] = "prior 12m dividend total is 0; yoy undefined"

    if start_6 <= 0 or end_6 <= 0:
        reasons["cagr_6m_pct"] = (
            f"requires positive dividends at start/end of 6m window (start={start_6:.2f}, end={end_6:.2f})"
        )
        reasons["income_growth_rate_6m_pct"] = (
            f"requires positive dividends at start/end of 6m window (start={start_6:.2f}, end={end_6:.2f})"
        )

    if start_12 <= 0 or end_12 <= 0:
        reasons["income_growth_rate_12m_pct"] = (
            f"requires positive dividends at start/end of 12m window (start={start_12:.2f}, end={end_12:.2f})"
        )

    return reasons

import calendar
import json
import math
import sqlite3
import statistics
import uuid
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ..config import settings
import structlog
from ..utils import sha256_json, now_utc_iso, to_local_date, to_local_datetime_iso
from .validation import validate_period_snapshot
from . import metrics

TRADE_TYPES = {"buy", "buy_shares", "reinvest", "reinvestment", "sell", "sell_shares", "redemption"}
DIVIDEND_CUT_THRESHOLD = 0.10


def _last_valid_price(df):
    if df is None or getattr(df, "empty", True):
        return None
    price_col = "adj_close" if "adj_close" in df.columns else "close"
    if price_col not in df.columns:
        return None
    series = df[price_col].dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])


def _price_series(df):
    if df is None or getattr(df, "empty", True):
        return None
    price_col = "adj_close" if "adj_close" in df.columns else "close"
    if price_col not in df.columns:
        return None
    if "date" in df.columns:
        series = df.set_index("date")[price_col]
    else:
        series = df[price_col]
    series = pd.to_numeric(series, errors="coerce").dropna()
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    return series.sort_index()


def _parse_date(val):
    if not val:
        return None
    text = str(val).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
        return dt.date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _month_start(d: date) -> date:
    return d.replace(day=1)


def _month_end(d: date) -> date:
    if d.month == 12:
        return date(d.year, 12, 31)
    return (d.replace(month=d.month + 1, day=1) - timedelta(days=1))


def _add_months(d: date, months: int) -> date:
    year = d.year + (d.month - 1 + months) // 12
    month = (d.month - 1 + months) % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _quarter_start(d: date) -> date:
    q = (d.month - 1) // 3
    return date(d.year, 1 + q * 3, 1)


def _year_start(d: date) -> date:
    return date(d.year, 1, 1)


def _safe_divide(a, b):
    if a is None or b in (None, 0, 0.0):
        return None
    return float(a / b)


def _is_number(val):
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def _risk_quality_category(sortino: float | None, sharpe: float | None) -> str | None:
    if sharpe in (0, 0.0):
        return "concerning"
    if sortino is None or sharpe is None:
        return None
    score = sortino / sharpe
    if score >= 1.2 and sortino >= 1.0:
        return "excellent"
    if score >= 1.0 and sortino >= 0.6:
        return "good"
    if score >= 0.8 or sortino >= 0.5:
        return "acceptable"
    return "concerning"


def _volatility_profile(sortino: float | None, sharpe: float | None) -> str | None:
    if sortino is None or sharpe is None:
        return None
    if sortino > sharpe:
        return "upside_biased"
    if sortino < sharpe:
        return "downside_biased"
    return "balanced"


def _round_money(val):
    return None if val is None else round(float(val), 2)


def _round_pct(val):
    return None if val is None else round(float(val), 3)


def _frequency_from_ex_dates(ex_dates):
    if len(ex_dates) < 2:
        return None
    gaps = sorted((b - a).days for a, b in zip(ex_dates[:-1], ex_dates[1:]) if b > a)
    if not gaps:
        return None
    median_gap = gaps[len(gaps) // 2]
    if median_gap < 45:
        return "monthly"
    if median_gap < 100:
        return "quarterly"
    if median_gap < 190:
        return "semiannual"
    return "annual"


def _frequency_from_dates(dates: list[date]) -> str | None:
    if not dates:
        return None
    dates = sorted(dates)
    return _frequency_from_ex_dates(dates)


def _gap_days(dates: list[date]) -> list[int]:
    if len(dates) < 2:
        return []
    return [max(1, (b - a).days) for a, b in zip(dates[:-1], dates[1:]) if b > a]


def _next_ex_date_est(ex_dates):
    if len(ex_dates) < 2:
        return None
    gaps = sorted((b - a).days for a, b in zip(ex_dates[:-1], ex_dates[1:]) if b > a)
    if not gaps:
        return None
    median_gap = gaps[len(gaps) // 2]
    return ex_dates[-1] + timedelta(days=median_gap)


def _payouts_per_year(freq: str | None):
    if freq == "monthly":
        return 12
    if freq == "quarterly":
        return 4
    if freq == "semiannual":
        return 2
    if freq == "annual":
        return 1
    return None


def _provenance_entry(source_type, provider, method, inputs, fetched_at, note=None):
    entry = {
        "source_type": source_type,
        "provider": provider,
        "method": method,
        "inputs": inputs,
        "validated_by": [],
        "fetched_at": fetched_at,
    }
    if note is not None:
        entry["note"] = note
    return entry


def _estimate_upcoming_per_share(div_events, as_of_date: date):
    if not div_events:
        return None, None
    ex_dates = [ev["ex_date"] for ev in div_events if ev.get("ex_date")]
    if not ex_dates:
        return None, None
    ex_dates.sort()
    freq = _frequency_from_ex_dates(ex_dates)
    payouts = _payouts_per_year(freq)

    trailing = [ev["amount"] for ev in div_events if ev["ex_date"] >= as_of_date - timedelta(days=365)]
    amounts = trailing if trailing else [ev["amount"] for ev in div_events]
    amounts = [amt for amt in amounts if amt is not None]
    if not amounts:
        return None, freq

    window = payouts if payouts else min(3, len(amounts))
    sample = amounts[-window:]
    sample_sorted = sorted(sample)
    per_share = sample_sorted[len(sample_sorted) // 2]
    return per_share, freq


def _portfolio_value_series(holdings: dict, price_series: dict):
    series_list = []
    for sym, h in holdings.items():
        series = price_series.get(sym)
        if series is None or series.empty:
            continue
        series_list.append(series * float(h["shares"]))
    if not series_list:
        return pd.Series(dtype=float)
    df = pd.concat(series_list, axis=1).sort_index()
    df = df.ffill()
    return df.sum(axis=1).dropna()


def _load_trade_counts(conn: sqlite3.Connection):
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT symbol, transaction_type FROM investment_transactions WHERE symbol IS NOT NULL"
    ).fetchall()
    counts = defaultdict(int)
    for symbol, tx_type in rows:
        if not symbol:
            continue
        tx_type = (tx_type or "").lower()
        if tx_type in TRADE_TYPES:
            counts[str(symbol).upper()] += 1
    return counts


def _load_dividend_transactions(conn: sqlite3.Connection):
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT symbol, date, amount
        FROM investment_transactions
        WHERE transaction_type='dividend' AND symbol IS NOT NULL
        """
    ).fetchall()
    out = []
    for symbol, date_str, amount in rows:
        dt = _parse_date(date_str)
        amt = None
        try:
            amt = abs(float(amount))
        except (TypeError, ValueError):
            amt = None
        if symbol and dt and amt is not None:
            out.append({"symbol": str(symbol).upper(), "date": dt, "amount": amt})
    return out


def _build_pay_history(div_tx: list[dict], max_events: int = 12):
    by_symbol_date = defaultdict(lambda: defaultdict(float))
    for tx in div_tx:
        sym = tx.get("symbol")
        dt = tx.get("date")
        amt = tx.get("amount")
        if not sym or not dt or not isinstance(amt, (int, float)):
            continue
        by_symbol_date[sym][dt] += float(amt)
    history = {}
    for sym, date_map in by_symbol_date.items():
        events = [{"date": dt, "amount": amt} for dt, amt in date_map.items()]
        events.sort(key=lambda item: item["date"])
        if max_events and len(events) > max_events:
            events = events[-max_events:]
        history[sym] = events
    return history


def _median_gap_days(dates: list[date]) -> int | None:
    if len(dates) < 2:
        return None
    gaps = sorted((b - a).days for a, b in zip(dates[:-1], dates[1:]) if b > a)
    if not gaps:
        return None
    return max(1, int(round(statistics.median(gaps))))


def _fallback_next_ex_date(last_ex_date: date | None, pay_history: list[dict]) -> date | None:
    if not last_ex_date or len(pay_history) < 2:
        return None
    dates = [ev.get("date") for ev in pay_history if ev.get("date")]
    gap = _median_gap_days(dates)
    if not gap:
        return None
    return last_ex_date + timedelta(days=gap)


def _median_amount(events: list[dict]) -> float | None:
    amounts = [ev.get("amount") for ev in events if isinstance(ev.get("amount"), (int, float))]
    if not amounts:
        return None
    return float(statistics.median(amounts))


def _dividend_reliability_metrics(
    symbol: str,
    div_tx: list[dict],
    provider_divs: dict,
    pay_history: dict,
    expected_frequency: str | None,
    as_of_date: date,
) -> dict:
    totals_12 = _monthly_income_totals(div_tx, as_of_date, 12, symbol=symbol)
    totals_6 = totals_12[-6:] if len(totals_12) >= 6 else totals_12

    mean_12 = statistics.mean(totals_12) if totals_12 else 0.0
    cv_12 = None
    consistency_score = 0.0
    if mean_12 > 0:
        cv_12 = statistics.pstdev(totals_12) / mean_12
        consistency_score = max(0.0, 1.0 - (cv_12 / 0.5))

    trend_6m = _trend_label(totals_6)
    growth_6m = None
    if len(totals_6) >= 2:
        growth_6m = _annualized_growth_rate(totals_6[0], totals_6[-1], len(totals_6) - 1)

    mean_6 = statistics.mean(totals_6) if totals_6 else 0.0
    vol_6m = None
    if mean_6 > 0:
        vol_6m = (statistics.pstdev(totals_6) / mean_6) * 100.0

    pay_events = pay_history.get(symbol, [])
    pay_dates = [ev.get("date") for ev in pay_events if ev.get("date") and ev.get("date") >= as_of_date - timedelta(days=365)]
    pay_dates = [d for d in pay_dates if d]
    pay_dates.sort()
    payment_frequency_actual = _frequency_from_dates(pay_dates)
    payment_frequency_expected = expected_frequency or payment_frequency_actual

    expected_count = _payouts_per_year(payment_frequency_expected)
    actual_count = len(pay_dates)
    missed_payments = None
    if expected_count is not None:
        missed_payments = max(expected_count - actual_count, 0)

    gap_days = _gap_days(pay_dates)
    avg_gap = round(statistics.mean(gap_days), 1) if gap_days else None
    timing_consistency = None
    if gap_days:
        mean_gap = statistics.mean(gap_days)
        if mean_gap > 0:
            timing_consistency = max(0.0, min(1.0, 1.0 - (statistics.pstdev(gap_days) / mean_gap)))

    events = []
    for ev in provider_divs.get(symbol, []):
        ex_date = ev.get("ex_date")
        amt = ev.get("amount")
        if ex_date and isinstance(amt, (int, float)) and ex_date >= as_of_date - timedelta(days=365):
            events.append((ex_date, float(amt)))
    events.sort(key=lambda item: item[0])
    if not events and pay_events:
        for ev in pay_events:
            dt = ev.get("date")
            amt = ev.get("amount")
            if dt and isinstance(amt, (int, float)) and dt >= as_of_date - timedelta(days=365):
                events.append((dt, float(amt)))
        events.sort(key=lambda item: item[0])

    cuts = 0
    last_increase = None
    last_decrease = None
    for prev, curr in zip(events[:-1], events[1:]):
        prev_amt = prev[1]
        curr_amt = curr[1]
        if prev_amt <= 0:
            continue
        change = (curr_amt - prev_amt) / prev_amt
        if change <= -DIVIDEND_CUT_THRESHOLD:
            cuts += 1
            last_decrease = curr[0]
        elif change >= DIVIDEND_CUT_THRESHOLD:
            last_increase = curr[0]

    return {
        "consistency_score": round(consistency_score, 3),
        "payment_frequency_actual": payment_frequency_actual,
        "payment_frequency_expected": payment_frequency_expected,
        "missed_payments_12m": missed_payments,
        "dividend_cuts_12m": cuts,
        "dividend_trend_6m": trend_6m,
        "dividend_growth_rate_6m_pct": _round_pct(growth_6m * 100 if growth_6m is not None else None),
        "dividend_volatility_6m_pct": _round_pct(vol_6m),
        "avg_days_between_payments": avg_gap,
        "payment_timing_consistency": round(timing_consistency, 3) if timing_consistency is not None else None,
        "last_increase_date": last_increase.isoformat() if last_increase else None,
        "last_decrease_date": last_decrease.isoformat() if last_decrease else None,
    }


def _load_provider_dividends(conn: sqlite3.Connection):
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT symbol, ex_date, pay_date, amount, source
        FROM dividend_events_provider
        """
    ).fetchall()
    out = defaultdict(list)
    seen = set()
    for symbol, ex_date, pay_date, amount, _source in rows:
        sym = str(symbol).upper() if symbol else None
        dt = _parse_date(ex_date)
        pay_dt = _parse_date(pay_date)
        try:
            amt = float(amount)
        except (TypeError, ValueError):
            amt = None
        if not sym or dt is None or amt is None:
            continue
        key = (sym, dt.isoformat(), amt)
        if key in seen:
            continue
        seen.add(key)
        out[sym].append({"ex_date": dt, "pay_date": pay_dt, "amount": amt})
    for sym in out:
        out[sym].sort(key=lambda item: item["ex_date"])
    return out


def _load_lm_ex_dates(conn: sqlite3.Connection):
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT symbol, ex_date, ex_date_est
        FROM dividend_events_lm
        """
    ).fetchall()
    out = defaultdict(list)
    seen = set()
    for symbol, ex_date, ex_date_est in rows:
        sym = str(symbol).upper() if symbol else None
        dt = _parse_date(ex_date_est) or _parse_date(ex_date)
        if not sym or dt is None:
            continue
        key = (sym, dt.isoformat())
        if key in seen:
            continue
        seen.add(key)
        out[sym].append(dt)
    for sym in out:
        out[sym].sort()
    return out


def _effective_ex_dates(sym: str, provider_divs: dict, lm_ex_dates: dict):
    ex_dates = [ev["ex_date"] for ev in provider_divs.get(sym, []) if ev.get("ex_date")]
    if not ex_dates:
        ex_dates = list(lm_ex_dates.get(sym, []))
    if not ex_dates:
        return []
    return sorted({dt for dt in ex_dates if dt})


def _median_pay_lag_days(provider_divs: dict) -> int:
    lags = []
    for events in provider_divs.values():
        for ev in events:
            ex = ev.get("ex_date")
            pay = ev.get("pay_date")
            if not ex or not pay:
                continue
            delta = (pay - ex).days
            if delta >= 0 and delta <= 60:
                lags.append(delta)
    if not lags:
        return 14
    return max(1, int(round(statistics.median(lags))))


def _symbol_pay_lag_days(events: list[dict], default_lag: int) -> int:
    lags = []
    for ev in events:
        ex = ev.get("ex_date")
        pay = ev.get("pay_date")
        if not ex or not pay:
            continue
        delta = (pay - ex).days
        if delta >= 0 and delta <= 60:
            lags.append(delta)
    if not lags:
        return default_lag
    return max(1, int(round(statistics.median(lags))))


def _estimate_expected_pay_events(
    provider_divs: dict,
    holdings: dict,
    window_start: date,
    window_end: date,
    pay_history: dict,
    ex_date_est_by_symbol: dict[str, date] | None = None,
):
    default_lag = _median_pay_lag_days(provider_divs)
    expected = []
    total_raw = 0.0
    ex_date_est_by_symbol = ex_date_est_by_symbol or {}
    for sym in sorted(holdings.keys()):
        history = pay_history.get(sym, [])
        actual_events = [ev for ev in history if window_start <= ev["date"] <= window_end]
        if actual_events:
            for ev in actual_events:
                ex_date_est = ex_date_est_by_symbol.get(sym)
                if not ex_date_est:
                    continue
                amount_est = ev.get("amount")
                expected.append(
                    {
                        "symbol": sym,
                        "ex_date_est": ex_date_est.isoformat(),
                        "pay_date_est": ev["date"].isoformat(),
                        "amount_est": _round_money(amount_est),
                    }
                )
                if isinstance(amount_est, (int, float)):
                    total_raw += float(amount_est)
            continue

        if len(history) >= 2:
            dates = [ev["date"] for ev in history]
            median_gap = _median_gap_days(dates)
            if median_gap:
                next_pay = dates[-1] + timedelta(days=median_gap)
                if window_start <= next_pay <= window_end:
                    ex_date_est = ex_date_est_by_symbol.get(sym)
                    if not ex_date_est:
                        continue
                    amount_est = _median_amount(history[-6:])
                    expected.append(
                        {
                            "symbol": sym,
                            "ex_date_est": ex_date_est.isoformat(),
                            "pay_date_est": next_pay.isoformat(),
                            "amount_est": _round_money(amount_est),
                        }
                    )
                    if isinstance(amount_est, (int, float)):
                        total_raw += float(amount_est)
            continue

        events = provider_divs.get(sym, [])
        if not events:
            continue
        lag_days = _symbol_pay_lag_days(events, default_lag)
        shares = float(holdings[sym]["shares"])
        for ev in events:
            ex = ev.get("ex_date")
            amt = ev.get("amount")
            if ex is None or amt is None:
                continue
            pay = ev.get("pay_date") or (ex + timedelta(days=lag_days))
            if pay < window_start or pay > window_end:
                continue
            amount_est = amt * shares
            total_raw += float(amount_est)
            expected.append(
                {
                    "symbol": sym,
                    "ex_date_est": ex.isoformat(),
                    "pay_date_est": pay.isoformat(),
                    "amount_est": _round_money(amount_est),
                }
            )
    expected.sort(key=lambda ev: (ev.get("pay_date_est") or "", ev.get("symbol") or ""))
    return expected, _round_money(total_raw)


def _load_account_balances(conn: sqlite3.Connection):
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT plaid_account_id, name, type, subtype, balance, as_of_date_local
        FROM account_balances
        ORDER BY as_of_date_local DESC
        """
    ).fetchall()
    latest = {}
    for plaid_account_id, name, type_val, subtype, balance, as_of_date_local in rows:
        if plaid_account_id in latest:
            continue
        latest[plaid_account_id] = {
            "plaid_account_id": plaid_account_id,
            "name": name,
            "type": type_val,
            "subtype": subtype,
            "balance": balance,
            "as_of_date_local": as_of_date_local,
        }
    return list(latest.values())


def _load_daily_snapshots_range(conn: sqlite3.Connection, start_date: date, end_date: date) -> list[tuple[date, dict]]:
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT as_of_date_local, payload_json
        FROM snapshot_daily_current
        WHERE as_of_date_local BETWEEN ? AND ?
        ORDER BY as_of_date_local ASC
        """,
        (start_date.isoformat(), end_date.isoformat()),
    ).fetchall()
    out = []
    for as_of_str, payload in rows:
        try:
            snap = json.loads(payload)
        except Exception:
            continue
        try:
            dt = date.fromisoformat(as_of_str)
        except Exception:
            continue
        out.append((dt, snap))
    return out


def _is_margin_account(name: str | None, type_val: str | None, subtype: str | None):
    text = " ".join([name or "", type_val or "", subtype or ""]).lower()
    if "borrow" in text or "margin" in text or "loan" in text:
        return True
    return False


def _dividend_window(events, start: date, end: date, holding_symbols: set[str]):
    filtered = [e for e in events if start <= e["date"] <= end]
    by_date = defaultdict(float)
    by_month = defaultdict(float)
    by_symbol = defaultdict(float)
    for e in filtered:
        by_date[e["date"].isoformat()] += e["amount"]
        by_month[e["date"].strftime("%Y-%m")] += e["amount"]
        by_symbol[e["symbol"]] += e["amount"]
    by_symbol_out = {}
    for sym, amt in sorted(by_symbol.items()):
        by_symbol_out[sym] = {
            "amount": round(amt, 2),
            "status": "active" if sym in holding_symbols else "inactive",
        }
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "total_dividends": round(sum(by_date.values()), 2),
        "by_date": {k: round(v, 2) for k, v in by_date.items()},
        "by_month": {k: round(v, 2) for k, v in by_month.items()},
        "by_symbol": by_symbol_out,
    }


def _month_keys(end_date: date, window_months: int) -> list[tuple[int, int]]:
    keys = []
    for i in range(window_months - 1, -1, -1):
        year = end_date.year + (end_date.month - 1 - i) // 12
        month = (end_date.month - 1 - i) % 12 + 1
        keys.append((year, month))
    return keys


def _monthly_income_totals(div_tx: list[dict], as_of_date: date, window_months: int, symbol: str | None = None) -> list[float]:
    totals_by_month: dict[tuple[int, int], float] = defaultdict(float)
    for tx in div_tx:
        dt = tx.get("date")
        if dt is None or dt > as_of_date:
            continue
        if symbol and tx.get("symbol") != symbol:
            continue
        totals_by_month[(dt.year, dt.month)] += float(tx.get("amount") or 0.0)
    months = _month_keys(as_of_date, window_months)
    return [totals_by_month.get(key, 0.0) for key in months]


def _annualized_growth_rate(start: float, end: float, periods: int) -> float | None:
    if start <= 0 or end <= 0 or periods <= 0:
        return None
    return (end / start) ** (12.0 / periods) - 1.0


def _trend_label(values: list[float], threshold_pct: float = 0.02) -> str | None:
    if not values or len(values) < 2:
        return None
    mean_val = statistics.mean(values)
    if mean_val <= 0:
        return "stable"
    slope = np.polyfit(range(len(values)), values, 1)[0]
    threshold = abs(mean_val) * threshold_pct
    if slope > threshold:
        return "growing"
    if slope < -threshold:
        return "declining"
    return "stable"


def _count_dividend_cuts_from_totals(totals: list[float], threshold: float = DIVIDEND_CUT_THRESHOLD) -> int:
    cuts = 0
    for prev, curr in zip(totals[:-1], totals[1:]):
        if prev > 0 and curr < prev * (1.0 - threshold):
            cuts += 1
    return cuts


def _income_volatility_30d_pct(div_tx: list[dict], as_of_date: date) -> float | None:
    start = as_of_date - timedelta(days=29)
    by_date: dict[date, float] = defaultdict(float)
    for tx in div_tx:
        dt = tx.get("date")
        if dt is None or dt < start or dt > as_of_date:
            continue
        by_date[dt] += float(tx.get("amount") or 0.0)
    totals = []
    for i in range(30):
        day = start + timedelta(days=i)
        totals.append(by_date.get(day, 0.0))
    mean = statistics.mean(totals) if totals else 0.0
    if mean <= 0:
        return None
    stdev = statistics.pstdev(totals)
    return round((stdev / mean) * 100.0, 3)


def _income_stability_metrics(div_tx: list[dict], as_of_date: date, window_months: int = 12) -> dict:
    totals_12 = _monthly_income_totals(div_tx, as_of_date, window_months)
    if not totals_12:
        return {
            "stability_score": 0.0,
            "coefficient_of_variation": None,
            "income_trend_6m": None,
            "income_growth_rate_6m_pct": None,
            "income_growth_rate_12m_pct": None,
            "income_volatility_30d_pct": None,
            "consecutive_months_positive": 0,
            "income_drawdown_max_pct": None,
            "dividend_cut_count_12m": 0,
            "missed_payment_count_12m": 0,
        }

    mean_12 = statistics.mean(totals_12)
    cv = None
    consistency_score = 0.0
    if mean_12 > 0:
        stdev_12 = statistics.pstdev(totals_12)
        cv = stdev_12 / mean_12 if mean_12 else None
        if cv is not None:
            consistency_score = max(0.0, 1.0 - (cv / 0.5))

    totals_6 = totals_12[-6:] if len(totals_12) >= 6 else totals_12
    trend_6m = _trend_label(totals_6)
    trend_bonus = 0.1 if trend_6m == "growing" else 0.0

    cuts = _count_dividend_cuts_from_totals(totals_12)
    missed = sum(1 for val in totals_12 if val <= 0)

    score = min(1.0, max(0.0, consistency_score + trend_bonus - (cuts * 0.15) - (missed * 0.2)))

    growth_6m = None
    if len(totals_6) >= 2:
        growth_6m = _annualized_growth_rate(totals_6[0], totals_6[-1], len(totals_6) - 1)
    growth_12m = None
    if len(totals_12) >= 2:
        growth_12m = _annualized_growth_rate(totals_12[0], totals_12[-1], len(totals_12) - 1)

    drawdown_max = None
    for prev, curr in zip(totals_12[:-1], totals_12[1:]):
        if prev > 0:
            change = (curr / prev) - 1.0
            if drawdown_max is None or change < drawdown_max:
                drawdown_max = change

    consecutive_positive = 0
    for val in reversed(totals_12):
        if val > 0:
            consecutive_positive += 1
        else:
            break

    return {
        "stability_score": round(score, 3),
        "coefficient_of_variation": round(cv, 3) if cv is not None else None,
        "income_trend_6m": trend_6m,
        "income_growth_rate_6m_pct": _round_pct(growth_6m * 100 if growth_6m is not None else None),
        "income_growth_rate_12m_pct": _round_pct(growth_12m * 100 if growth_12m is not None else None),
        "income_volatility_30d_pct": _round_pct(_income_volatility_30d_pct(div_tx, as_of_date)),
        "consecutive_months_positive": consecutive_positive,
        "income_drawdown_max_pct": _round_pct(drawdown_max * 100 if drawdown_max is not None else None),
        "dividend_cut_count_12m": cuts,
        "missed_payment_count_12m": missed,
    }


def _income_stability_score(div_tx: list[dict], as_of_date: date, window_months: int = 6) -> float:
    metrics = _income_stability_metrics(div_tx, as_of_date, window_months=max(window_months, 6))
    return float(metrics.get("stability_score") or 0.0)


def _income_growth_metrics(div_tx: list[dict], as_of_date: date) -> dict:
    totals_24 = _monthly_income_totals(div_tx, as_of_date, 24)
    totals_12 = totals_24[-12:] if len(totals_24) >= 12 else totals_24
    totals_6 = totals_12[-6:] if len(totals_12) >= 6 else totals_12

    mom_abs = None
    mom_pct = None
    if len(totals_12) >= 2:
        prev = totals_12[-2]
        curr = totals_12[-1]
        mom_abs = curr - prev
        if prev > 0:
            mom_pct = (curr / prev - 1.0) * 100.0

    qoq_pct = None
    if len(totals_12) >= 6:
        last_q = sum(totals_12[-3:])
        prev_q = sum(totals_12[-6:-3])
        if prev_q > 0:
            qoq_pct = (last_q / prev_q - 1.0) * 100.0

    yoy_pct = None
    if len(totals_24) >= 24:
        last_12 = sum(totals_24[-12:])
        prev_12 = sum(totals_24[-24:-12])
        if prev_12 > 0:
            yoy_pct = (last_12 / prev_12 - 1.0) * 100.0

    cagr_6m = None
    if len(totals_6) >= 2:
        cagr_6m = _annualized_growth_rate(totals_6[0], totals_6[-1], len(totals_6) - 1)

    trend_12m = None
    if len(totals_12) >= 12:
        first_half = totals_12[:6]
        second_half = totals_12[6:]
        mean_val = statistics.mean(totals_12) if totals_12 else 0.0
        if mean_val > 0:
            slope_1 = np.polyfit(range(len(first_half)), first_half, 1)[0]
            slope_2 = np.polyfit(range(len(second_half)), second_half, 1)[0]
            delta = slope_2 - slope_1
            threshold = abs(mean_val) * 0.01
            if delta > threshold:
                trend_12m = "accelerating"
            elif delta < -threshold:
                trend_12m = "decelerating"
            else:
                trend_12m = "stable"
        else:
            trend_12m = "stable"

    return {
        "mom_pct": _round_pct(mom_pct),
        "mom_absolute": _round_money(mom_abs),
        "qoq_pct": _round_pct(qoq_pct),
        "yoy_pct": _round_pct(yoy_pct),
        "cagr_6m_pct": _round_pct(cagr_6m * 100 if cagr_6m is not None else None),
        "trend_12m": trend_12m,
    }


def _tail_risk_category(cvar_1d_pct: float | None) -> str | None:
    if cvar_1d_pct is None:
        return None
    if cvar_1d_pct > -2.0:
        return "low"
    if cvar_1d_pct > -4.0:
        return "moderate"
    if cvar_1d_pct > -6.0:
        return "high"
    return "severe"


def _ulcer_index_category(val: float | None) -> str | None:
    if val is None:
        return None
    if val < 5.0:
        return "low"
    if val < 10.0:
        return "moderate"
    return "high"


def _drawdown_analysis(values: pd.Series) -> tuple[dict | None, dict | None]:
    if values is None or values.empty:
        return None, None
    v = values.dropna()
    if v.empty:
        return None, None
    if not isinstance(v.index, pd.DatetimeIndex):
        v = v.copy()
        v.index = pd.to_datetime(v.index)
    v = v.sort_index()

    peak_val = v.iloc[0]
    peak_date = v.index[0]
    trough_val = peak_val
    trough_date = peak_date
    in_drawdown = False
    episodes = []

    for dt, val in v.items():
        if val >= peak_val:
            if in_drawdown:
                recovery_date = dt
                depth_pct = (trough_val / peak_val - 1.0) * 100.0
                recovery_days = (recovery_date - trough_date).days
                duration_days = (recovery_date - peak_date).days
                episodes.append(
                    {
                        "depth_pct": depth_pct,
                        "recovery_days": recovery_days,
                        "duration_days": duration_days,
                        "peak_date": peak_date,
                        "trough_date": trough_date,
                        "recovery_date": recovery_date,
                    }
                )
                in_drawdown = False
            peak_val = val
            peak_date = dt
            trough_val = val
            trough_date = dt
        else:
            in_drawdown = True
            if val < trough_val:
                trough_val = val
                trough_date = dt

    current_val = v.iloc[-1]
    current_date = v.index[-1]
    peak_series = v.cummax()
    current_peak = peak_series.iloc[-1]
    peak_candidates = v[v == current_peak]
    current_peak_date = peak_candidates.index[-1] if not peak_candidates.empty else v.index[-1]

    if current_val < current_peak:
        current_drawdown_pct = (current_val / current_peak - 1.0) * 100.0
        days_since_peak = (current_date - current_peak_date).days
        values_since_peak = v.loc[current_peak_date:]
        trough_since_peak = values_since_peak.min()
        recovery_progress = None
        if trough_since_peak < current_peak:
            if current_val > trough_since_peak:
                recovery_progress = (current_val - trough_since_peak) / (current_peak - trough_since_peak) * 100.0
            else:
                recovery_progress = 0.0
        drawdown_status = {
            "currently_in_drawdown": True,
            "current_drawdown_depth_pct": _round_pct(current_drawdown_pct),
            "current_drawdown_duration_days": days_since_peak,
            "days_since_peak": days_since_peak,
            "peak_value": _round_money(current_peak),
            "peak_date": current_peak_date.date().isoformat(),
            "recovery_progress_pct": _round_pct(recovery_progress),
        }
    else:
        drawdown_status = {
            "currently_in_drawdown": False,
            "current_drawdown_depth_pct": 0.0,
            "current_drawdown_duration_days": 0,
            "days_since_peak": 0,
            "peak_value": _round_money(current_val),
            "peak_date": current_date.date().isoformat(),
            "recovery_progress_pct": 100.0,
        }

    total_drawdowns = len(episodes) + (1 if drawdown_status.get("currently_in_drawdown") else 0)
    recovered = len(episodes)
    recovery_days_list = [ep["recovery_days"] for ep in episodes if ep["recovery_days"] > 0]
    speed_list = []
    for ep in episodes:
        if ep["recovery_days"] > 0:
            speed_list.append(abs(ep["depth_pct"]) / ep["recovery_days"])

    recovery_metrics = {
        "avg_recovery_time_days": round(sum(recovery_days_list) / len(recovery_days_list), 1) if recovery_days_list else None,
        "fastest_recovery_days": min(recovery_days_list) if recovery_days_list else None,
        "slowest_recovery_days": max(recovery_days_list) if recovery_days_list else None,
        "recovery_success_rate_pct": round((recovered / total_drawdowns) * 100.0, 1) if total_drawdowns else None,
        "median_recovery_speed_pct_per_day": round(statistics.median(speed_list), 2) if speed_list else None,
    }

    if drawdown_status.get("currently_in_drawdown"):
        speed = recovery_metrics.get("median_recovery_speed_pct_per_day")
        remaining_pct = abs(drawdown_status.get("current_drawdown_depth_pct") or 0.0)
        if speed and speed > 0:
            recovery_metrics["estimated_days_to_recovery"] = int(math.ceil(remaining_pct / speed))
        else:
            recovery_metrics["estimated_days_to_recovery"] = None
    else:
        recovery_metrics["estimated_days_to_recovery"] = 0

    return drawdown_status, recovery_metrics


def _window_slice(series: pd.Series, end_dt: pd.Timestamp, window_days: int) -> pd.Series | None:
    if series is None or series.empty:
        return None
    if not isinstance(series.index, pd.DatetimeIndex):
        series = series.copy()
        series.index = pd.to_datetime(series.index)
    series = series.sort_index()
    start_dt = end_dt - pd.Timedelta(days=window_days)
    window = series.loc[series.index >= start_dt]
    return window if not window.empty else None


def _sum_dividends(div_tx: list[dict], symbol: str, start_date: date, end_date: date) -> float:
    total = 0.0
    for tx in div_tx:
        if tx.get("symbol") != symbol:
            continue
        dt = tx.get("date")
        if dt is None or dt < start_date or dt > end_date:
            continue
        total += float(tx.get("amount") or 0.0)
    return total


def _return_attribution(
    holdings: dict,
    price_series: dict,
    div_tx: list[dict],
    end_dt: pd.Timestamp,
    window_days: int,
) -> tuple[dict, dict]:
    portfolio_series = _portfolio_value_series(holdings, price_series)
    window_portfolio = _window_slice(portfolio_series, end_dt, window_days)
    if window_portfolio is None or window_portfolio.empty:
        return {}, {}

    start_val = float(window_portfolio.iloc[0])
    end_val = float(window_portfolio.iloc[-1])
    start_date = window_portfolio.index[0].date()
    end_date = window_portfolio.index[-1].date()

    per_symbol = {}
    income_total = 0.0
    price_total = 0.0

    for sym, h in holdings.items():
        series = price_series.get(sym)
        window = _window_slice(series, end_dt, window_days) if series is not None else None
        if window is None or window.empty:
            continue
        shares = float(h.get("shares") or 0.0)
        if shares == 0:
            continue
        start_price = float(window.iloc[0])
        end_price = float(window.iloc[-1])
        position_start_value = start_price * shares
        price_return_dollars = (end_price - start_price) * shares
        income_dollars = _sum_dividends(div_tx, sym, start_date, end_date)
        total_return_dollars = price_return_dollars + income_dollars

        weight_pct = (position_start_value / start_val * 100.0) if start_val else None
        contribution_pct = (total_return_dollars / start_val * 100.0) if start_val else None
        income_contribution_pct = (income_dollars / start_val * 100.0) if start_val else None
        price_contribution_pct = (price_return_dollars / start_val * 100.0) if start_val else None

        roi_on_cost = None
        if h.get("cost_basis"):
            roi_on_cost = (total_return_dollars / float(h["cost_basis"])) * 100.0
        roi_annualized = None
        if roi_on_cost is not None:
            base = 1.0 + (roi_on_cost / 100.0)
            if base > 0:
                roi_annualized = (base ** (365.0 / window_days) - 1.0) * 100.0

        per_symbol[sym] = {
            "return_contribution_pct": _round_pct(contribution_pct),
            "return_contribution_dollars": _round_money(total_return_dollars),
            "income_contribution": {"pct": _round_pct(income_contribution_pct), "dollars": _round_money(income_dollars)},
            "price_contribution": {"pct": _round_pct(price_contribution_pct), "dollars": _round_money(price_return_dollars)},
            "roi_on_cost_pct": _round_pct(roi_on_cost),
            "roi_annualized_pct": _round_pct(roi_annualized),
            "position_weight_pct": _round_pct(weight_pct),
        }

        income_total += income_dollars
        price_total += price_return_dollars

    total_return_dollars = price_total + income_total
    total_return_pct = (total_return_dollars / start_val * 100.0) if start_val else None
    income_pct = (income_total / start_val * 100.0) if start_val else None
    price_pct = (price_total / start_val * 100.0) if start_val else None

    summary = {
        "total_return_pct": _round_pct(total_return_pct),
        "total_return_dollars": _round_money(total_return_dollars),
        "income_contribution_pct": _round_pct(income_pct),
        "income_contribution_dollars": _round_money(income_total),
        "price_contribution_pct": _round_pct(price_pct),
        "price_contribution_dollars": _round_money(price_total),
        "window": {"start": start_date.isoformat(), "end": end_date.isoformat(), "days": window_days},
    }

    return summary, per_symbol


def _return_attribution_all_periods(
    holdings: dict,
    price_series: dict,
    div_tx: list[dict],
    end_dt: pd.Timestamp,
) -> tuple[dict, dict]:
    periods = {"1m": 30, "3m": 90, "6m": 180, "12m": 365}
    portfolio_rollups = {}
    per_symbol = defaultdict(dict)

    for label, days in periods.items():
        summary, per_symbol_metrics = _return_attribution(holdings, price_series, div_tx, end_dt, days)
        if not summary:
            continue

        ranked = sorted(
            per_symbol_metrics.items(),
            key=lambda item: (item[1].get("return_contribution_pct") or 0.0),
            reverse=True,
        )
        for idx, (sym, metrics_out) in enumerate(ranked, start=1):
            metrics_out["rank_this_period"] = idx

        prev_end_dt = end_dt - pd.Timedelta(days=days)
        _prev_summary, prev_metrics = _return_attribution(holdings, price_series, div_tx, prev_end_dt, days)
        prev_ranked = sorted(
            prev_metrics.items(),
            key=lambda item: (item[1].get("return_contribution_pct") or 0.0),
            reverse=True,
        )
        prev_ranks = {sym: idx for idx, (sym, _metrics) in enumerate(prev_ranked, start=1)}

        top = []
        bottom = []
        for sym, metrics_out in ranked[:3]:
            top.append(
                {
                    "symbol": sym,
                    "contribution_pct": metrics_out.get("return_contribution_pct"),
                    "contribution_dollars": metrics_out.get("return_contribution_dollars"),
                    "contribution_breakdown": {
                        "income": (metrics_out.get("income_contribution") or {}).get("pct"),
                        "price": (metrics_out.get("price_contribution") or {}).get("pct"),
                    },
                }
            )
        bottom_ranked = sorted(
            per_symbol_metrics.items(),
            key=lambda item: (item[1].get("return_contribution_pct") or 0.0),
        )
        for sym, metrics_out in bottom_ranked[:3]:
            bottom.append(
                {
                    "symbol": sym,
                    "contribution_pct": metrics_out.get("return_contribution_pct"),
                    "contribution_dollars": metrics_out.get("return_contribution_dollars"),
                    "contribution_breakdown": {
                        "income": (metrics_out.get("income_contribution") or {}).get("pct"),
                        "price": (metrics_out.get("price_contribution") or {}).get("pct"),
                    },
                }
            )

        summary["top_contributors"] = top
        summary["bottom_contributors"] = bottom
        portfolio_rollups[f"return_attribution_{label}"] = summary

        for sym, metrics_out in per_symbol_metrics.items():
            metrics_out["rank_last_period"] = prev_ranks.get(sym)
            per_symbol[sym][label] = metrics_out

    return portfolio_rollups, per_symbol


def _symbol_metrics(series, benchmark_series):
    if series is None or series.empty:
        return {}
    returns = metrics.time_weighted_returns(series)
    benchmark_returns = metrics.time_weighted_returns(benchmark_series) if benchmark_series is not None else None
    beta, corr = metrics.beta_and_corr(returns, benchmark_returns, window_days=365)
    max_dd, dd_dur = metrics.max_drawdown(series, window_days=365)
    var_90, cvar_90 = metrics.var_cvar(returns, alpha=0.10, window_days=365)
    var_95, cvar_95 = metrics.var_cvar(returns, alpha=0.05, window_days=365)
    var_99, cvar_99 = metrics.var_cvar(returns, alpha=0.01, window_days=365)
    vol_30d = metrics.annualized_volatility(returns, window_days=30)
    vol_90d = metrics.annualized_volatility(returns, window_days=90)
    downside = metrics.downside_deviation(returns, window_days=365)
    sharpe = metrics.sharpe_ratio(returns, window_days=365)
    sortino = metrics.sortino_ratio(returns, window_days=365)
    sortino_6m = metrics.sortino_ratio(returns, window_days=180)
    sortino_3m = metrics.sortino_ratio(returns, window_days=90)
    sortino_1m = metrics.sortino_ratio(returns, window_days=30)
    risk_quality_score = _safe_divide(sortino, sharpe)
    risk_quality_category = _risk_quality_category(sortino, sharpe)
    volatility_profile = _volatility_profile(sortino, sharpe)
    twr_1m = metrics.twr(series, window_days=30)
    twr_3m = metrics.twr(series, window_days=90)
    twr_6m = metrics.twr(series, window_days=180)
    twr_12m = metrics.twr(series, window_days=365)
    calmar = _safe_divide(twr_12m, abs(max_dd) if max_dd is not None else None)
    return {
        "vol_30d_pct": _round_pct(vol_30d * 100 if vol_30d is not None else None),
        "vol_90d_pct": _round_pct(vol_90d * 100 if vol_90d is not None else None),
        "beta_3y": _round_pct(beta),
        "max_drawdown_1y_pct": _round_pct(max_dd * 100 if max_dd is not None else None),
        "sortino_1y": _round_pct(sortino),
        "sortino_6m": _round_pct(sortino_6m),
        "sortino_3m": _round_pct(sortino_3m),
        "sortino_1m": _round_pct(sortino_1m),
        "downside_dev_1y_pct": _round_pct(downside * 100 if downside is not None else None),
        "sharpe_1y": _round_pct(sharpe),
        "calmar_1y": _round_pct(calmar),
        "risk_quality_score": _round_pct(risk_quality_score),
        "risk_quality_category": risk_quality_category,
        "volatility_profile": volatility_profile,
        "drawdown_duration_1y_days": dd_dur,
        "var_90_1d_pct": _round_pct(var_90 * 100 if var_90 is not None else None),
        "var_95_1d_pct": _round_pct(var_95 * 100 if var_95 is not None else None),
        "var_99_1d_pct": _round_pct(var_99 * 100 if var_99 is not None else None),
        "var_95_1w_pct": _round_pct(metrics.scale_by_time(var_95, 5) * 100 if var_95 is not None else None),
        "var_95_1m_pct": _round_pct(metrics.scale_by_time(var_95, 21) * 100 if var_95 is not None else None),
        "cvar_90_1d_pct": _round_pct(cvar_90 * 100 if cvar_90 is not None else None),
        "cvar_95_1d_pct": _round_pct(cvar_95 * 100 if cvar_95 is not None else None),
        "cvar_99_1d_pct": _round_pct(cvar_99 * 100 if cvar_99 is not None else None),
        "cvar_95_1w_pct": _round_pct(metrics.scale_by_time(cvar_95, 5) * 100 if cvar_95 is not None else None),
        "cvar_95_1m_pct": _round_pct(metrics.scale_by_time(cvar_95, 21) * 100 if cvar_95 is not None else None),
        "corr_1y": _round_pct(corr),
        "twr_1m_pct": _round_pct(twr_1m * 100 if twr_1m is not None else None),
        "twr_3m_pct": _round_pct(twr_3m * 100 if twr_3m is not None else None),
        "twr_6m_pct": _round_pct(twr_6m * 100 if twr_6m is not None else None),
        "twr_12m_pct": _round_pct(twr_12m * 100 if twr_12m is not None else None),
    }


def _build_margin_guidance(total_market_value, margin_loan_balance, projected_monthly_income):
    total_market_value = total_market_value or 0.0
    margin_loan_balance = margin_loan_balance or 0.0
    ltv_now_pct = _safe_divide(margin_loan_balance, total_market_value)
    ltv_now_pct = _round_pct(ltv_now_pct * 100 if ltv_now_pct is not None else None)
    monthly_interest_now = margin_loan_balance * settings.margin_apr_current / 12.0
    monthly_interest_future = margin_loan_balance * settings.margin_apr_future / 12.0
    income_cov_now = _safe_divide(projected_monthly_income, monthly_interest_now)
    income_cov_future = _safe_divide(projected_monthly_income, monthly_interest_future)

    modes = []
    for mode, max_pct, stress_drawdown, min_coverage in [
        ("conservative", 20.0, 0.40, 2.0),
        ("balanced", 25.0, 0.30, 1.5),
        ("aggressive", 30.0, 0.20, 1.2),
    ]:
        max_balance = total_market_value * (max_pct / 100.0)
        repay_amt = max(margin_loan_balance - max_balance, 0.0)
        action = "repay" if repay_amt > 0 else "hold"
        new_balance = margin_loan_balance - repay_amt
        ltv_after = _safe_divide(new_balance, total_market_value)
        ltv_stress = _safe_divide(margin_loan_balance, total_market_value * (1.0 - stress_drawdown)) if total_market_value else None
        after_monthly_interest = new_balance * settings.margin_apr_current / 12.0
        after_income_cov = _safe_divide(projected_monthly_income, after_monthly_interest)
        modes.append(
            {
                "mode": mode,
                "action": action,
                "amount": _round_money(repay_amt),
                "ltv_now_pct": ltv_now_pct,
                "ltv_stress_pct": _round_pct(ltv_stress * 100 if ltv_stress is not None else None),
                "monthly_interest_now": _round_money(monthly_interest_now),
                "income_interest_coverage_now": _round_pct(income_cov_now),
                "monthly_interest_future": _round_money(monthly_interest_future),
                "income_interest_coverage_future": _round_pct(income_cov_future),
                "after_action": {
                    "new_loan_balance": _round_money(new_balance),
                    "ltv_after_action_pct": _round_pct(ltv_after * 100 if ltv_after is not None else None),
                    "monthly_interest_now": _round_money(after_monthly_interest),
                    "income_interest_coverage_now": _round_pct(after_income_cov),
                },
                "constraints": {
                    "max_margin_pct": max_pct,
                    "stress_drawdown_pct": stress_drawdown * 100,
                    "min_income_coverage": min_coverage,
                },
            }
        )

    return {
        "modes": modes,
        "rates": {
            "apr_current_pct": round(settings.margin_apr_current * 100, 2),
            "apr_future_pct": round(settings.margin_apr_future * 100, 2),
            "apr_future_date": settings.margin_apr_future_date,
        },
        "selected_mode": "balanced",
    }


def _trend_direction(values: list[float], threshold: float) -> str | None:
    if len(values) < 2:
        return None
    slope = np.polyfit(range(len(values)), values, 1)[0]
    if slope > threshold:
        return "rising"
    if slope < -threshold:
        return "declining"
    return "stable"


def _margin_call_distance(total_market_value, margin_loan_balance, risk: dict, threshold: float = 0.30) -> dict | None:
    if not total_market_value or not margin_loan_balance:
        return None
    margin_call_value = margin_loan_balance / (1 - threshold)
    decline_pct = (margin_call_value / total_market_value - 1.0) * 100.0
    decline_dollars = abs(total_market_value - margin_call_value)
    daily_vol_pct = None
    vol_30d = risk.get("vol_30d_pct")
    if isinstance(vol_30d, (int, float)):
        daily_vol_pct = vol_30d / math.sqrt(252)
    days_at_vol = None
    if daily_vol_pct and daily_vol_pct > 0:
        days_at_vol = abs(decline_pct) / (daily_vol_pct * 2.0)
    buffer_pct = abs(decline_pct)
    if buffer_pct > 40:
        status = "safe"
    elif buffer_pct > 20:
        status = "caution"
    else:
        status = "danger"
    return {
        "portfolio_decline_to_call_pct": _round_pct(decline_pct),
        "dollar_decline_to_call": _round_money(decline_dollars),
        "days_at_current_volatility": int(round(days_at_vol)) if isinstance(days_at_vol, (int, float)) else None,
        "buffer_status": status,
    }


def _interest_rate_scenarios(margin_loan_balance: float, current_rate_pct: float, monthly_income: float) -> dict:
    scenarios = {}
    for shock in (1.0, 2.0, 3.0):
        new_rate_pct = current_rate_pct + shock
        monthly_expense = (margin_loan_balance * new_rate_pct / 100.0) / 12.0
        annual_expense = monthly_expense * 12.0
        coverage = _safe_divide(monthly_income, monthly_expense) if monthly_expense else None
        scenarios[f"rate_plus_{int(shock * 100)}bp"] = {
            "new_rate_pct": round(new_rate_pct, 2),
            "monthly_expense": _round_money(monthly_expense),
            "annual_expense": _round_money(annual_expense),
            "income_coverage_ratio": _round_pct(coverage),
            "expense_as_pct_of_income": _round_pct(_safe_divide(monthly_expense, monthly_income) * 100 if monthly_income else None),
        }
    return scenarios


def _margin_trend_summary(daily_snaps: list[tuple[date, dict]]) -> dict | None:
    if not daily_snaps:
        return None
    ltv_vals = []
    expense_vals = []
    for _dt, snap in daily_snaps:
        totals = (snap.get("totals") or {}) if snap else {}
        mv = totals.get("market_value")
        margin = totals.get("margin_loan_balance")
        if isinstance(mv, (int, float)) and isinstance(margin, (int, float)) and mv:
            ltv_vals.append((margin / mv) * 100.0)
        if isinstance(margin, (int, float)):
            expense_vals.append((margin * settings.margin_apr_current) / 12.0)

    if not ltv_vals:
        return None

    return {
        "ltv_avg": _round_pct(sum(ltv_vals) / len(ltv_vals)),
        "ltv_max": _round_pct(max(ltv_vals)),
        "ltv_min": _round_pct(min(ltv_vals)),
        "ltv_std": _round_pct(statistics.pstdev(ltv_vals) if len(ltv_vals) > 1 else 0.0),
        "ltv_trend": _trend_direction(ltv_vals, 0.01),
        "interest_expense_avg": _round_money(sum(expense_vals) / len(expense_vals)) if expense_vals else None,
        "interest_expense_trend": _trend_direction(expense_vals, 1.0) if expense_vals else None,
    }


def _build_margin_stress(
    total_market_value: float | None,
    margin_loan_balance: float | None,
    projected_monthly_income: float | None,
    risk: dict,
    performance: dict,
    daily_snaps: list[tuple[date, dict]],
) -> dict:
    total_market_value = total_market_value or 0.0
    margin_loan_balance = margin_loan_balance or 0.0
    projected_monthly_income = projected_monthly_income or 0.0
    rate_current_pct = round(settings.margin_apr_current * 100, 2)

    monthly_interest = (margin_loan_balance * settings.margin_apr_current) / 12.0
    annual_interest = monthly_interest * 12.0

    current = {
        "ltv_pct": _round_pct(_safe_divide(margin_loan_balance, total_market_value) * 100 if total_market_value else None),
        "margin_loan_balance": _round_money(margin_loan_balance),
        "portfolio_value": _round_money(total_market_value),
        "net_liquidation_value": _round_money(total_market_value - margin_loan_balance) if total_market_value else None,
        "monthly_interest_expense": _round_money(monthly_interest),
        "annual_interest_expense": _round_money(annual_interest),
        "interest_rate_current_pct": rate_current_pct,
    }

    stress_scenarios = {
        "margin_call_distance": _margin_call_distance(total_market_value, margin_loan_balance, risk) or {},
        "interest_rate_shock": _interest_rate_scenarios(margin_loan_balance, rate_current_pct, projected_monthly_income),
    }

    roi_on_margin = None
    performance_val = performance.get("twr_12m_pct") if isinstance(performance, dict) else None
    if isinstance(performance_val, (int, float)):
        roi_on_margin = performance_val

    interest_pct_of_income = _safe_divide(monthly_interest, projected_monthly_income) * 100 if projected_monthly_income else None
    return_dollars = None
    if isinstance(performance_val, (int, float)) and total_market_value:
        return_dollars = total_market_value * (performance_val / 100.0)
    interest_pct_of_returns = _safe_divide(annual_interest, return_dollars) * 100 if return_dollars else None

    efficiency = {
        "roi_on_margin_capital_1y_pct": _round_pct(roi_on_margin),
        "interest_expense_as_pct_of_income": _round_pct(interest_pct_of_income),
        "interest_expense_as_pct_of_returns": _round_pct(interest_pct_of_returns),
        "net_benefit_of_leverage_1y_pct": _round_pct(roi_on_margin - rate_current_pct) if roi_on_margin is not None else None,
    }

    return {
        "current": current,
        "stress_scenarios": stress_scenarios,
        "efficiency": efficiency,
        "historical_trends_90d": _margin_trend_summary(daily_snaps),
    }


def _build_goal_progress(projected_monthly_income, total_market_value, target_monthly, as_of_date: date | None = None):
    if not target_monthly or target_monthly <= 0 or not total_market_value:
        return None
    portfolio_yield_pct = _safe_divide(projected_monthly_income * 12.0, total_market_value)
    portfolio_yield_pct = portfolio_yield_pct * 100 if portfolio_yield_pct is not None else None
    required = _safe_divide(target_monthly * 12.0, portfolio_yield_pct / 100 if portfolio_yield_pct else None)
    additional = required - total_market_value if required is not None else None
    progress_pct = _safe_divide(projected_monthly_income, target_monthly)
    shortfall = target_monthly - projected_monthly_income if projected_monthly_income is not None else None
    monthly_contribution = settings.goal_monthly_contribution or 0.0
    monthly_drip = projected_monthly_income or 0.0
    months_to_goal = None
    if additional is not None:
        if additional <= 0:
            months_to_goal = 0
        else:
            monthly_inflow = monthly_contribution + monthly_drip
            if monthly_inflow > 0:
                months_to_goal = int(math.ceil(additional / monthly_inflow))
    assumptions = "based_on_current_yield_with_contrib_and_drip"
    if monthly_contribution <= 0 and monthly_drip <= 0:
        assumptions = "based_on_current_yield"
    estimated_goal_date = _add_months(as_of_date, months_to_goal).isoformat() if as_of_date and months_to_goal is not None else None
    return {
        "portfolio_yield_pct": _round_pct(portfolio_yield_pct),
        "required_portfolio_value_at_goal": _round_money(required),
        "additional_investment_needed": _round_money(additional),
        "target_monthly": float(target_monthly),
        "assumptions": assumptions,
        "months_to_goal": months_to_goal,
        "progress_pct": _round_pct(progress_pct * 100 if progress_pct is not None else None),
        "growth_window_months": settings.goal_growth_window_months,
        "shortfall": _round_money(shortfall),
        "estimated_goal_date": estimated_goal_date,
        "current_projected_monthly": _round_money(projected_monthly_income),
    }


def _build_goal_progress_net(projected_monthly_income, total_market_value, margin_loan_balance, target_monthly):
    if not target_monthly or target_monthly <= 0 or not total_market_value:
        return None
    portfolio_yield_pct = _safe_divide(projected_monthly_income * 12.0, total_market_value)
    portfolio_yield_pct = portfolio_yield_pct * 100 if portfolio_yield_pct is not None else None
    annual_interest_now = (margin_loan_balance or 0.0) * settings.margin_apr_current
    annual_interest_future = (margin_loan_balance or 0.0) * settings.margin_apr_future
    required_now = _safe_divide(target_monthly * 12.0 + annual_interest_now, portfolio_yield_pct / 100 if portfolio_yield_pct else None)
    required_future = _safe_divide(target_monthly * 12.0 + annual_interest_future, portfolio_yield_pct / 100 if portfolio_yield_pct else None)
    additional_now = required_now - total_market_value if required_now is not None else None
    additional_future = required_future - total_market_value if required_future is not None else None
    monthly_interest_now = (margin_loan_balance or 0.0) * settings.margin_apr_current / 12.0
    monthly_interest_future = (margin_loan_balance or 0.0) * settings.margin_apr_future / 12.0
    projected_monthly_net = projected_monthly_income - monthly_interest_now
    progress_pct = _safe_divide(projected_monthly_net, target_monthly)
    progress_pct_future = _safe_divide(projected_monthly_income - monthly_interest_future, target_monthly)
    return {
        "portfolio_yield_pct": _round_pct(portfolio_yield_pct),
        "current_projected_monthly_net": _round_money(projected_monthly_net),
        "required_portfolio_value_at_goal_now": _round_money(required_now),
        "assumptions": "same_yield_structure; loan unchanged",
        "additional_investment_needed_now": _round_money(additional_now),
        "progress_pct": _round_pct(progress_pct * 100 if progress_pct is not None else None),
        "monthly_interest_now": _round_money(monthly_interest_now),
        "future_rate_sensitivity": {
            "apr_future_pct": round(settings.margin_apr_future * 100, 2),
            "progress_pct_future": _round_pct(progress_pct_future * 100 if progress_pct_future is not None else None),
            "additional_investment_needed_future": _round_money(additional_future),
            "required_portfolio_value_at_goal_future": _round_money(required_future),
            "monthly_interest_future": _round_money(monthly_interest_future),
        },
        "target_monthly": float(target_monthly),
    }


def _coerce_fred_series(df, series_id: str):
    if df is None or getattr(df, "empty", True):
        return None
    data = df.copy()
    cols = {str(c).lower(): c for c in data.columns}
    date_col = cols.get("date")
    value_col = cols.get("value")
    if value_col is None:
        value_col = data.columns[0] if len(data.columns) == 1 else cols.get(series_id.lower())
    if date_col is None:
        for col in data.columns:
            if "date" in str(col).lower():
                date_col = col
                break
    if date_col is None:
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index().rename(columns={"index": "date"})
            date_col = "date"
    if date_col is None or value_col is None:
        return None
    data = data.rename(columns={date_col: "date", value_col: "value"})
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["date", "value"]).sort_values("date")
    return data[["date", "value"]]


def _fred_series(md, series_id: str):
    try:
        df = md.fred.series(series_id)
    except Exception:
        return None
    return _coerce_fred_series(df, series_id)


def _series_to_pd(df):
    if df is None or getattr(df, "empty", True):
        return None
    series = df.set_index("date")["value"].astype(float)
    series.index = pd.to_datetime(series.index)
    return series.dropna().sort_index()


def _latest_value(df):
    if df is None or getattr(df, "empty", True):
        return None, None
    row = df.dropna().iloc[-1]
    return float(row["value"]), row["date"]


def _latest_series_value(series: pd.Series | None):
    if series is None or series.empty:
        return None, None
    value = float(series.iloc[-1])
    dt = series.index[-1].date()
    return value, dt


def _yoy_series(df):
    series = _series_to_pd(df)
    if series is None or series.size < 13:
        return None
    return series.pct_change(12) * 100


def _zscore_series(series: pd.Series | None):
    if series is None or series.empty:
        return None
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return series * 0.0
    return (series - mean) / std


def _trend_block(series: pd.Series | None):
    if series is None or series.empty:
        return None
    vals = series.dropna()
    if vals.empty:
        return None
    latest = float(vals.iloc[-1])
    sma_7 = float(vals.tail(7).mean())
    sma_30 = float(vals.tail(30).mean())
    return {
        "sma_7d": round(sma_7, 3),
        "sma_30d": round(sma_30, 3),
        "delta_7d": round(latest - sma_7, 3),
        "delta_30d": round(latest - sma_30, 3),
    }


def _build_macro_snapshot(md, as_of_date_local: date):
    if md is None or getattr(md, "fred", None) is None:
        return {"snapshot": None, "trends": None, "history": {"records": [], "meta": {"updated_at": now_utc_iso(), "count": 0}}, "provenance": None}

    series_ids = {
        "ten_year_yield": "DGS10",
        "two_year_yield": "DGS2",
        "fed_funds_rate": "FEDFUNDS",
        "unemployment_rate": "UNRATE",
        "m2_money_supply_bn": "M2SL",
        "hy_spread": "BAMLH0A0HYM2",
        "vix": "VIXCLS",
        "cpi": "CPIAUCSL",
        "gdp": "GDP",
    }

    series_data = {key: _fred_series(md, sid) for key, sid in series_ids.items()}
    dates = []
    values = {}
    for key, df in series_data.items():
        val, dt = _latest_value(df)
        values[key] = val
        if dt:
            dates.append(dt)

    ten_year = values.get("ten_year_yield")
    two_year = values.get("two_year_yield")
    yield_spread = ten_year - two_year if ten_year is not None and two_year is not None else None
    curve_inversion_depth = abs(yield_spread) if yield_spread is not None and yield_spread < 0 else 0.0 if yield_spread is not None else None

    cpi_yoy_series = _yoy_series(series_data.get("cpi"))
    cpi_yoy, cpi_date = _latest_series_value(cpi_yoy_series)
    if cpi_date:
        dates.append(cpi_date)

    real_10y = ten_year - cpi_yoy if ten_year is not None and cpi_yoy is not None else None
    m2_yoy_series = _yoy_series(series_data.get("m2_money_supply_bn"))

    # Macro composite series
    vix_series = _series_to_pd(series_data.get("vix"))
    hy_series = _series_to_pd(series_data.get("hy_spread"))
    stress_series = _zscore_series(vix_series)
    if stress_series is not None and hy_series is not None and not hy_series.empty:
        hy_z = _zscore_series(hy_series)
        if hy_z is not None:
            aligned = pd.concat([stress_series, hy_z], axis=1).dropna()
            if not aligned.empty:
                stress_series = aligned.mean(axis=1)
    macro_stress_score, stress_date = _latest_series_value(stress_series)
    if stress_date:
        dates.append(stress_date)

    real_yield_series = None
    dgs10_series = _series_to_pd(series_data.get("ten_year_yield"))
    if dgs10_series is not None and not dgs10_series.empty and cpi_yoy_series is not None and not cpi_yoy_series.empty:
        cpi_aligned = cpi_yoy_series.reindex(dgs10_series.index, method="ffill")
        real_yield_series = dgs10_series - cpi_aligned
    valuation_series = None
    if real_yield_series is not None:
        valuation_series = _zscore_series(real_yield_series) * -1
    valuation_heat_index, valuation_date = _latest_series_value(valuation_series)
    if valuation_date:
        dates.append(valuation_date)

    liquidity_series = _zscore_series(m2_yoy_series)
    liquidity_index, liquidity_date = _latest_series_value(liquidity_series)
    if liquidity_date:
        dates.append(liquidity_date)

    snapshot_date = max(dates) if dates else as_of_date_local
    fetched_at = now_utc_iso()

    snapshot = {
        "date": snapshot_date.isoformat(),
        "m2_money_supply_bn": _round_money(values.get("m2_money_supply_bn")),
        "vix": _round_money(values.get("vix")),
        "unemployment_rate": _round_money(values.get("unemployment_rate")),
        "gdp_nowcast": _round_money(values.get("gdp")),
        "yield_spread_10y_2y": _round_money(yield_spread),
        "ten_year_yield": _round_money(ten_year),
        "hy_spread_bps": _round_money(values.get("hy_spread") * 100 if values.get("hy_spread") is not None else None),
        "cpi_yoy": _round_money(cpi_yoy),
        "curve_inversion_depth": _round_money(curve_inversion_depth),
        "fed_funds_rate": _round_money(values.get("fed_funds_rate")),
        "two_year_yield": _round_money(two_year),
        "real_10y_yield": _round_money(real_10y),
        "macro_stress_score": _round_pct(macro_stress_score),
        "valuation_heat_index": _round_pct(valuation_heat_index),
        "liquidity_index": _round_pct(liquidity_index),
        "meta": {
            "fetched_at": fetched_at,
            "source": "fred",
            "schema_version": "1.0",
            "series_ids": {key: sid for key, sid in series_ids.items()},
        },
    }

    trends = {
        "date": snapshot_date.isoformat(),
        "trends": {
            "macro_stress_score": _trend_block(stress_series),
            "valuation_heat_index": _trend_block(valuation_series),
            "liquidity_index": _trend_block(liquidity_series),
        },
        "meta": {"computed_at": now_utc_iso(), "window_days": 30, "schema_version": "1.0"},
    }

    history_records = []
    if snapshot_date and macro_stress_score is not None and valuation_heat_index is not None and liquidity_index is not None:
        history_records.append(
            {
                "date": snapshot_date.isoformat(),
                "valuation_heat_index": _round_pct(valuation_heat_index),
                "real_10y_yield": _round_money(real_10y),
                "curve_inversion_depth": _round_money(curve_inversion_depth),
                "macro_stress_score": _round_pct(macro_stress_score),
                "liquidity_index": _round_pct(liquidity_index),
            }
        )

    history = {"records": history_records, "meta": {"updated_at": now_utc_iso(), "count": len(history_records)}}
    provenance = {"fetched_at": fetched_at, "schema_version": "1.0", "source": "fred"}
    return {"snapshot": snapshot, "trends": trends, "history": history, "provenance": provenance}


def build_daily_snapshot(conn: sqlite3.Connection, holdings: dict, md) -> tuple[dict, list[dict]]:
    dt_utc = datetime.now(timezone.utc)
    as_of_utc = dt_utc.isoformat()
    as_of_local = to_local_datetime_iso(dt_utc, settings.local_tz)
    as_of_date_local = to_local_date(dt_utc, settings.local_tz, settings.daily_cutover)
    as_of_date_str = as_of_date_local.isoformat()

    cur = conn.cursor()
    trade_counts = _load_trade_counts(conn)
    div_tx = _load_dividend_transactions(conn)
    pay_history = _build_pay_history(div_tx)
    provider_divs = _load_provider_dividends(conn)
    lm_ex_dates = _load_lm_ex_dates(conn)
    account_balances = _load_account_balances(conn)

    holding_symbols = set(holdings.keys())
    div_by_symbol = defaultdict(list)
    for tx in div_tx:
        div_by_symbol[tx["symbol"]].append(tx)

    price_series = {}
    missing_prices = []
    prices_as_of = None
    for sym in holdings:
        series = _price_series(md.prices.get(sym))
        price_series[sym] = series
        if series is None or series.empty:
            missing_prices.append(sym)
        else:
            last_date = series.index.max().date()
            if prices_as_of is None or last_date > prices_as_of:
                prices_as_of = last_date

    benchmark_symbol = settings.benchmark_primary
    benchmark_series = _price_series(md.prices.get(benchmark_symbol))

    holdings_out = []
    sources = []
    total_market_value = 0.0
    total_cost_basis = 0.0
    total_forward_12m = 0.0
    ex_dates_by_symbol = {}

    for sym in sorted(holdings.keys()):
        h = holdings[sym]
        series = price_series.get(sym)
        last_price = None
        if series is not None and not series.empty:
            last_price = float(series.iloc[-1])
        mv = last_price * float(h["shares"]) if last_price is not None else None
        total_market_value += mv or 0.0
        total_cost_basis += float(h["cost_basis"]) if h.get("cost_basis") is not None else 0.0

        div_events = provider_divs.get(sym, [])
        ex_dates = _effective_ex_dates(sym, provider_divs, lm_ex_dates)
        trailing_12m_div_ps = sum(
            ev["amount"]
            for ev in div_events
            if ev["ex_date"] >= as_of_date_local - timedelta(days=365)
        )
        last_ex_date = ex_dates[-1] if ex_dates else None
        frequency = _frequency_from_ex_dates(ex_dates)
        next_ex_date_est = _next_ex_date_est(ex_dates)
        if next_ex_date_est is None and last_ex_date:
            next_ex_date_est = _fallback_next_ex_date(last_ex_date, pay_history.get(sym, []))
        ex_dates_by_symbol[sym] = ex_dates
        forward_12m_div_ps = trailing_12m_div_ps if trailing_12m_div_ps else 0.0
        forward_method = "t12m" if trailing_12m_div_ps else None
        forward_12m_dividend = forward_12m_div_ps * float(h["shares"])
        total_forward_12m += forward_12m_dividend

        unrealized_pnl = mv - float(h["cost_basis"]) if mv is not None and h.get("cost_basis") is not None else None
        unrealized_pct = _safe_divide(unrealized_pnl, float(h["cost_basis"])) if unrealized_pnl is not None else None
        avg_cost = _safe_divide(float(h["cost_basis"]), float(h["shares"])) if h.get("cost_basis") is not None else None

        div_30d = sum(ev["amount"] for ev in div_by_symbol.get(sym, []) if ev["date"] >= as_of_date_local - timedelta(days=30))
        div_qtd = sum(ev["amount"] for ev in div_by_symbol.get(sym, []) if ev["date"] >= _quarter_start(as_of_date_local))
        div_ytd = sum(ev["amount"] for ev in div_by_symbol.get(sym, []) if ev["date"] >= _year_start(as_of_date_local))

        metrics_out = _symbol_metrics(series, benchmark_series) if series is not None else {}
        trailing_12m_yield = _safe_divide(trailing_12m_div_ps, last_price) if last_price else None
        forward_yield = _safe_divide(forward_12m_div_ps, last_price) if last_price else None

        ultimate = {
            "trailing_12m_div_ps": _round_money(trailing_12m_div_ps) if trailing_12m_div_ps else None,
            "trailing_12m_yield_pct": _round_pct(trailing_12m_yield * 100 if trailing_12m_yield else None),
            "forward_yield_pct": _round_pct(forward_yield * 100 if forward_yield else None),
            "forward_yield_method": forward_method,
            "distribution_frequency": frequency,
            "next_ex_date_est": next_ex_date_est.isoformat() if next_ex_date_est else None,
            "derived_fields": [
                "distribution_frequency",
                "forward_12m_div_ps",
                "forward_yield_pct",
                "next_ex_date_est",
                "trailing_12m_yield_pct",
            ],
            "last_ex_date": last_ex_date.isoformat() if last_ex_date else None,
            "forward_12m_div_ps": _round_money(forward_12m_div_ps) if forward_12m_div_ps else None,
        }
        ultimate.update(metrics_out)

        fetched_at = now_utc_iso()
        provider_name = "unknown"
        if md.provenance.get(sym):
            for entry in md.provenance[sym]:
                if entry.get("success"):
                    provider_name = entry.get("provider", "unknown")
                    break
            if provider_name == "unknown":
                provider_name = md.provenance[sym][0].get("provider", "unknown")
        ultimate_provenance = {
            "price_history": _provenance_entry("pulled", provider_name, "price_history", [sym], fetched_at),
            "benchmark_history": _provenance_entry("pulled", provider_name, "price_history", [benchmark_symbol], fetched_at),
            "last_price": _provenance_entry("pulled", provider_name, "price_history_last", [sym], fetched_at),
            "trailing_12m_div_ps": _provenance_entry("derived", "internal", "ttm_dividends", [sym], fetched_at),
            "last_ex_date": _provenance_entry("derived", "internal", "dividend_history", [sym], fetched_at),
            "distribution_frequency": _provenance_entry("derived", "internal", "freq_from_exdate_gaps", [sym], fetched_at),
            "next_ex_date_est": _provenance_entry("derived", "internal", "median_gap_projection", [sym], fetched_at),
            "forward_12m_div_ps": _provenance_entry("derived", "internal", "forward_equals_ttm", [sym], fetched_at),
            "trailing_12m_yield_pct": _provenance_entry("derived", "internal", "ttm_yield", [sym, "last_price"], fetched_at),
            "forward_yield_pct": _provenance_entry("derived", "internal", "forward_yield", [sym, "last_price"], fetched_at),
        }
        for key in [
            "vol_30d_pct",
            "vol_90d_pct",
            "max_drawdown_1y_pct",
            "drawdown_duration_1y_days",
            "sharpe_1y",
            "sortino_1y",
            "sortino_6m",
            "sortino_3m",
            "sortino_1m",
            "downside_dev_1y_pct",
            "calmar_1y",
            "risk_quality_score",
            "risk_quality_category",
            "volatility_profile",
            "var_90_1d_pct",
            "var_95_1d_pct",
            "var_99_1d_pct",
            "var_95_1w_pct",
            "var_95_1m_pct",
            "cvar_90_1d_pct",
            "cvar_95_1d_pct",
            "cvar_99_1d_pct",
            "cvar_95_1w_pct",
            "cvar_95_1m_pct",
            "corr_1y",
            "beta_3y",
            "twr_1m_pct",
            "twr_3m_pct",
            "twr_6m_pct",
            "twr_12m_pct",
        ]:
            ultimate_provenance[key] = _provenance_entry("derived", "internal", "price_history_metrics", [sym], fetched_at)

        dividend_reliability = _dividend_reliability_metrics(
            sym,
            div_tx,
            provider_divs,
            pay_history,
            frequency,
            as_of_date_local,
        )
        dividend_reliability_provenance = {
            key: _provenance_entry("derived", "internal", "dividend_reliability", [sym], fetched_at)
            for key in dividend_reliability
        }

        holding = {
            "symbol": sym,
            "shares": float(h["shares"]),
            "cost_basis": _round_money(h["cost_basis"]),
            "avg_cost": _round_money(avg_cost),
            "trades": trade_counts.get(sym, 0),
            "last_price": _round_money(last_price),
            "market_value": _round_money(mv),
            "unrealized_pnl": _round_money(unrealized_pnl),
            "unrealized_pct": _round_pct(unrealized_pct * 100 if unrealized_pct is not None else None),
            "weight_pct": None,
            "forward_12m_dividend": _round_money(forward_12m_dividend),
            "last_ex_date": last_ex_date.isoformat() if last_ex_date else None,
            "forward_method": forward_method,
            "projected_monthly_dividend": _round_money(_safe_divide(forward_12m_dividend, 12.0)),
            "current_yield_pct": _round_pct(_safe_divide(forward_12m_dividend, mv) * 100 if mv is not None else None),
            "yield_on_cost_pct": _round_pct(_safe_divide(forward_12m_dividend, h["cost_basis"]) * 100 if h.get("cost_basis") else None),
            "ultimate": ultimate,
            "dividend_reliability": dividend_reliability,
            "dividends_30d": _round_money(div_30d),
            "dividends_qtd": _round_money(div_qtd),
            "dividends_ytd": _round_money(div_ytd),
            "ultimate_provenance": ultimate_provenance,
            "dividend_reliability_provenance": dividend_reliability_provenance,
        }
        holdings_out.append(holding)

        for i, p in enumerate(md.provenance.get(sym, []), start=1):
            sources.append(
                {
                    "as_of_date_local": as_of_date_str,
                    "scope": "symbol",
                    "symbol": sym,
                    "field_path": "holdings[].last_price",
                    "value_text": None,
                    "value_num": last_price if i == 1 else None,
                    "value_int": None,
                    "value_type": "num",
                    "source_type": "market",
                    "provider": p.get("provider", "unknown"),
                    "endpoint": p.get("endpoint"),
                    "params": p.get("params") or {},
                    "source_ref": None,
                    "commit_sha": None,
                    "provider_rank": i,
                }
            )

    # weights
    for holding in holdings_out:
        mv = holding.get("market_value")
        holding["weight_pct"] = _round_pct(
            _safe_divide(mv, total_market_value) * 100 if mv is not None and total_market_value else None
        )

    margin_loan_balance = 0.0
    for account in account_balances:
        if _is_margin_account(account.get("name"), account.get("type"), account.get("subtype")):
            try:
                margin_loan_balance += abs(float(account.get("balance") or 0.0))
            except (TypeError, ValueError):
                pass
    net_liquidation_value = None
    if isinstance(total_market_value, (int, float)) and isinstance(margin_loan_balance, (int, float)):
        net_liquidation_value = total_market_value - margin_loan_balance

    totals = {
        "cost_basis": _round_money(total_cost_basis),
        "margin_loan_balance": _round_money(margin_loan_balance),
        "market_value": _round_money(total_market_value),
        "net_liquidation_value": _round_money(net_liquidation_value),
        "margin_to_portfolio_pct": _round_pct(_safe_divide(margin_loan_balance, total_market_value) * 100 if total_market_value else None),
        "unrealized_pnl": _round_money(total_market_value - total_cost_basis),
        "unrealized_pct": _round_pct(_safe_divide(total_market_value - total_cost_basis, total_cost_basis) * 100 if total_cost_basis else None),
    }

    income = {
        "projected_monthly_income": _round_money(_safe_divide(total_forward_12m, 12.0)),
        "forward_12m_total": _round_money(total_forward_12m),
        "portfolio_current_yield_pct": _round_pct(_safe_divide(total_forward_12m, total_market_value) * 100 if total_market_value else None),
        "portfolio_yield_on_cost_pct": _round_pct(_safe_divide(total_forward_12m, total_cost_basis) * 100 if total_cost_basis else None),
    }

    # dividend windows
    window_30d = _dividend_window(div_tx, as_of_date_local - timedelta(days=30), as_of_date_local, holding_symbols)
    window_ytd = _dividend_window(div_tx, _year_start(as_of_date_local), as_of_date_local, holding_symbols)
    window_qtd = _dividend_window(div_tx, _quarter_start(as_of_date_local), as_of_date_local, holding_symbols)
    realized_mtd = _dividend_window(div_tx, _month_start(as_of_date_local), as_of_date_local, holding_symbols)

    projected_monthly_income = income["projected_monthly_income"] or 0.0
    received_mtd = realized_mtd["total_dividends"]
    projected_vs_received = {
        "pct_of_projection": _round_pct(_safe_divide(received_mtd, projected_monthly_income) * 100 if projected_monthly_income else None),
        "projected": _round_money(projected_monthly_income),
        "received": _round_money(received_mtd),
        "difference": _round_money(projected_monthly_income - received_mtd),
        "window": {
            "start": _month_start(as_of_date_local).isoformat(),
            "end": as_of_date_local.isoformat(),
            "label": "month_to_date",
        },
        "alt": {
            "projected": 0.0,
            "mode": "paydate",
            "window": {"start": _month_start(as_of_date_local).isoformat(), "end": as_of_date_local.isoformat()},
            "expected_events": [],
        },
    }

    ex_date_est_by_symbol = {}
    for holding in holdings_out:
        sym = holding.get("symbol")
        if not sym:
            continue
        next_ex = (holding.get("ultimate") or {}).get("next_ex_date_est")
        next_dt = _parse_date(next_ex)
        if next_dt:
            ex_date_est_by_symbol[sym] = next_dt

    expected_events, projected_alt = _estimate_expected_pay_events(
        provider_divs,
        holdings,
        _month_start(as_of_date_local),
        as_of_date_local,
        pay_history,
        ex_date_est_by_symbol,
    )
    projected_vs_received["alt"]["projected"] = projected_alt or 0.0
    projected_vs_received["alt"]["expected_events"] = expected_events

    dividends = {
        "projected_vs_received": projected_vs_received,
        "windows": {"30d": window_30d, "ytd": window_ytd, "qtd": window_qtd},
        "realized_mtd": realized_mtd,
    }

    # dividends upcoming
    window_start = as_of_date_local
    window_end = _month_end(as_of_date_local)
    upcoming_events = []
    for sym in sorted(holding_symbols):
        events = [ev for ev in provider_divs.get(sym, []) if window_start <= ev["ex_date"] <= window_end]
        if not events:
            ex_dates = ex_dates_by_symbol.get(sym) or _effective_ex_dates(sym, provider_divs, lm_ex_dates)
            next_est = _next_ex_date_est(ex_dates)
            if next_est is None and ex_dates:
                next_est = _fallback_next_ex_date(ex_dates[-1], pay_history.get(sym, []))
            if next_est and window_start <= next_est <= window_end:
                per_share, _freq = _estimate_upcoming_per_share(provider_divs.get(sym, []), as_of_date_local)
                if per_share is not None:
                    events = [{"ex_date": next_est, "amount": per_share}]
        for ev in events:
            shares = float(holdings[sym]["shares"])
            amount_est = ev["amount"] * shares
            upcoming_events.append(
                {"symbol": sym, "ex_date_est": ev["ex_date"].isoformat(), "amount_est": _round_money(amount_est)}
            )

    projected_upcoming = sum(ev["amount_est"] for ev in upcoming_events if ev.get("amount_est") is not None)
    dividends_upcoming = {
        "projected": _round_money(projected_upcoming),
        "events": upcoming_events,
        "window": {"start": window_start.isoformat(), "end": window_end.isoformat(), "mode": "exdate"},
        "meta": {
            "matches_projected": True,
            "sum_of_events": _round_money(projected_upcoming),
        },
    }

    portfolio_values = _portfolio_value_series(holdings, price_series)
    bench_values = benchmark_series if benchmark_series is not None else None
    performance = metrics.portfolio_performance(portfolio_values)
    risk = metrics.portfolio_risk(portfolio_values, bench_values)
    risk["portfolio_risk_quality"] = _risk_quality_category(risk.get("sortino_1y"), risk.get("sharpe_1y"))
    income_stability = _income_stability_metrics(div_tx, as_of_date_local, window_months=12)
    income_growth = _income_growth_metrics(div_tx, as_of_date_local)
    risk["income_stability_score"] = income_stability.get("stability_score")

    drawdown_status, recovery_metrics = _drawdown_analysis(portfolio_values)
    if drawdown_status:
        risk["drawdown_status"] = drawdown_status
    if recovery_metrics:
        risk["recovery_metrics"] = recovery_metrics

    ulcer_val = risk.get("ulcer_index_1y")
    risk["ulcer_index_category"] = _ulcer_index_category(ulcer_val)
    pain_adjusted_return = _safe_divide(performance.get("twr_12m_pct"), ulcer_val)
    risk["pain_adjusted_return"] = _round_pct(pain_adjusted_return)
    risk["omega_threshold"] = 0.0

    tail_risk = {
        "cvar_95_1d_pct": risk.get("cvar_95_1d_pct"),
        "cvar_95_1w_pct": risk.get("cvar_95_1w_pct"),
        "cvar_95_1m_pct": risk.get("cvar_95_1m_pct"),
        "cvar_90_1d_pct": risk.get("cvar_90_1d_pct"),
        "cvar_99_1d_pct": risk.get("cvar_99_1d_pct"),
        "var_95_1d_pct": risk.get("var_95_1d_pct"),
        "var_95_1w_pct": risk.get("var_95_1w_pct"),
        "var_95_1m_pct": risk.get("var_95_1m_pct"),
        "cvar_to_income_ratio": None,
        "worst_expected_loss_1w": None,
    }
    if isinstance(total_market_value, (int, float)):
        cvar_1w = risk.get("cvar_95_1w_pct")
        if isinstance(cvar_1w, (int, float)):
            tail_risk["worst_expected_loss_1w"] = _round_money(total_market_value * abs(cvar_1w) / 100.0)
        cvar_1m = risk.get("cvar_95_1m_pct")
        projected_income = income.get("projected_monthly_income") or 0.0
        if isinstance(cvar_1m, (int, float)) and projected_income:
            expected_loss = total_market_value * abs(cvar_1m) / 100.0
            tail_risk["cvar_to_income_ratio"] = _round_pct(_safe_divide(expected_loss, projected_income))
        else:
            tail_risk["cvar_to_income_ratio"] = None
    tail_risk["tail_risk_category"] = _tail_risk_category(risk.get("cvar_95_1d_pct"))

    bench_twr_12m = metrics.twr(bench_values, window_days=365) if bench_values is not None else None
    bench_twr_12m_pct = bench_twr_12m * 100 if bench_twr_12m is not None else None
    excess_return = None
    if isinstance(performance.get("twr_12m_pct"), (int, float)) and isinstance(bench_twr_12m_pct, (int, float)):
        excess_return = performance.get("twr_12m_pct") - bench_twr_12m_pct
    vs_benchmark = {
        "benchmark": benchmark_symbol,
        "excess_return_1y_pct": _round_pct(excess_return),
        "tracking_error_1y_pct": risk.get("tracking_error_1y_pct"),
        "information_ratio_1y": risk.get("information_ratio_1y"),
        "active_share_pct": 100.0,
        "correlation_to_benchmark": risk.get("corr_1y"),
    }

    end_dt = portfolio_values.index.max() if not portfolio_values.empty else None
    return_attribution_rollups = {}
    return_attribution_by_symbol = {}
    if end_dt is not None:
        return_attribution_rollups, return_attribution_by_symbol = _return_attribution_all_periods(
            holdings,
            price_series,
            div_tx,
            end_dt,
        )
        for holding in holdings_out:
            sym = holding.get("symbol")
            if not sym:
                continue
            per_period = return_attribution_by_symbol.get(sym, {})
            for label, metrics_out in per_period.items():
                holding[f"contribution_analysis_{label}"] = metrics_out

    portfolio_rollups = {
        "performance": performance,
        "benchmark": benchmark_symbol,
        "risk": risk,
        "income_stability": income_stability,
        "income_growth": income_growth,
        "tail_risk": tail_risk,
        "vs_benchmark": vs_benchmark,
        "meta": {"version": f"v{as_of_date_str}", "method": "approx-holdings"},
    }
    portfolio_rollups.update(return_attribution_rollups)

    totals_prov = {}
    for key in ["cost_basis", "margin_loan_balance", "market_value", "net_liquidation_value", "margin_to_portfolio_pct", "unrealized_pnl", "unrealized_pct"]:
        totals_prov[key] = _provenance_entry("derived", "internal", "recompute_totals", ["holdings"], now_utc_iso())

    income_prov = {}
    for key in ["projected_monthly_income", "forward_12m_total", "portfolio_current_yield_pct", "portfolio_yield_on_cost_pct"]:
        income_prov[key] = _provenance_entry("derived", "internal", "income_from_holdings", ["holdings"], now_utc_iso())

    portfolio_rollups_prov = {
        "performance": {k: _provenance_entry("derived", "internal", "compute_portfolio_rollups", ["holdings"], now_utc_iso()) for k in performance},
        "risk": {k: _provenance_entry("derived", "internal", "compute_portfolio_rollups", ["holdings"], now_utc_iso()) for k in risk},
    }
    portfolio_rollups_prov["risk"]["income_stability_score"] = _provenance_entry(
        "derived",
        "internal",
        "income_stability_metrics",
        ["dividend_transactions", "window_months:12"],
        now_utc_iso(),
    )
    portfolio_rollups_prov["income_stability"] = {k: _provenance_entry("derived", "internal", "income_stability_metrics", ["dividend_transactions"], now_utc_iso()) for k in income_stability}
    portfolio_rollups_prov["income_growth"] = {k: _provenance_entry("derived", "internal", "income_growth_metrics", ["dividend_transactions"], now_utc_iso()) for k in income_growth}
    portfolio_rollups_prov["tail_risk"] = {k: _provenance_entry("derived", "internal", "tail_risk_metrics", ["portfolio_rollups", "income"], now_utc_iso()) for k in tail_risk}
    portfolio_rollups_prov["vs_benchmark"] = {k: _provenance_entry("derived", "internal", "benchmark_comparison", ["portfolio_values", "benchmark_values"], now_utc_iso()) for k in vs_benchmark}
    for key in return_attribution_rollups:
        portfolio_rollups_prov[key] = _provenance_entry("derived", "internal", "return_attribution", ["holdings", "dividends", "prices"], now_utc_iso())

    income_provenance = income_prov
    holdings_provenance = _provenance_entry("derived", "internal", "reconstruct_holdings", ["investment_transactions"], now_utc_iso())
    dividends_provenance = _provenance_entry("derived", "internal", "normalize_dividends", ["transactions"], now_utc_iso())
    dividends_upcoming_provenance = _provenance_entry("derived", "internal", "project_upcoming_dividends", ["holdings"], now_utc_iso())
    margin_guidance = _build_margin_guidance(total_market_value, margin_loan_balance, income["projected_monthly_income"] or 0.0)
    margin_guidance_provenance = _provenance_entry("derived", "internal", "margin_guidance_calculator", ["totals", "income"], now_utc_iso())

    history_start = as_of_date_local - timedelta(days=89)
    recent_snaps = _load_daily_snapshots_range(conn, history_start, as_of_date_local)
    margin_stress = _build_margin_stress(
        total_market_value,
        margin_loan_balance,
        income["projected_monthly_income"] or 0.0,
        risk,
        performance,
        recent_snaps,
    )
    margin_stress_provenance = _provenance_entry(
        "derived",
        "internal",
        "margin_stress_analysis",
        ["totals", "income", "portfolio_rollups", "daily_snapshots"],
        now_utc_iso(),
    )

    goal_progress = _build_goal_progress(
        income["projected_monthly_income"] or 0.0,
        total_market_value,
        settings.goal_target_monthly,
        as_of_date_local,
    )
    goal_progress_net = _build_goal_progress_net(income["projected_monthly_income"] or 0.0, total_market_value, margin_loan_balance, settings.goal_target_monthly)
    goal_progress_provenance = _provenance_entry(
        "derived",
        "internal",
        "goal_progress_projection",
        ["income", "goal_settings", "monthly_contribution", "as_of_date_local"],
        now_utc_iso(),
    )

    missing_paths = []
    for idx, sym in enumerate(sorted(missing_prices)):
        missing_paths.append(f"holdings[{idx}].last_price")

    missing_pct = _safe_divide(len(missing_prices), len(holding_symbols)) or 0.0
    coverage = {
        "derived_pct": _round_pct(100.0 - (missing_pct * 100)),
        "validated_pct": 0.0,
        "pulled_pct": _round_pct(missing_pct * 100),
        "filled_pct": _round_pct(100.0 - (missing_pct * 100)),
        "missing_pct": _round_pct(missing_pct * 100),
        "missing_paths": missing_paths,
        "conflict_paths": [],
    }

    last_run = cur.execute(
        "SELECT finished_at_utc FROM runs WHERE status='succeeded' ORDER BY finished_at_utc DESC LIMIT 1"
    ).fetchone()
    last_sync = last_run[0] if last_run else None

    meta = {
        "snapshot_created_at": now_utc_iso(),
        "price_partial": bool(missing_prices),
        "source": "db",
        "last_transaction_sync_at": last_sync,
        "cache_control": {"revalidate": "when-stale", "no_store": False},
        "served_from": "db",
        "notes": [],
        "changes": ["Expanded portfolio metrics (income stability, tail risk, attribution, margin stress)"],
        "cache": {
            "pricing": {"ttl_seconds": settings.cache_ttl_hours * 3600, "bypassed": False},
            "yf_dividends": {"ttl_seconds": settings.cache_ttl_hours * 3600, "bypassed": False},
        },
        "snapshot_age_days": 0,
        "schema_version": "3.0",
        "filled_from_existing": False,
    }

    plaid_account_id = None
    if settings.lm_plaid_account_ids:
        try:
            plaid_account_id = int(settings.lm_plaid_account_ids.split(",")[0].strip())
        except ValueError:
            plaid_account_id = settings.lm_plaid_account_ids.split(",")[0].strip()

    macro = _build_macro_snapshot(md, as_of_date_local)

    daily = {
        "as_of": as_of_date_str,
        "as_of_utc": as_of_utc,
        "as_of_date_local": as_of_date_str,
        "plaid_account_id": plaid_account_id,
        "prices_as_of": prices_as_of.isoformat() if prices_as_of else as_of_date_str,
        "holdings": holdings_out,
        "count": len(holdings_out),
        "missing_prices": missing_prices,
        "totals": totals,
        "total_market_value": totals["market_value"],
        "income": income,
        "dividends": dividends,
        "dividends_upcoming": dividends_upcoming,
        "portfolio_rollups": portfolio_rollups,
        "goal_progress": goal_progress,
        "goal_progress_net": goal_progress_net,
        "margin_guidance": margin_guidance,
        "margin_stress": margin_stress,
        "coverage": coverage,
        "holdings_provenance": holdings_provenance,
        "totals_provenance": totals_prov,
        "income_provenance": income_provenance,
        "portfolio_rollups_provenance": portfolio_rollups_prov,
        "dividends_provenance": dividends_provenance,
        "dividends_upcoming_provenance": dividends_upcoming_provenance,
        "margin_guidance_provenance": margin_guidance_provenance,
        "margin_stress_provenance": margin_stress_provenance,
        "goal_progress_provenance": goal_progress_provenance,
        "macro": macro,
        "meta": meta,
        "cached": False,
    }
    return daily, sources


def persist_daily_snapshot(conn: sqlite3.Connection, daily: dict, run_id: str, force: bool = False) -> bool:
    as_of_date_local = daily.get("as_of_date_local") or daily["as_of"][:10]
    payload_sha = sha256_json(daily)
    cur = conn.cursor()
    if not force:
        existing = cur.execute(
            "SELECT payload_sha256 FROM snapshot_daily_current WHERE as_of_date_local=?",
            (as_of_date_local,),
        ).fetchone()
        if existing and existing[0] == payload_sha:
            return False
    cur.execute(
        """
        INSERT OR REPLACE INTO snapshot_daily_current(as_of_date_local, built_from_run_id, payload_json, payload_sha256, updated_at_utc)
        VALUES(?,?,?,?,?)
        """,
        (as_of_date_local, run_id, json.dumps(daily), payload_sha, now_utc_iso()),
    )
    conn.commit()
    return True


def _period_bounds(period_type: str, on: date):
    if period_type == "WEEK":
        start = on - timedelta(days=on.weekday())
        end = start + timedelta(days=6)
    elif period_type == "MONTH":
        start = on.replace(day=1)
        end = (start.replace(year=start.year + 1, month=1, day=1) - timedelta(days=1)) if start.month == 12 else (start.replace(month=start.month + 1, day=1) - timedelta(days=1))
    elif period_type == "QUARTER":
        q = (on.month - 1) // 3
        start_month = 1 + q * 3
        start = date(on.year, start_month, 1)
        end = (date(on.year + 1, 1, 1) - timedelta(days=1)) if start_month == 10 else (date(on.year, start_month + 3, 1) - timedelta(days=1))
    elif period_type == "YEAR":
        start = date(on.year, 1, 1)
        end = date(on.year, 12, 31)
    else:
        raise ValueError("bad period")
    return start, end


def maybe_persist_periodic(conn: sqlite3.Connection, run_id: str, daily: dict):
    as_of_date_local = daily.get("as_of_date_local") or daily["as_of"][:10]
    dt = date.fromisoformat(as_of_date_local)
    for period in ["WEEK", "MONTH", "QUARTER", "YEAR"]:
        start, end = _period_bounds(period, dt)
        if dt != end:
            continue
        try:
            from .periods import build_period_snapshot
            snapshot_type = {
                "WEEK": "weekly",
                "MONTH": "monthly",
                "QUARTER": "quarterly",
                "YEAR": "yearly",
            }.get(period)
            if not snapshot_type:
                raise ValueError(f"unknown period {period}")
            snapshot = build_period_snapshot(conn, snapshot_type=snapshot_type, as_of=str(end), mode="final")
        except Exception as exc:
            log.error("period_snapshot_build_failed", run_id=run_id, period=period, err=str(exc))
            continue
        ok, reasons = validate_period_snapshot(snapshot)
        if not ok:
            log.error("period_snapshot_invalid", run_id=run_id, period=period, reasons=reasons)
            continue
        payload_sha = sha256_json(snapshot)
        cur = conn.cursor()
        existing = cur.execute(
            """
            SELECT payload_sha256
            FROM snapshots
            WHERE period_type=? AND period_start_date=? AND period_end_date=?
            """,
            (period, str(start), str(end)),
        ).fetchone()
        if existing and existing[0] == payload_sha:
            continue
        cur.execute(
            """
            INSERT OR REPLACE INTO snapshots(snapshot_id, period_type, period_start_date, period_end_date, built_from_run_id, payload_json, payload_sha256, created_at_utc)
            VALUES(?,?,?,?,?,?,?,?)
            """,
            (str(uuid.uuid4()), period, str(start), str(end), run_id, json.dumps(snapshot), payload_sha, now_utc_iso()),
        )
        conn.commit()


def diff_payloads(a: dict, b: dict):
    def _strip_provenance(obj):
        if isinstance(obj, dict):
            return {k: _strip_provenance(v) for k, v in obj.items() if "provenance" not in str(k).lower()}
        if isinstance(obj, list):
            return [_strip_provenance(v) for v in obj]
        return obj

    a = _strip_provenance(a)
    b = _strip_provenance(b)
    out = {}
    keys = set(a.keys()) | set(b.keys())
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            sub = diff_payloads(va, vb)
            if sub:
                out[k] = sub
        elif va != vb:
            out[k] = {"left": va, "right": vb}
    return out
log = structlog.get_logger()

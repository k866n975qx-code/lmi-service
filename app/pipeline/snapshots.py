import calendar
import json
import math
import re
import sqlite3
import statistics
import uuid
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ..config import settings
import structlog
from ..utils import sha256_json, now_utc_iso, to_local_date, to_local_datetime_iso
from .validation import validate_period_snapshot
from . import metrics

TRADE_TYPES = {"buy", "buy_shares", "reinvest", "reinvestment", "sell", "sell_shares", "redemption"}
ACQUIRE_TYPES = {"buy", "buy_shares", "reinvest", "reinvestment"}
SELL_TYPES = {"sell", "sell_shares", "redemption"}
SHARE_TX_TYPES = ACQUIRE_TYPES | SELL_TYPES
_SHARES_RE = re.compile(r"\b([0-9]*\.?[0-9]+)\s+shares?\b", re.IGNORECASE)
DIVIDEND_CUT_THRESHOLD = 0.10
DIVIDEND_CUT_LOOKAHEAD_DAYS = 90
SPECIAL_DIVIDEND_FACTOR = 2.5
DIVIDEND_PAY_MATCH_WINDOW_DAYS = 3


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


def _last_business_day(d: date) -> date:
    """Return the last business day (Mon-Fri) of the month containing *d*."""
    me = _month_end(d)
    while me.weekday() >= 5:
        me -= timedelta(days=1)
    return me


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
    return _frequency_from_gap_days(median_gap)


def _frequency_from_gap_days(gap_days: int | None) -> str | None:
    if gap_days is None:
        return None
    if gap_days < 45:
        return "monthly"
    if gap_days < 100:
        return "quarterly"
    if gap_days < 190:
        return "semiannual"
    return "annual"


def _frequency_from_recent_ex_dates(ex_dates: list[date], recent: int = 6) -> str | None:
    if len(ex_dates) < 2:
        return None
    dates = sorted(ex_dates)
    if recent and len(dates) > recent:
        dates = dates[-recent:]
    return _frequency_from_ex_dates(dates)


def _frequency_from_dates(dates: list[date]) -> str | None:
    if not dates:
        return None
    dates = sorted(dates)
    return _frequency_from_ex_dates(dates)


def _gap_days(dates: list[date]) -> list[int]:
    if len(dates) < 2:
        return []
    return [max(1, (b - a).days) for a, b in zip(dates[:-1], dates[1:]) if b > a]


def _annualized_dividend_series(
    events: list[tuple[date, float]],
    history_dates: list[date] | None = None,
) -> list[dict]:
    if not events:
        return []
    events = sorted(events, key=lambda item: item[0])
    dates = [dt for dt, _amt in events]
    history = sorted({dt for dt in (history_dates or dates) if dt})
    history_index = {dt: idx for idx, dt in enumerate(history)}
    out = []
    for idx, (dt, amt) in enumerate(events):
        gap_prev = None
        gap_next = None
        hist_idx = history_index.get(dt)
        if hist_idx is not None:
            if hist_idx > 0:
                gap_prev = (dt - history[hist_idx - 1]).days
            if hist_idx + 1 < len(history):
                gap_next = (history[hist_idx + 1] - dt).days
        if gap_prev is None or gap_prev <= 0:
            gap_prev = (dt - dates[idx - 1]).days if idx > 0 else None
        if gap_next is None or gap_next <= 0:
            gap_next = (dates[idx + 1] - dt).days if idx + 1 < len(dates) else None
        gap_days = gap_prev if gap_prev and gap_prev > 0 else gap_next
        freq = _frequency_from_gap_days(gap_days)
        per_year = _payouts_per_year(freq) or 1
        out.append({"date": dt, "amount": amt, "annualized": amt * per_year, "special": False})
    history = []
    for ev in out:
        if len(history) >= 3:
            window = history[-6:]
            median = statistics.median(window)
            if median > 0 and ev["annualized"] > median * SPECIAL_DIVIDEND_FACTOR:
                ev["special"] = True
        history.append(ev["annualized"])
    return out


def _detect_ex_date_pattern(
    ex_dates: list[date],
    min_samples: int = 4,
    match_threshold: float = 0.70,
) -> dict | None:
    """Detect recurring day-of-month pattern in ex-dates.

    Tries three pattern types in order:
      1. Nth weekday of month (e.g. 4th Monday)
      2. Nth-to-last business day of month
      3. Fixed calendar day of month

    Returns dict with pattern_type, params, confidence -- or None.
    """
    if len(ex_dates) < min_samples:
        return None

    dates = sorted(ex_dates)
    n = len(dates)

    # --- Pattern 1: Nth weekday of month ---
    wd_week = [(d.weekday(), (d.day - 1) // 7 + 1) for d in dates]
    wd_counts = Counter(wd_week)
    (best_wd, best_wn), best_count = wd_counts.most_common(1)[0]
    confidence = best_count / n
    if confidence >= match_threshold:
        return {
            "pattern_type": "nth_weekday",
            "params": {"weekday": best_wd, "week_num": best_wn},
            "confidence": round(confidence, 3),
        }

    # --- Pattern 2: Nth-to-last business day ---
    def _bdays_from_end(d: date) -> int:
        me = _month_end(d)
        count = 0
        cursor = me
        while cursor > d:
            if cursor.weekday() < 5:
                count += 1
            cursor -= timedelta(days=1)
        return count

    bdays = [_bdays_from_end(d) for d in dates]
    bd_counts = Counter(bdays)
    best_bd, best_bd_count = bd_counts.most_common(1)[0]
    # allow +/-1 tolerance
    tolerant_count = sum(1 for b in bdays if abs(b - best_bd) <= 1)
    confidence = tolerant_count / n
    if confidence >= match_threshold:
        return {
            "pattern_type": "nth_last_bday",
            "params": {"bdays_from_end": best_bd},
            "confidence": round(confidence, 3),
        }

    # --- Pattern 3: Fixed day of month ---
    day_counts = Counter(d.day for d in dates)
    best_day, best_day_count = day_counts.most_common(1)[0]
    confidence = best_day_count / n
    if confidence >= match_threshold:
        return {
            "pattern_type": "fixed_day",
            "params": {"day": best_day},
            "confidence": round(confidence, 3),
        }

    return None


def _project_next_from_pattern(last_ex_date: date, pattern: dict, frequency_months: int) -> date | None:
    """Project the next ex-date based on a detected pattern."""
    ptype = pattern["pattern_type"]
    params = pattern["params"]

    target_month = last_ex_date.month + frequency_months
    target_year = last_ex_date.year + (target_month - 1) // 12
    target_month = (target_month - 1) % 12 + 1

    if ptype == "nth_weekday":
        weekday = params["weekday"]
        week_num = params["week_num"]
        first = date(target_year, target_month, 1)
        offset = (weekday - first.weekday()) % 7
        candidate = first + timedelta(days=offset + 7 * (week_num - 1))
        # clamp to same month (e.g. 5th Monday may overflow)
        if candidate.month != target_month:
            candidate -= timedelta(days=7)
        return candidate if candidate > last_ex_date else None

    if ptype == "nth_last_bday":
        bdays_from_end = params["bdays_from_end"]
        me = _month_end(date(target_year, target_month, 1))
        cursor = me
        count = 0
        while count < bdays_from_end and cursor.day > 1:
            if cursor.weekday() < 5:
                count += 1
            cursor -= timedelta(days=1)
        # cursor may be on weekend, walk to prior weekday
        while cursor.weekday() >= 5:
            cursor -= timedelta(days=1)
        return cursor if cursor > last_ex_date else None

    if ptype == "fixed_day":
        target_day = min(params["day"], calendar.monthrange(target_year, target_month)[1])
        candidate = date(target_year, target_month, target_day)
        return candidate if candidate > last_ex_date else None

    return None


_FREQ_MONTHS = {"monthly": 1, "quarterly": 3, "semiannual": 6, "annual": 12}


def _next_ex_date_est(ex_dates):
    if len(ex_dates) < 2:
        return None

    # Try pattern detection first
    pattern = _detect_ex_date_pattern(ex_dates)
    if pattern is not None:
        frequency = _frequency_from_ex_dates(ex_dates)
        months = _FREQ_MONTHS.get(frequency, 1)
        projected = _project_next_from_pattern(ex_dates[-1], pattern, months)
        if projected is not None and projected > ex_dates[-1]:
            return projected

    # Fallback: median gap
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


def _best_per_share_amount(
    sym: str,
    ex_date_est: date | None,
    provider_divs: dict,
    pay_history: list[dict],
) -> float | None:
    """Estimate per-share dividend amount for an upcoming ex-date.

    Priority:
      1. Most recent provider-sourced amount for this symbol
      2. Closest provider event to estimated ex-date
      3. Trailing median from LM payment history
    """
    events = provider_divs.get(sym, [])

    if events:
        recent = sorted(
            (ev for ev in events if ev.get("amount") is not None),
            key=lambda ev: ev["ex_date"],
            reverse=True,
        )
        if recent:
            return recent[0]["amount"]

    if events and ex_date_est:
        closest = min(
            (ev for ev in events if ev.get("amount") is not None),
            key=lambda ev: abs((ev["ex_date"] - ex_date_est).days),
            default=None,
        )
        if closest:
            return closest["amount"]

    if pay_history:
        amounts = [ev["amount"] for ev in pay_history[-6:] if isinstance(ev.get("amount"), (int, float))]
        if amounts:
            return float(statistics.median(amounts))

    return None


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


def _load_symbol_trade_deltas(conn: sqlite3.Connection, symbols: set[str]) -> dict[str, dict[date, float]]:
    if not symbols:
        return {}
    cur = conn.cursor()
    symbol_list = sorted({str(sym).upper() for sym in symbols if sym})
    if not symbol_list:
        return {}
    sym_placeholders = ",".join("?" for _ in symbol_list)
    type_list = sorted(SHARE_TX_TYPES)
    type_placeholders = ",".join("?" for _ in type_list)
    rows = cur.execute(
        f"""
        SELECT symbol, COALESCE(transaction_datetime, date), transaction_type, quantity, name
        FROM investment_transactions
        WHERE symbol IN ({sym_placeholders})
          AND lower(transaction_type) IN ({type_placeholders})
        ORDER BY COALESCE(transaction_datetime, date) ASC, lm_transaction_id ASC
        """,
        tuple(symbol_list + type_list),
    ).fetchall()
    deltas: dict[str, dict[date, float]] = defaultdict(lambda: defaultdict(float))
    for symbol, dt_str, tx_type, quantity, name in rows:
        if not symbol:
            continue
        dt = _parse_date(dt_str)
        if dt is None:
            continue
        qty = _coerce_float(quantity)
        if qty is None:
            qty = _extract_quantity(name)
        if qty is None or qty == 0:
            continue
        tx_type = (tx_type or "").lower()
        if qty < 0:
            delta = float(qty)
        elif tx_type in ACQUIRE_TYPES:
            delta = float(qty)
        elif tx_type in SELL_TYPES:
            delta = float(-qty)
        else:
            continue
        sym = str(symbol).upper()
        deltas[sym][dt] += delta
    return deltas


def _load_symbol_splits(conn: sqlite3.Connection, symbols: set[str]) -> dict[str, dict[date, list[float]]]:
    if not symbols:
        return {}
    cur = conn.cursor()
    symbol_list = sorted({str(sym).upper() for sym in symbols if sym})
    if not symbol_list:
        return {}
    sym_placeholders = ",".join("?" for _ in symbol_list)
    rows = cur.execute(
        f"""
        SELECT symbol, ex_date, ratio
        FROM split_events
        WHERE symbol IN ({sym_placeholders})
        """,
        tuple(symbol_list),
    ).fetchall()
    splits: dict[str, dict[date, list[float]]] = defaultdict(lambda: defaultdict(list))
    for symbol, ex_date, ratio in rows:
        if not symbol:
            continue
        dt = _parse_date(ex_date)
        ratio_val = _coerce_float(ratio)
        if dt is None or ratio_val is None or ratio_val <= 0:
            continue
        sym = str(symbol).upper()
        splits[sym][dt].append(float(ratio_val))
    return splits


def _build_position_index(conn: sqlite3.Connection, symbols: set[str]) -> dict[str, tuple[list[date], list[float]]]:
    if not symbols:
        return {}
    trade_deltas = _load_symbol_trade_deltas(conn, symbols)
    split_events = _load_symbol_splits(conn, symbols)
    index: dict[str, tuple[list[date], list[float]]] = {}
    all_syms = set(trade_deltas.keys()) | set(split_events.keys())
    for sym in sorted(all_syms):
        trade_map = trade_deltas.get(sym, {})
        split_map = split_events.get(sym, {})
        if not trade_map and not split_map:
            continue
        dates = sorted(set(trade_map.keys()) | set(split_map.keys()))
        shares = 0.0
        date_list: list[date] = []
        share_list: list[float] = []
        for dt in dates:
            ratios = split_map.get(dt) or []
            for ratio in ratios:
                shares *= ratio
            shares += trade_map.get(dt, 0.0)
            date_list.append(dt)
            share_list.append(shares)
        index[sym] = (date_list, share_list)
    return index


def _shares_at_date(position_index: dict[str, tuple[list[date], list[float]]], symbol: str, target_date: date) -> float:
    if not position_index or not symbol or not target_date:
        return 0.0
    entry = position_index.get(str(symbol).upper())
    if not entry:
        return 0.0
    dates, shares = entry
    idx = bisect_right(dates, target_date) - 1
    if idx < 0:
        return 0.0
    return float(shares[idx])


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


def _payment_received(pay_history: dict[str, list[dict]], symbol: str, pay_date: date, tolerance_days: int) -> bool:
    if not symbol or not pay_date:
        return False
    events = pay_history.get(str(symbol).upper(), [])
    for ev in events:
        dt = ev.get("date")
        if dt and abs((dt - pay_date).days) <= tolerance_days:
            return True
    return False


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


def _coerce_float(val):
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _extract_quantity(text: str | None):
    if not text:
        return None
    match = _SHARES_RE.search(text)
    return float(match.group(1)) if match else None


def _months_between(start: date, end: date) -> int:
    if start > end:
        return 0
    return (end.year - start.year) * 12 + (end.month - start.month) + 1


def _load_first_acquired_dates(conn: sqlite3.Connection) -> dict[str, date]:
    cur = conn.cursor()
    type_placeholders = ",".join("?" for _ in sorted(ACQUIRE_TYPES))
    rows = cur.execute(
        f"""
        SELECT symbol, MIN(COALESCE(transaction_datetime, date))
        FROM investment_transactions
        WHERE symbol IS NOT NULL
          AND lower(transaction_type) IN ({type_placeholders})
        GROUP BY symbol
        """,
        tuple(sorted(ACQUIRE_TYPES)),
    ).fetchall()
    out: dict[str, date] = {}
    for symbol, dt_str in rows:
        sym = str(symbol).upper() if symbol else None
        dt = _parse_date(dt_str)
        if sym and dt:
            out[sym] = dt
    return out


def _dividend_reliability_metrics(
    symbol: str,
    div_tx: list[dict],
    provider_divs: dict,
    pay_history: dict,
    expected_frequency: str | None,
    as_of_date: date,
    first_acquired: date | None = None,
    position_index: dict[str, tuple[list[date], list[float]]] | None = None,
) -> dict:
    cutoff = as_of_date - timedelta(days=365)
    window_start = cutoff
    if first_acquired and first_acquired <= as_of_date and first_acquired > cutoff:
        window_start = first_acquired
    window_months = max(1, _months_between(window_start, as_of_date)) if window_start else 12

    totals_12 = _monthly_income_totals(div_tx, as_of_date, window_months, symbol=symbol)

    # If no actual received dividends, fall back to provider ex-date amounts
    # over the full 12-month window for trend/growth/volatility.
    if not any(t > 0 for t in totals_12):
        provider_events = provider_divs.get(symbol, [])
        if provider_events:
            prov_by_month: dict[tuple[int, int], float] = defaultdict(float)
            for ev in provider_events:
                ex = ev.get("ex_date")
                amt = ev.get("amount")
                if ex and isinstance(amt, (int, float)) and cutoff <= ex <= as_of_date:
                    prov_by_month[(ex.year, ex.month)] += float(amt)
            if prov_by_month:
                months = _month_keys(as_of_date, 12)
                totals_12 = [prov_by_month.get(key, 0.0) for key in months]

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
    pay_dates = [ev.get("date") for ev in pay_events if ev.get("date") and ev.get("date") >= window_start]
    pay_dates = [d for d in pay_dates if d]
    pay_dates.sort()

    provider_events = provider_divs.get(symbol, [])
    # Use full 365-day cutoff for provider ex-dates — the stock's dividend
    # schedule is independent of when the position was acquired.
    ex_dates = sorted(
        {
            ev.get("ex_date")
            for ev in provider_events
            if ev.get("ex_date") and cutoff <= ev.get("ex_date") <= as_of_date
        }
    )

    payment_frequency_actual = (
        _frequency_from_recent_ex_dates(ex_dates, recent=6)
        or _frequency_from_ex_dates(ex_dates)
        or _frequency_from_dates(pay_dates)
    )
    payment_frequency_expected = expected_frequency or payment_frequency_actual
    if payment_frequency_actual and expected_frequency and expected_frequency != payment_frequency_actual:
        payment_frequency_expected = payment_frequency_actual

    default_pay_lag = _median_pay_lag_days(provider_divs)
    symbol_pay_lag = _symbol_pay_lag_days(provider_events, default_pay_lag)
    due_cutoff = as_of_date - timedelta(days=symbol_pay_lag) if symbol_pay_lag else as_of_date
    ex_dates_due = sorted(
        {
            ev.get("ex_date")
            for ev in provider_events
            if ev.get("ex_date") and cutoff <= ev.get("ex_date") <= due_cutoff
        }
    )

    if position_index:
        ex_dates_due = [dt for dt in ex_dates_due if _shares_at_date(position_index, symbol, dt) > 0]

    missed_payments = 0
    if pay_dates:
        expected_count = len(ex_dates_due) if ex_dates_due else None
        if expected_count is None and len(pay_dates) >= 2:
            gap_days = _median_gap_days(pay_dates)
            if gap_days:
                span_days = (due_cutoff - pay_dates[0]).days
                if span_days >= 0:
                    expected_count = int(span_days // gap_days) + 1
        if expected_count is not None:
            missed_payments = max(expected_count - len(pay_dates), 0)

    timing_dates = pay_dates if len(pay_dates) >= 2 else ex_dates
    gap_days = _gap_days(timing_dates)
    avg_gap = round(statistics.mean(gap_days), 1) if gap_days else None
    timing_consistency = None
    if gap_days:
        mean_gap = statistics.mean(gap_days)
        if mean_gap > 0:
            timing_consistency = max(0.0, min(1.0, 1.0 - (statistics.pstdev(gap_days) / mean_gap)))

    cut_window_end = as_of_date + timedelta(days=DIVIDEND_CUT_LOOKAHEAD_DAYS)
    events = []
    for ev in provider_divs.get(symbol, []):
        ex_date = ev.get("ex_date")
        amt = ev.get("amount")
        if ex_date and isinstance(amt, (int, float)) and cutoff <= ex_date <= cut_window_end:
            events.append((ex_date, float(amt)))
    events.sort(key=lambda item: item[0])
    if not events and pay_events:
        for ev in pay_events:
            dt = ev.get("date")
            amt = ev.get("amount")
            if dt and isinstance(amt, (int, float)) and window_start <= dt <= as_of_date:
                events.append((dt, float(amt)))
        events.sort(key=lambda item: item[0])

    cuts = 0
    last_increase = None
    last_decrease = None
    history_dates = sorted(
        {
            ev.get("ex_date")
            for ev in provider_divs.get(symbol, [])
            if ev.get("ex_date")
        }
    )
    cut_series = _annualized_dividend_series(events, history_dates)
    for prev, curr in zip(cut_series[:-1], cut_series[1:]):
        if prev.get("special") or curr.get("special"):
            continue
        prev_amt = prev.get("annualized")
        curr_amt = curr.get("annualized")
        if prev_amt is None or curr_amt is None or prev_amt <= 0:
            continue
        change = (curr_amt - prev_amt) / prev_amt
        if change <= -DIVIDEND_CUT_THRESHOLD:
            cuts += 1
            last_decrease = curr.get("date")
        elif change >= DIVIDEND_CUT_THRESHOLD:
            last_increase = curr.get("date")

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
        ORDER BY symbol, ex_date,
            CASE WHEN pay_date IS NOT NULL AND pay_date != '' THEN 0 ELSE 1 END,
            CASE source
                WHEN 'nasdaq' THEN 0
                WHEN 'yfinance' THEN 1
                WHEN 'yahooquery' THEN 2
                WHEN 'openbb' THEN 3
                ELSE 4
            END
        """
    ).fetchall()
    # Group by (symbol, ex_date).  When multiple providers report the same
    # dividend with slightly different amounts (e.g. 0.6359 vs 0.6360),
    # average the amounts and keep the best pay_date (ORDER BY already
    # places records with a pay_date first).
    # If an amount is far from the group average (>10%), it's likely a
    # special dividend and kept as a separate record.
    grouped: dict[tuple[str, str], list[dict]] = {}
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
        key = (sym, dt.isoformat())
        if key not in grouped:
            grouped[key] = [{"ex_date": dt, "pay_date": pay_dt, "amt_sum": amt, "count": 1}]
        else:
            # Find a bucket whose average is close to this amount
            matched = False
            for bucket in grouped[key]:
                avg = bucket["amt_sum"] / bucket["count"]
                if avg > 0 and abs(amt - avg) / avg <= 0.10:
                    bucket["amt_sum"] += amt
                    bucket["count"] += 1
                    if pay_dt and not bucket["pay_date"]:
                        bucket["pay_date"] = pay_dt
                    matched = True
                    break
            if not matched:
                grouped[key].append({"ex_date": dt, "pay_date": pay_dt, "amt_sum": amt, "count": 1})

    out = defaultdict(list)
    for (sym, _dt_str), buckets in grouped.items():
        for rec in buckets:
            out[sym].append({
                "ex_date": rec["ex_date"],
                "pay_date": rec["pay_date"],
                "amount": rec["amt_sum"] / rec["count"],
            })
    for sym in out:
        out[sym].sort(key=lambda item: item["ex_date"])
    return out


def _load_symbol_pay_lag_days(conn: sqlite3.Connection) -> dict[str, int]:
    """Compute per-symbol median ex→pay lag.
    Priority: provider records with real pay_date, then LM matched records."""
    cur = conn.cursor()

    # Provider records with actual pay_dates (most authoritative)
    provider_rows = cur.execute(
        """
        SELECT symbol, ex_date, pay_date
        FROM dividend_events_provider
        WHERE pay_date IS NOT NULL AND pay_date != ''
        """
    ).fetchall()
    provider_lags: dict[str, list[int]] = defaultdict(list)
    for symbol, ex_date, pay_date in provider_rows:
        sym = str(symbol).upper() if symbol else None
        ex_dt = _parse_date(ex_date)
        pay_dt = _parse_date(pay_date)
        if not sym or not ex_dt or not pay_dt:
            continue
        delta = (pay_dt - ex_dt).days
        if 0 <= delta <= 60:
            provider_lags[sym].append(delta)

    # LM matched records (fallback)
    lm_rows = cur.execute(
        """
        SELECT symbol, ex_date_est, pay_date_est
        FROM dividend_events_lm
        WHERE ex_date_est IS NOT NULL AND pay_date_est IS NOT NULL
        """
    ).fetchall()
    lm_lags: dict[str, list[int]] = defaultdict(list)
    for symbol, ex_date_est, pay_date_est in lm_rows:
        sym = str(symbol).upper() if symbol else None
        ex_dt = _parse_date(ex_date_est)
        pay_dt = _parse_date(pay_date_est)
        if not sym or not ex_dt or not pay_dt:
            continue
        delta = (pay_dt - ex_dt).days
        if 0 <= delta <= 60:
            lm_lags[sym].append(delta)

    out: dict[str, int] = {}
    all_symbols = set(provider_lags.keys()) | set(lm_lags.keys())
    for sym in all_symbols:
        p_vals = provider_lags.get(sym, [])
        if len(p_vals) >= 3:
            out[sym] = max(1, int(round(statistics.median(p_vals))))
        elif p_vals:
            combined = p_vals + lm_lags.get(sym, [])
            out[sym] = max(1, int(round(statistics.median(combined))))
        else:
            l_vals = lm_lags.get(sym, [])
            if l_vals:
                out[sym] = max(1, int(round(statistics.median(l_vals))))
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
            if 0 <= delta <= 60:
                lags.append(delta)
    if not lags:
        return 14
    return max(1, min(30, int(round(statistics.median(lags)))))


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


def _estimate_pay_from_ex(
    ex: date,
    sym: str,
    pay_lag_by_symbol: dict[str, int] | None,
    default_lag: int,
    events: list[dict],
    ex_date_patterns: dict[str, dict] | None = None,
) -> date:
    """Estimate pay-date from an ex-date using historical patterns.

    Tier 1 (caller handles): provider-sourced pay_date
    Tier 2: symbol-specific lag → last business day of the target month
    Tier 3: last business day of the ex-date's month (default)

    For monthly payers with ex-dates in the last ~10 days of the month,
    pay is capped to the same month (month-end funds always pay within
    the same month, and LM-derived lags are inflated by sync delays).
    For quarterly/annual payers the lag is used as-is since the payment
    genuinely lands in a different month.
    """
    sym_lag = pay_lag_by_symbol.get(sym) if pay_lag_by_symbol else None
    freq = _frequency_from_ex_dates(
        sorted(ev["ex_date"] for ev in events if ev.get("ex_date"))
    ) if events else None

    if sym_lag is not None and sym_lag >= 1:
        target = ex + timedelta(days=sym_lag)
        # Monthly payers with ex-date near month-end: keep pay in same month.
        # Their LM-derived lags are inflated by brokerage sync delays.
        if freq == "monthly" and ex.day >= _month_end(ex).day - 10:
            if target.month != ex.month or target.year != ex.year:
                return _last_business_day(ex)
        return _last_business_day(target)

    # No symbol-specific lag → pay in same month as ex-date
    return _last_business_day(ex)


def _estimate_expected_pay_events(
    provider_divs: dict,
    holdings: dict,
    window_start: date,
    window_end: date,
    pay_history: dict,
    as_of_date: date | None = None,
    ex_date_est_by_symbol: dict[str, date] | None = None,
    position_index: dict[str, tuple[list[date], list[float]]] | None = None,
    pay_lag_by_symbol: dict[str, int] | None = None,
    ex_date_patterns: dict[str, dict] | None = None,
):
    default_lag = _median_pay_lag_days(provider_divs)
    if as_of_date is None:
        as_of_date = window_end
    expected = []
    total_raw = 0.0
    ex_date_est_by_symbol = ex_date_est_by_symbol or {}

    def _best_ex_date_for_pay(sym: str, pay_date: date):
        events = provider_divs.get(sym, [])
        matches = []
        for ev in events:
            ex = ev.get("ex_date")
            pay = ev.get("pay_date")
            if not ex or not pay:
                continue
            delta = abs((pay - pay_date).days)
            if delta <= DIVIDEND_PAY_MATCH_WINDOW_DAYS:
                matches.append((delta, ex))
        if matches:
            matches.sort(key=lambda item: item[0])
            return matches[0][1]
        est = ex_date_est_by_symbol.get(sym)
        if est and est <= pay_date and (pay_date - est).days <= 60:
            return est
        past_ex = [ev.get("ex_date") for ev in events if ev.get("ex_date") and ev.get("ex_date") <= pay_date]
        if past_ex:
            return max(past_ex)
        lag_days = None
        if pay_lag_by_symbol:
            lag_days = pay_lag_by_symbol.get(sym)
        if lag_days is None:
            lag_days = _symbol_pay_lag_days(events, default_lag)
        return pay_date - timedelta(days=lag_days) if lag_days else None
    def _pay_date_seen(seen: list[date], candidate: date) -> bool:
        for dt in seen:
            if abs((candidate - dt).days) <= DIVIDEND_PAY_MATCH_WINDOW_DAYS:
                return True
        return False
    def _shares_for(sym: str, ex_date: date) -> float:
        if position_index:
            return _shares_at_date(position_index, sym, ex_date)
        return float(holdings[sym]["shares"])

    for sym in sorted(holdings.keys()):
        history = pay_history.get(sym, [])
        events = provider_divs.get(sym, [])
        pay_dates_seen = []
        window_has_event = False

        # Actual payments in the window: count toward projection, but do not list.
        for ev in history:
            pay_date = ev.get("date")
            if not pay_date or pay_date < window_start or pay_date > window_end:
                continue
            amount_est = ev.get("amount")
            if isinstance(amount_est, (int, float)):
                total_raw += float(amount_est)
            pay_dates_seen.append(pay_date)
            window_has_event = True

        # Provider-backed payments in the window (pattern-aware pay-date).
        if events:
            for ev in events:
                ex = ev.get("ex_date")
                amt = ev.get("amount")
                if ex is None or amt is None:
                    continue
                # Tier 1: provider-sourced pay_date, else pattern-aware estimation
                pay = ev.get("pay_date")
                if pay is None:
                    pay = _estimate_pay_from_ex(
                        ex, sym, pay_lag_by_symbol, default_lag, events, ex_date_patterns)
                if pay < window_start or pay > window_end:
                    continue
                window_has_event = True
                if _pay_date_seen(pay_dates_seen, pay):
                    continue
                shares = _shares_for(sym, ex)
                if shares <= 0:
                    continue
                amount_est = amt * shares
                total_raw += float(amount_est)
                if pay <= as_of_date and _payment_received(pay_history, sym, pay, DIVIDEND_PAY_MATCH_WINDOW_DAYS):
                    pay_dates_seen.append(pay)
                    continue
                status = "overdue" if pay < as_of_date else "pending"
                expected.append(
                    {
                        "symbol": sym,
                        "ex_date_est": ex.isoformat(),
                        "pay_date_est": pay.isoformat(),
                        "amount_est": _round_money(amount_est),
                        "status": status,
                    }
                )
                pay_dates_seen.append(pay)

        # Estimate a missing payment when no events land in the window.
        # Fire when we have LM history (2+ records) OR provider data + projected ex-date.
        has_projection = bool(ex_date_est_by_symbol.get(sym) and events)
        if not window_has_event and (len(history) >= 2 or has_projection):
            ex_date_est = ex_date_est_by_symbol.get(sym)
            next_pay = None
            if ex_date_est:
                next_pay_candidate = _estimate_pay_from_ex(
                    ex_date_est, sym, pay_lag_by_symbol, default_lag,
                    provider_divs.get(sym, []), ex_date_patterns)
                if window_start <= next_pay_candidate <= window_end:
                    next_pay = next_pay_candidate
            if next_pay is None:
                dates = [ev.get("date") for ev in history if ev.get("date")]
                median_gap = _median_gap_days(dates)
                if median_gap:
                    next_pay = dates[-1] + timedelta(days=median_gap)
            if next_pay and window_start <= next_pay <= window_end and not _pay_date_seen(pay_dates_seen, next_pay):
                if ex_date_est is None:
                    ex_date_est = _best_ex_date_for_pay(sym, next_pay)
                per_share = _best_per_share_amount(sym, ex_date_est, provider_divs, history)
                amount_est = None
                if per_share is not None and ex_date_est:
                    shares = _shares_for(sym, ex_date_est)
                    if shares > 0:
                        amount_est = float(per_share) * shares
                if amount_est is None:
                    amount_est = _median_amount(history[-6:])
                if isinstance(amount_est, (int, float)):
                    total_raw += float(amount_est)
                if not _payment_received(pay_history, sym, next_pay, DIVIDEND_PAY_MATCH_WINDOW_DAYS):
                    status = "overdue" if next_pay < as_of_date else "pending"
                    expected.append(
                        {
                            "symbol": sym,
                            "ex_date_est": ex_date_est.isoformat() if ex_date_est else None,
                            "pay_date_est": next_pay.isoformat(),
                            "amount_est": _round_money(amount_est),
                            "status": status,
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


def _income_stability_metrics(
    div_tx: list[dict],
    as_of_date: date,
    window_months: int = 12,
    start_date: date | None = None,
) -> dict:
    if start_date and start_date <= as_of_date:
        window_months = max(1, min(window_months, _months_between(start_date, as_of_date)))
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


def _income_stability_score(
    div_tx: list[dict],
    as_of_date: date,
    window_months: int = 6,
    start_date: date | None = None,
) -> float:
    metrics = _income_stability_metrics(
        div_tx,
        as_of_date,
        window_months=max(window_months, 6),
        start_date=start_date,
    )
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


def _append_current_value(values: pd.Series | None, current_value: float | None, current_date: date | None) -> pd.Series | None:
    if current_value is None or current_date is None:
        return values
    ts = pd.Timestamp(current_date)
    if values is None or values.empty:
        return pd.Series([float(current_value)], index=[ts])
    series = values.copy()
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    series.loc[ts] = float(current_value)
    return series.sort_index()


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
        # Support both v4 (top-level totals) and v5 (portfolio.totals)
        totals = (snap.get("totals") or (snap.get("portfolio") or {}).get("totals") or {}) if snap else {}
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


def _build_goal_tiers(
    projected_monthly_income,
    total_market_value,
    margin_loan_balance,
    target_monthly,
    performance_metrics,
    as_of_date: date | None = None,
):
    """
    Build comprehensive dividend goal tracking across 6 tiers:
    1. Modest Appreciation: Price appreciation only (6m TWR)
    2. Conservative Build: Contributions + 6m TWR (no DRIP)
    3. DRIP Build: DRIP + contribution + 6m TWR
    4. Leveraged Growth: DRIP + contribution + midrange + margin
    5. Optimized Growth: DRIP + contribution + midrange + margin
    6. Maximum Growth: DRIP + contribution + 12m TWR + margin

    Price appreciation in 3 steps based on actual portfolio TWR:
    - Modest (Tiers 1-3): 6-month TWR
    - Midrange (Tiers 4-5): Average of 6m and 12m TWR
    - Max (Tier 6): 12-month TWR
    """
    if not target_monthly or target_monthly <= 0 or not total_market_value:
        return None

    monthly_contribution = settings.goal_monthly_contribution or 0.0
    monthly_drip = projected_monthly_income or 0.0
    target_ltv_pct = settings.goal_target_ltv_pct / 100.0
    portfolio_yield_pct = _safe_divide(projected_monthly_income * 12.0, total_market_value)
    portfolio_yield_pct_display = portfolio_yield_pct * 100 if portfolio_yield_pct is not None else None

    # Current state (for reference in calculations)
    current_ltv = _safe_divide(margin_loan_balance or 0.0, total_market_value)

    # Extract appreciation rates from actual portfolio performance (3 steps)
    twr_6m = performance_metrics.get("twr_6m_pct") or 10.0
    twr_12m = performance_metrics.get("twr_12m_pct") or 15.0

    modest_appreciation = twr_6m  # Tiers 1-3
    midrange_appreciation = (twr_6m + twr_12m) / 2.0  # Tiers 4-5
    max_appreciation = twr_12m  # Tier 6

    tiers = []

    # Helper to calculate months to goal with compounding
    def calculate_timeline(
        initial_portfolio_value,
        initial_margin_balance,
        monthly_contrib,
        monthly_div_income,
        drip_enabled,
        annual_appreciation_pct,
        target_monthly_income,
        maintain_ltv=False,
        ltv_target=0.0,
    ):
        """Calculate months to reach target income with compounding growth"""
        if portfolio_yield_pct is None or portfolio_yield_pct <= 0:
            return None, None, None

        # Calculate required portfolio value to generate target income
        required_portfolio = target_monthly_income * 12.0 / portfolio_yield_pct

        # If we're already there
        if initial_portfolio_value >= required_portfolio:
            return 0, required_portfolio, initial_portfolio_value

        # Simulate month-by-month growth
        portfolio_value = initial_portfolio_value
        margin_balance = initial_margin_balance or 0.0
        months = 0
        max_months = 1200  # 100 years - user doesn't care about caps

        monthly_appreciation = annual_appreciation_pct / 12.0 / 100.0
        margin_apr = settings.margin_apr_current  # e.g., 0.0415
        monthly_margin_rate = margin_apr / 12.0

        while portfolio_value < required_portfolio and months < max_months:
            months += 1

            # 1. Apply price appreciation to existing portfolio
            portfolio_value *= (1 + monthly_appreciation)

            # 2. Add margin interest to balance
            if maintain_ltv and margin_balance > 0:
                margin_interest = margin_balance * monthly_margin_rate
                margin_balance += margin_interest

            # 3. Calculate DRIP amount
            drip_amount = 0.0
            if drip_enabled:
                drip_amount = portfolio_value * portfolio_yield_pct / 12.0

            # 4. Calculate additional margin needed to maintain target LTV
            additional_margin = 0.0
            if maintain_ltv and ltv_target > 0:
                # User's formula: additional_margin = (0.3 * (portfolio_value + drip + contribution) - margin_balance) / 0.7
                # This maintains 30% LTV: (margin_balance + additional_margin) / (portfolio_value + drip + contribution + additional_margin) = 0.3
                total_before_margin = portfolio_value + drip_amount + monthly_contrib
                additional_margin = (ltv_target * total_before_margin - margin_balance) / (1.0 - ltv_target)
                additional_margin = max(0.0, additional_margin)  # Can't unborrow
                margin_balance += additional_margin

            # 5. Total amount invested this month
            amount_purchased = drip_amount + monthly_contrib + additional_margin

            # 6. Add to portfolio
            portfolio_value += amount_purchased

        if months >= max_months:
            return None, required_portfolio, portfolio_value

        return months, required_portfolio, portfolio_value

    # Tier 1: Modest Appreciation Only
    tier1_months, tier1_required, tier1_final = calculate_timeline(
        initial_portfolio_value=total_market_value,
        initial_margin_balance=margin_loan_balance,
        monthly_contrib=0.0,
        monthly_div_income=monthly_drip,
        drip_enabled=False,
        annual_appreciation_pct=modest_appreciation,
        target_monthly_income=target_monthly,
        maintain_ltv=False,
    )

    tiers.append({
        "tier": 1,
        "name": "Modest Appreciation",
        "description": f"{modest_appreciation:.1f}% appreciation only (no contributions, no DRIP)",
        "assumptions": {
            "monthly_contribution": 0.0,
            "drip_enabled": False,
            "annual_appreciation_pct": _round_pct(modest_appreciation),
            "ltv_maintained": False,
        },
        "months_to_goal": tier1_months,
        "estimated_goal_date": _add_months(as_of_date, tier1_months).isoformat() if as_of_date and tier1_months is not None else None,
        "required_portfolio_value": _round_money(tier1_required),
        "final_portfolio_value": _round_money(tier1_final),
        "additional_investment_needed": _round_money(tier1_required - total_market_value if tier1_required else None),
    })

    # Tier 2: Conservative Build
    tier2_months, tier2_required, tier2_final = calculate_timeline(
        initial_portfolio_value=total_market_value,
        initial_margin_balance=margin_loan_balance,
        monthly_contrib=monthly_contribution,
        monthly_div_income=monthly_drip,
        drip_enabled=False,
        annual_appreciation_pct=modest_appreciation,
        target_monthly_income=target_monthly,
        maintain_ltv=False,
    )

    tiers.append({
        "tier": 2,
        "name": "Conservative Build",
        "description": f"${monthly_contribution}/mo contributions, {modest_appreciation:.1f}% appreciation (no DRIP)",
        "assumptions": {
            "monthly_contribution": _round_money(monthly_contribution),
            "drip_enabled": False,
            "annual_appreciation_pct": _round_pct(modest_appreciation),
            "ltv_maintained": False,
        },
        "months_to_goal": tier2_months,
        "estimated_goal_date": _add_months(as_of_date, tier2_months).isoformat() if as_of_date and tier2_months is not None else None,
        "required_portfolio_value": _round_money(tier2_required),
        "final_portfolio_value": _round_money(tier2_final),
        "additional_investment_needed": _round_money(tier2_required - total_market_value if tier2_required else None),
    })

    # Tier 3: DRIP Build
    tier3_months, tier3_required, tier3_final = calculate_timeline(
        initial_portfolio_value=total_market_value,
        initial_margin_balance=margin_loan_balance,
        monthly_contrib=monthly_contribution,
        monthly_div_income=monthly_drip,
        drip_enabled=True,
        annual_appreciation_pct=modest_appreciation,
        target_monthly_income=target_monthly,
        maintain_ltv=False,
    )

    tiers.append({
        "tier": 3,
        "name": "DRIP Build",
        "description": f"DRIP + ${monthly_contribution}/mo + {modest_appreciation:.1f}% appreciation",
        "assumptions": {
            "monthly_contribution": _round_money(monthly_contribution),
            "drip_enabled": True,
            "annual_appreciation_pct": _round_pct(modest_appreciation),
            "ltv_maintained": False,
        },
        "months_to_goal": tier3_months,
        "estimated_goal_date": _add_months(as_of_date, tier3_months).isoformat() if as_of_date and tier3_months is not None else None,
        "required_portfolio_value": _round_money(tier3_required),
        "final_portfolio_value": _round_money(tier3_final),
        "additional_investment_needed": _round_money(tier3_required - total_market_value if tier3_required else None),
    })

    # Tier 4: Leveraged Growth (Modest)
    tier4_months, tier4_required, tier4_final = calculate_timeline(
        initial_portfolio_value=total_market_value,
        initial_margin_balance=margin_loan_balance,
        monthly_contrib=monthly_contribution,
        monthly_div_income=monthly_drip,
        drip_enabled=True,
        annual_appreciation_pct=modest_appreciation,
        target_monthly_income=target_monthly,
        maintain_ltv=True,
        ltv_target=target_ltv_pct,
    )

    tiers.append({
        "tier": 4,
        "name": "Leveraged Growth",
        "description": f"DRIP + ${monthly_contribution}/mo + {modest_appreciation:.1f}% + {settings.goal_target_ltv_pct}% LTV",
        "assumptions": {
            "monthly_contribution": _round_money(monthly_contribution),
            "drip_enabled": True,
            "annual_appreciation_pct": _round_pct(modest_appreciation),
            "ltv_maintained": True,
            "target_ltv_pct": _round_pct(settings.goal_target_ltv_pct),
        },
        "months_to_goal": tier4_months,
        "estimated_goal_date": _add_months(as_of_date, tier4_months).isoformat() if as_of_date and tier4_months is not None else None,
        "required_portfolio_value": _round_money(tier4_required),
        "final_portfolio_value": _round_money(tier4_final),
        "additional_investment_needed": _round_money(tier4_required - total_market_value if tier4_required else None),
    })

    # Tier 5: Optimized Growth (Midrange)
    tier5_months, tier5_required, tier5_final = calculate_timeline(
        initial_portfolio_value=total_market_value,
        initial_margin_balance=margin_loan_balance,
        monthly_contrib=monthly_contribution,
        monthly_div_income=monthly_drip,
        drip_enabled=True,
        annual_appreciation_pct=midrange_appreciation,
        target_monthly_income=target_monthly,
        maintain_ltv=True,
        ltv_target=target_ltv_pct,
    )

    tiers.append({
        "tier": 5,
        "name": "Optimized Growth",
        "description": f"DRIP + ${monthly_contribution}/mo + {midrange_appreciation:.1f}% + {settings.goal_target_ltv_pct}% LTV",
        "assumptions": {
            "monthly_contribution": _round_money(monthly_contribution),
            "drip_enabled": True,
            "annual_appreciation_pct": _round_pct(midrange_appreciation),
            "ltv_maintained": True,
            "target_ltv_pct": _round_pct(settings.goal_target_ltv_pct),
        },
        "months_to_goal": tier5_months,
        "estimated_goal_date": _add_months(as_of_date, tier5_months).isoformat() if as_of_date and tier5_months is not None else None,
        "required_portfolio_value": _round_money(tier5_required),
        "final_portfolio_value": _round_money(tier5_final),
        "additional_investment_needed": _round_money(tier5_required - total_market_value if tier5_required else None),
    })

    # Tier 6: Maximum Growth
    tier6_months, tier6_required, tier6_final = calculate_timeline(
        initial_portfolio_value=total_market_value,
        initial_margin_balance=margin_loan_balance,
        monthly_contrib=monthly_contribution,
        monthly_div_income=monthly_drip,
        drip_enabled=True,
        annual_appreciation_pct=max_appreciation,
        target_monthly_income=target_monthly,
        maintain_ltv=True,
        ltv_target=target_ltv_pct,
    )

    tiers.append({
        "tier": 6,
        "name": "Maximum Growth",
        "description": f"DRIP + ${monthly_contribution}/mo + {max_appreciation:.1f}% + {settings.goal_target_ltv_pct}% LTV",
        "assumptions": {
            "monthly_contribution": _round_money(monthly_contribution),
            "drip_enabled": True,
            "annual_appreciation_pct": _round_pct(max_appreciation),
            "ltv_maintained": True,
            "target_ltv_pct": _round_pct(settings.goal_target_ltv_pct),
        },
        "months_to_goal": tier6_months,
        "estimated_goal_date": _add_months(as_of_date, tier6_months).isoformat() if as_of_date and tier6_months is not None else None,
        "required_portfolio_value": _round_money(tier6_required),
        "final_portfolio_value": _round_money(tier6_final),
        "additional_investment_needed": _round_money(tier6_required - total_market_value if tier6_required else None),
    })

    return {
        "tiers": tiers,
        "current_state": {
            "portfolio_value": _round_money(total_market_value),
            "projected_monthly_income": _round_money(projected_monthly_income),
            "portfolio_yield_pct": _round_pct(portfolio_yield_pct_display),
            "current_ltv_pct": _round_pct(current_ltv * 100 if current_ltv else None),
            "margin_loan_balance": _round_money(margin_loan_balance),
            "target_monthly": _round_money(target_monthly),
        },
        "provenance": _provenance_entry(
            "derived",
            "internal",
            "goal_tiers_projection",
            ["income", "goal_settings", "monthly_contribution", "ltv_settings"],
            now_utc_iso(),
        ),
    }


def _detect_likely_tier(conn: sqlite3.Connection, projected_monthly_income: float, margin_loan_balance: float, total_market_value: float) -> dict:
    """
    Auto-detect the likely goal tier based on actual user behavior:
    - DRIP: Check for reinvest transactions in last 6 months
    - Contributions: Check if goal_monthly_contribution is set
    - Leverage: Check if margin is being actively maintained near target
    """
    cur = conn.cursor()

    # Check for DRIP (dividend transactions in last 180 days = DRIP active)
    # All dividends are reinvested, so if we see dividend transactions, DRIP is on
    six_months_ago = (date.today() - timedelta(days=180)).isoformat()
    dividend_count = cur.execute(
        """SELECT COUNT(*) FROM investment_transactions
           WHERE transaction_type = 'dividend'
           AND date >= ?""",
        (six_months_ago,)
    ).fetchone()[0]
    drip_detected = dividend_count > 0

    # Check for contributions
    contributions_detected = settings.goal_monthly_contribution > 0

    # Check for leverage maintenance (LTV within 5% of target)
    current_ltv = (margin_loan_balance or 0.0) / total_market_value if total_market_value > 0 else 0.0
    target_ltv = settings.goal_target_ltv_pct / 100.0
    leverage_maintained = abs(current_ltv - target_ltv) < 0.05 if margin_loan_balance and margin_loan_balance > 0 else False

    # Determine tier based on behavior pattern
    tier = 1
    confidence = "high"
    notes = []

    if drip_detected and contributions_detected and leverage_maintained:
        tier = 5  # Optimized Growth
        notes.append("DRIP active, contributions enabled, leverage maintained")
    elif drip_detected and contributions_detected:
        tier = 3  # DRIP Build
        notes.append("DRIP active, contributions enabled")
    elif contributions_detected and leverage_maintained:
        tier = 4  # Leveraged Growth
        notes.append("Contributions enabled, leverage maintained")
        confidence = "medium"
    elif contributions_detected:
        tier = 2  # Conservative Build
        notes.append("Contributions enabled only")
    elif drip_detected:
        tier = 3  # DRIP Build
        notes.append("DRIP active only")
        confidence = "medium"
    else:
        tier = 1  # Modest Appreciation
        notes.append("No active contributions or DRIP detected")
        confidence = "low"

    tier_names = {
        1: "Modest Appreciation",
        2: "Conservative Build",
        3: "DRIP Build",
        4: "Leveraged Growth",
        5: "Optimized Growth",
        6: "Maximum Growth"
    }

    return {
        "tier": tier,
        "name": tier_names[tier],
        "confidence": confidence,
        "detection_basis": {
            "drip_detected": drip_detected,
            "dividend_count_6m": dividend_count,
            "contributions_detected": contributions_detected,
            "monthly_contribution_amount": settings.goal_monthly_contribution,
            "leverage_maintained": leverage_maintained,
            "current_ltv_pct": _round_pct(current_ltv * 100),
            "target_ltv_pct": _round_pct(target_ltv * 100),
            "notes": "; ".join(notes)
        }
    }


def _calculate_expected_value(
    starting_value: float,
    starting_margin: float,
    starting_income: float,
    months_elapsed: float,
    annual_appreciation_pct: float,
    monthly_contribution: float,
    drip_enabled: bool,
    maintain_ltv: bool,
    target_ltv_pct: float,
    portfolio_yield_pct: float
) -> dict:
    """
    Calculate expected portfolio state after N months based on tier assumptions.
    Mirrors the compounding logic from calculate_timeline but returns intermediate state.
    """
    portfolio_value = starting_value
    margin_balance = starting_margin or 0.0

    monthly_appreciation = annual_appreciation_pct / 12.0 / 100.0
    margin_apr = settings.margin_apr_current
    monthly_margin_rate = margin_apr / 12.0

    for _ in range(int(months_elapsed)):
        # 1. Price appreciation
        portfolio_value *= (1 + monthly_appreciation)

        # 2. Margin interest
        if maintain_ltv and margin_balance > 0:
            margin_interest = margin_balance * monthly_margin_rate
            margin_balance += margin_interest

        # 3. DRIP
        drip_amount = 0.0
        if drip_enabled and portfolio_yield_pct > 0:
            drip_amount = portfolio_value * portfolio_yield_pct / 12.0

        # 4. Additional margin to maintain LTV
        additional_margin = 0.0
        if maintain_ltv and target_ltv_pct > 0:
            total_before_margin = portfolio_value + drip_amount + monthly_contribution
            additional_margin = (target_ltv_pct * total_before_margin - margin_balance) / (1.0 - target_ltv_pct)
            additional_margin = max(0.0, additional_margin)
            margin_balance += additional_margin

        # 5. Add investments
        portfolio_value += drip_amount + monthly_contribution + additional_margin

    # Expected income grows with portfolio (assuming yield stays constant)
    if portfolio_yield_pct > 0:
        expected_monthly = portfolio_value * portfolio_yield_pct / 12.0
    else:
        # starting_income is already monthly — don't divide again
        expected_monthly = starting_income

    return {
        "portfolio_value": portfolio_value,
        "margin_balance": margin_balance,
        "monthly_income": expected_monthly
    }


def _build_goal_pace(
    conn: sqlite3.Connection,
    goal_tiers: dict,
    projected_monthly_income: float,
    total_market_value: float,
    margin_loan_balance: float,
    target_monthly: float,
    portfolio_yield_pct: float,
    as_of_date: date
) -> dict:
    """
    Build pace tracking toward monthly dividend goal with:
    - Auto-detected likely tier based on user behavior
    - Multiple time windows (MTD, QTD, YTD, 30d, 60d, 90d, since inception)
    - Expected vs actual comparisons for MV and income
    - Ahead/behind metrics and dollar amounts needed
    """
    if not goal_tiers or not target_monthly or target_monthly <= 0:
        return None

    # Auto-detect likely tier
    likely_tier_info = _detect_likely_tier(conn, projected_monthly_income, margin_loan_balance, total_market_value)
    likely_tier_num = likely_tier_info["tier"]

    # Get the tier details
    tiers_list = goal_tiers.get("tiers", [])
    likely_tier = next((t for t in tiers_list if t["tier"] == likely_tier_num), None)

    if not likely_tier:
        return None

    # Get portfolio inception date
    cur = conn.cursor()
    inception_row = cur.execute(
        """SELECT MIN(date) FROM investment_transactions
           WHERE transaction_type IN ('buy', 'buy_shares')"""
    ).fetchone()
    inception_date = date.fromisoformat(inception_row[0]) if inception_row and inception_row[0] else date(2024, 10, 31)

    # Define time windows
    today = as_of_date
    windows_def = {
        "mtd": (date(today.year, today.month, 1), today),
        "qtd": (date(today.year, ((today.month - 1) // 3) * 3 + 1, 1), today),
        "ytd": (date(today.year, 1, 1), today),
        "30d": (today - timedelta(days=30), today),
        "60d": (today - timedelta(days=60), today),
        "90d": (today - timedelta(days=90), today),
        "since_inception": (inception_date, today)
    }

    # Get tier assumptions
    assumptions = likely_tier.get("assumptions", {})
    annual_appreciation = assumptions.get("annual_appreciation_pct", 0.0)
    monthly_contribution = assumptions.get("monthly_contribution", 0.0)
    drip_enabled = assumptions.get("drip_enabled", False)
    maintain_ltv = assumptions.get("ltv_maintained", False)
    target_ltv = assumptions.get("target_ltv_pct", 0.0) / 100.0 if "target_ltv_pct" in assumptions else 0.0

    windows = {}

    for window_name, (start_date, end_date) in windows_def.items():
        # Get snapshot at start of window
        start_snapshot = cur.execute(
            """SELECT payload_json FROM snapshot_daily_current
               WHERE as_of_date_local <= ?
               ORDER BY as_of_date_local DESC LIMIT 1""",
            (start_date.isoformat(),)
        ).fetchone()

        if not start_snapshot:
            continue

        start_data = json.loads(start_snapshot[0])
        # Support both v4 (top-level totals/income) and v5 (portfolio.totals/income)
        start_totals = start_data.get("totals") or (start_data.get("portfolio") or {}).get("totals") or {}
        start_inc = start_data.get("income") or (start_data.get("portfolio") or {}).get("income") or {}
        start_mv = start_totals.get("market_value", 0.0)
        start_margin = start_totals.get("margin_loan_balance", 0.0)
        start_income = start_inc.get("projected_monthly_income", 0.0)
        start_yield = start_inc.get("portfolio_current_yield_pct", 0.0) / 100.0 if start_inc.get("portfolio_current_yield_pct") else 0.0

        # Calculate months elapsed
        days_elapsed = (end_date - start_date).days
        months_elapsed = days_elapsed / 30.0

        # Calculate expected state based on tier assumptions
        expected = _calculate_expected_value(
            starting_value=start_mv,
            starting_margin=start_margin,
            starting_income=start_income,
            months_elapsed=months_elapsed,
            annual_appreciation_pct=annual_appreciation,
            monthly_contribution=monthly_contribution,
            drip_enabled=drip_enabled,
            maintain_ltv=maintain_ltv,
            target_ltv_pct=target_ltv,
            portfolio_yield_pct=start_yield
        )

        # Actual state (current)
        actual_mv = total_market_value
        actual_margin = margin_loan_balance or 0.0
        actual_income = projected_monthly_income or 0.0

        # Calculate deltas
        mv_delta = actual_mv - expected["portfolio_value"]
        mv_delta_pct = (mv_delta / expected["portfolio_value"] * 100) if expected["portfolio_value"] > 0 else 0.0
        income_delta = actual_income - expected["monthly_income"]
        income_delta_pct = (income_delta / expected["monthly_income"] * 100) if expected["monthly_income"] > 0 else 0.0

        # Calculate pace (months ahead/behind)
        # If MV is higher than expected, that accelerates the timeline
        # Use the tier's required portfolio value to calculate impact
        required_pv = likely_tier.get("required_portfolio_value", 0.0)
        if required_pv and required_pv > 0:
            # How much of the gap did we close vs expected?
            expected_gap_closed = expected["portfolio_value"] - start_mv
            actual_gap_closed = actual_mv - start_mv
            extra_progress = actual_gap_closed - expected_gap_closed

            # Convert to months: if we closed extra ground, we're ahead
            if expected_gap_closed > 0:
                months_ahead_behind = (extra_progress / expected_gap_closed) * months_elapsed
            else:
                months_ahead_behind = 0.0
        else:
            months_ahead_behind = 0.0

        # Calculate amount needed to get back on track
        amount_needed = expected["portfolio_value"] - actual_mv if actual_mv < expected["portfolio_value"] else 0.0
        amount_surplus = actual_mv - expected["portfolio_value"] if actual_mv > expected["portfolio_value"] else 0.0

        # Determine if on track (at or above 95% of expected — being ahead counts)
        pct_of_pace = (actual_mv / expected["portfolio_value"] * 100) if expected["portfolio_value"] > 0 else 100.0
        on_track = pct_of_pace >= 95.0

        windows[window_name] = {
            "window_name": window_name.upper(),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "days_in_window": days_elapsed,
            "months_elapsed": _round_money(months_elapsed),
            "expected": {
                "portfolio_value": _round_money(expected["portfolio_value"]),
                "monthly_income": _round_money(expected["monthly_income"]),
                "margin_balance": _round_money(expected["margin_balance"])
            },
            "actual": {
                "portfolio_value": _round_money(actual_mv),
                "monthly_income": _round_money(actual_income),
                "margin_balance": _round_money(actual_margin)
            },
            "delta": {
                "portfolio_value": _round_money(mv_delta),
                "portfolio_value_pct": _round_pct(mv_delta_pct),
                "monthly_income": _round_money(income_delta),
                "monthly_income_pct": _round_pct(income_delta_pct)
            },
            "pace": {
                "months_ahead_behind": round(months_ahead_behind, 2),
                "pct_of_tier_pace": _round_pct(pct_of_pace),
                "on_track": on_track,
                "amount_needed": _round_money(amount_needed),
                "amount_surplus": _round_money(amount_surplus)
            }
        }

    # Calculate overall current pace (using since_inception as primary indicator)
    inception_pace = windows.get("since_inception", {}).get("pace", {})
    months_ahead_behind = inception_pace.get("months_ahead_behind", 0.0)

    # Revise goal date based on current pace
    original_months = likely_tier.get("months_to_goal")
    if original_months is not None:
        revised_months = max(0, original_months - months_ahead_behind)
        revised_goal_date = _add_months(as_of_date, int(revised_months))
    else:
        revised_months = None
        revised_goal_date = None

    # Determine pace category
    if months_ahead_behind >= 3:
        pace_category = "ahead"
    elif months_ahead_behind >= -3:
        pace_category = "on_track"
    elif months_ahead_behind >= -6:
        pace_category = "behind"
    else:
        pace_category = "off_track"

    # Factor breakdown (from since_inception window)
    inception_window = windows.get("since_inception", {})
    inception_mv_delta = inception_window.get("delta", {}).get("portfolio_value", 0.0)
    inception_income_delta = inception_window.get("delta", {}).get("monthly_income", 0.0)

    return {
        "likely_tier": likely_tier_info,
        "baseline_projection": {
            "tier_number": likely_tier_num,
            "tier_name": likely_tier.get("name"),
            "original_goal_date": likely_tier.get("estimated_goal_date"),
            "original_months_to_goal": likely_tier.get("months_to_goal"),
            "required_portfolio_value": likely_tier.get("required_portfolio_value"),
            "assumptions": assumptions
        },
        "inception_date": inception_date.isoformat(),
        "windows": windows,
        "current_pace": {
            "months_ahead_behind": round(months_ahead_behind, 2),
            "revised_goal_date": revised_goal_date.isoformat() if revised_goal_date else None,
            "revised_months_to_goal": int(revised_months) if revised_months is not None else None,
            "on_track": pace_category in ("ahead", "on_track"),
            "pace_category": pace_category
        },
        "factors": {
            "market_value_impact": _round_money(inception_mv_delta),
            "income_growth_impact": _round_money(inception_income_delta * 12),  # Annualized
            "description": "Factors based on since-inception performance vs expected tier trajectory"
        },
        "provenance": _provenance_entry(
            "derived",
            "internal",
            "goal_pace_tracking",
            ["goal_tiers", "snapshot_history", "transaction_history"],
            now_utc_iso(),
        ),
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
    # Reconcile: overwrite estimated dates with actual provider data
    from .corporate_actions import reconcile_estimates_with_provider
    reconcile_estimates_with_provider(conn)
    pay_lag_by_symbol = _load_symbol_pay_lag_days(conn)
    lm_ex_dates = _load_lm_ex_dates(conn)
    account_balances = _load_account_balances(conn)
    first_acquired_dates = _load_first_acquired_dates(conn)

    holding_symbols = set(holdings.keys())
    position_index = _build_position_index(conn, holding_symbols)
    div_by_symbol = defaultdict(list)
    for tx in div_tx:
        div_by_symbol[tx["symbol"]].append(tx)

    def _parse_iso(dt_str: str | None):
        if not dt_str:
            return None
        try:
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except ValueError:
            return None

    price_series = {}
    missing_prices = []
    prices_as_of_date = None
    prices_as_of_utc = None
    quote_map = getattr(md, "quotes", None) or {}
    for sym in holdings:
        series = _price_series(md.prices.get(sym))
        price_series[sym] = series

        quote = quote_map.get(sym) if isinstance(quote_map, dict) else None
        quote_price = None
        quote_as_of_dt = None
        if isinstance(quote, dict):
            if isinstance(quote.get("price"), (int, float)):
                quote_price = float(quote.get("price"))
            quote_as_of_dt = _parse_iso(quote.get("as_of_utc") or quote.get("fetched_at_utc"))
            if quote_as_of_dt and (prices_as_of_utc is None or quote_as_of_dt > prices_as_of_utc):
                prices_as_of_utc = quote_as_of_dt

        if series is None or series.empty:
            if quote_price is None:
                missing_prices.append(sym)
        else:
            last_date = series.index.max().date()
            if prices_as_of_date is None or last_date > prices_as_of_date:
                prices_as_of_date = last_date

    benchmark_symbol = settings.benchmark_primary
    benchmark_series = _price_series(md.prices.get(benchmark_symbol))

    holdings_out = []
    sources = []
    total_market_value = 0.0
    total_cost_basis = 0.0
    total_forward_12m = 0.0
    ex_dates_by_symbol = {}
    ex_date_patterns_by_symbol: dict[str, dict] = {}

    for sym in sorted(holdings.keys()):
        h = holdings[sym]
        series = price_series.get(sym)
        quote = quote_map.get(sym) if isinstance(quote_map, dict) else None
        quote_price = None
        quote_provider = None
        quote_fetched_at = None
        if isinstance(quote, dict):
            if isinstance(quote.get("price"), (int, float)):
                quote_price = float(quote.get("price"))
            quote_provider = quote.get("provider")
            quote_fetched_at = quote.get("fetched_at_utc")

        fetched_at = now_utc_iso()
        provider_name = "unknown"
        if md.provenance.get(sym):
            for entry in md.provenance[sym]:
                if entry.get("success"):
                    provider_name = entry.get("provider", "unknown")
                    break
            if provider_name == "unknown":
                provider_name = md.provenance[sym][0].get("provider", "unknown")

        last_price = None
        last_price_provenance = _provenance_entry("pulled", provider_name, "price_history_last", [sym], fetched_at)
        if quote_price is not None:
            last_price = quote_price
            last_price_provenance = _provenance_entry(
                "pulled",
                quote_provider or provider_name,
                "quote",
                [sym],
                quote_fetched_at or fetched_at,
            )
        elif series is not None and not series.empty:
            last_price = float(series.iloc[-1])
        mv = last_price * float(h["shares"]) if last_price is not None else None
        total_market_value += mv or 0.0
        total_cost_basis += float(h["cost_basis"]) if h.get("cost_basis") is not None else 0.0

        div_events = provider_divs.get(sym, [])
        ex_dates = _effective_ex_dates(sym, provider_divs, lm_ex_dates)
        ex_dates_past = [dt for dt in ex_dates if dt <= as_of_date_local]
        ex_dates_future = [dt for dt in ex_dates if dt > as_of_date_local]
        trailing_12m_div_ps = sum(
            ev["amount"]
            for ev in div_events
            if (as_of_date_local - timedelta(days=365)) <= ev["ex_date"] <= as_of_date_local
        )
        last_ex_date = ex_dates_past[-1] if ex_dates_past else (ex_dates[-1] if ex_dates else None)
        first_acquired = first_acquired_dates.get(sym)
        freq_start = as_of_date_local - timedelta(days=365)
        if first_acquired and freq_start < first_acquired <= as_of_date_local:
            freq_start = first_acquired
        freq_end = as_of_date_local + timedelta(days=DIVIDEND_CUT_LOOKAHEAD_DAYS)
        freq_ex_dates = [dt for dt in ex_dates if freq_start <= dt <= freq_end]
        frequency = (
            _frequency_from_recent_ex_dates(freq_ex_dates, recent=6)
            or _frequency_from_ex_dates(freq_ex_dates)
            or _frequency_from_recent_ex_dates(ex_dates_past, recent=6)
            or _frequency_from_ex_dates(ex_dates_past)
            or _frequency_from_recent_ex_dates(ex_dates, recent=6)
            or _frequency_from_ex_dates(ex_dates)
        )
        next_ex_date_est = ex_dates_future[0] if ex_dates_future else _next_ex_date_est(ex_dates_past)
        if next_ex_date_est is None and last_ex_date:
            next_ex_date_est = _fallback_next_ex_date(last_ex_date, pay_history.get(sym, []))
        ex_dates_by_symbol[sym] = ex_dates
        sym_pattern = _detect_ex_date_pattern(ex_dates)
        if sym_pattern:
            ex_date_patterns_by_symbol[sym] = sym_pattern
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
        ultimate_provenance = {
            "price_history": _provenance_entry("pulled", provider_name, "price_history", [sym], fetched_at),
            "benchmark_history": _provenance_entry("pulled", provider_name, "price_history", [benchmark_symbol], fetched_at),
            "last_price": last_price_provenance,
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
            first_acquired_dates.get(sym),
            position_index,
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
    pay_window_start = _month_start(as_of_date_local)
    pay_window_end = _month_end(as_of_date_local)
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
        pay_window_start,
        pay_window_end,
        pay_history,
        as_of_date_local,
        ex_date_est_by_symbol,
        position_index,
        pay_lag_by_symbol,
        ex_date_patterns_by_symbol,
    )
    event_projected = projected_alt or 0.0

    projected_vs_received = {
        "pct_of_projection": _round_pct(_safe_divide(received_mtd, event_projected) * 100 if event_projected else None),
        "projected": _round_money(event_projected),
        "received": _round_money(received_mtd),
        "difference": _round_money(event_projected - received_mtd),
        "expected_events": expected_events,
        "window": {
            "start": pay_window_start.isoformat(),
            "end": pay_window_end.isoformat(),
            "label": "current_month",
            "mode": "event_based",
        },
        "alt": {
            "projected": _round_money(projected_monthly_income),
            "mode": "yield_based",
        },
    }

    dividends = {
        "projected_vs_received": projected_vs_received,
        "windows": {"30d": window_30d, "ytd": window_ytd, "qtd": window_qtd},
        "realized_mtd": realized_mtd,
    }

    # dividends upcoming (pay-date based, exclude paid events)
    window_start = pay_window_start
    window_end = pay_window_end
    upcoming_events = []
    for ev in expected_events:
        pay_dt = _parse_date(ev.get("pay_date_est")) or _parse_date(ev.get("ex_date_est"))
        if not pay_dt:
            continue
        if pay_dt < window_start or pay_dt > window_end:
            continue
        upcoming_events.append(ev)

    projected_upcoming = sum(ev["amount_est"] for ev in upcoming_events if ev.get("amount_est") is not None)
    dividends_upcoming = {
        "projected": _round_money(projected_upcoming),
        "events": upcoming_events,
        "window": {"start": window_start.isoformat(), "end": window_end.isoformat(), "mode": "paydate"},
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
    portfolio_start = min(first_acquired_dates.values()) if first_acquired_dates else None
    income_stability = _income_stability_metrics(
        div_tx,
        as_of_date_local,
        window_months=12,
        start_date=portfolio_start,
    )
    per_ticker_cut_count = 0
    per_ticker_missed_count = 0
    per_ticker_seen = False
    for holding in holdings_out:
        rel = holding.get("dividend_reliability") or {}
        cut = rel.get("dividend_cuts_12m")
        missed = rel.get("missed_payments_12m")
        if isinstance(cut, (int, float)):
            per_ticker_cut_count += int(cut)
            per_ticker_seen = True
        if isinstance(missed, (int, float)):
            per_ticker_missed_count += int(missed)
            per_ticker_seen = True
    if per_ticker_seen:
        income_stability["dividend_cut_count_12m"] = per_ticker_cut_count
        income_stability["missed_payment_count_12m"] = per_ticker_missed_count
    income_growth = _income_growth_metrics(div_tx, as_of_date_local)
    risk["income_stability_score"] = income_stability.get("stability_score")

    drawdown_values = _append_current_value(portfolio_values, total_market_value, as_of_date_local)
    drawdown_status, recovery_metrics = _drawdown_analysis(drawdown_values)
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

    # Build comprehensive goal tiers
    goal_tiers = _build_goal_tiers(
        income["projected_monthly_income"] or 0.0,
        total_market_value,
        margin_loan_balance,
        settings.goal_target_monthly,
        performance,
        as_of_date_local,
    )

    # Extract most optimistic tier (tier 6) for display alongside existing goal_progress
    goal_progress_optimistic = None
    if goal_tiers and goal_tiers.get("tiers"):
        tier6 = next((t for t in goal_tiers["tiers"] if t["tier"] == 6), None)
        if tier6:
            goal_progress_optimistic = {
                "tier_name": tier6["name"],
                "description": tier6["description"],
                "months_to_goal": tier6["months_to_goal"],
                "estimated_goal_date": tier6["estimated_goal_date"],
                "required_portfolio_value": tier6["required_portfolio_value"],
                "final_portfolio_value": tier6["final_portfolio_value"],
                "assumptions": tier6["assumptions"],
            }

    # Build pace tracking toward goal
    goal_pace = _build_goal_pace(
        conn,
        goal_tiers,
        income["projected_monthly_income"] or 0.0,
        total_market_value,
        margin_loan_balance,
        settings.goal_target_monthly,
        income.get("portfolio_current_yield_pct", 0.0) / 100.0 if income.get("portfolio_current_yield_pct") else 0.0,
        as_of_date_local,
    )

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
        "changes": ["Added goal pace tracking with auto-detected tier and time windows"],
        "cache": {
            "pricing": {"ttl_seconds": settings.cache_ttl_hours * 3600, "bypassed": False},
            "yf_dividends": {"ttl_seconds": settings.cache_ttl_hours * 3600, "bypassed": False},
            "quotes": {"ttl_seconds": settings.quote_ttl_minutes * 60, "bypassed": False},
        },
        "snapshot_age_days": 0,
        "schema_version": "4.0",
        "filled_from_existing": False,
    }

    plaid_account_id = None
    if settings.lm_plaid_account_ids:
        try:
            plaid_account_id = int(settings.lm_plaid_account_ids.split(",")[0].strip())
        except ValueError:
            plaid_account_id = settings.lm_plaid_account_ids.split(",")[0].strip()

    macro = _build_macro_snapshot(md, as_of_date_local)

    if prices_as_of_date is None:
        prices_as_of_date = as_of_date_local
    if prices_as_of_utc:
        prices_as_of_date = max(prices_as_of_date, prices_as_of_utc.date())

    daily = {
        "as_of": as_of_date_str,
        "as_of_utc": as_of_utc,
        "as_of_date_local": as_of_date_str,
        "plaid_account_id": plaid_account_id,
        "prices_as_of": prices_as_of_date.isoformat() if prices_as_of_date else as_of_date_str,
        "prices_as_of_utc": prices_as_of_utc.isoformat() if prices_as_of_utc else None,
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
        "goal_progress_optimistic": goal_progress_optimistic,
        "goal_tiers": goal_tiers,
        "goal_pace": goal_pace,
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

    # ── Transform to V5 schema ───────────────────────────────────────────
    from .snapshot_v5 import transform_to_v5
    daily = transform_to_v5(daily)

    return daily, sources


def persist_daily_snapshot(conn: sqlite3.Connection, daily: dict, run_id: str, force: bool = False) -> bool:
    # Get as_of_date_local from V5 timestamps or fall back to V4 format
    timestamps = daily.get("timestamps") or {}
    as_of_date_local = timestamps.get("portfolio_data_as_of_local") or daily.get("as_of_date_local")
    if not as_of_date_local:
        as_of_utc = timestamps.get("portfolio_data_as_of_utc") or daily.get("as_of_utc") or ""
        as_of_date_local = as_of_utc[:10] if as_of_utc else ""
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
    # Get as_of_date_local from V5 timestamps or fall back to V4 format
    timestamps = daily.get("timestamps") or {}
    as_of_date_local = timestamps.get("portfolio_data_as_of_local") or daily.get("as_of_date_local")
    if not as_of_date_local:
        as_of_utc = timestamps.get("portfolio_data_as_of_utc") or daily.get("as_of_utc") or ""
        as_of_date_local = as_of_utc[:10] if as_of_utc else ""
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

        # Clean up rolling snapshots for this completed period
        rolling_type = f"{period}_ROLLING"
        deleted = cur.execute(
            """
            DELETE FROM snapshots
            WHERE period_type=? AND period_start_date=?
            """,
            (rolling_type, str(start)),
        )
        deleted_count = deleted.rowcount
        if deleted_count > 0:
            conn.commit()
            log.info("rolling_snapshots_cleaned", period=period, start=str(start), deleted=deleted_count)


def persist_rolling_summaries(conn: sqlite3.Connection, run_id: str, daily: dict, as_of_date_local: str):
    """
    Persist rolling period summaries (week-to-date, month-to-date, etc.) whenever daily snapshot updates.
    These summaries are stored with period_type WEEK_ROLLING, MONTH_ROLLING, etc.
    """
    from . import snap_compat as sc
    
    log = structlog.get_logger()
    # as_of_date_local is passed from DB column (V5 stores in timestamps.portfolio_data_as_of_local)
    dt = date.fromisoformat(as_of_date_local)

    # Define rolling period types
    rolling_periods = [
        ("WEEK_ROLLING", "weekly"),
        ("MONTH_ROLLING", "monthly"),
        ("QUARTER_ROLLING", "quarterly"),
        ("YEAR_ROLLING", "yearly"),
    ]

    for period_db_type, period_snapshot_type in rolling_periods:
        try:
            # Calculate period bounds for this rolling type
            start, end = _period_bounds(period_db_type.replace("_ROLLING", ""), dt)

            # Skip if this date is the period end - a final snapshot should exist or will be created
            # Rolling summaries are only for incomplete periods
            if dt == end:
                log.debug("rolling_snapshot_skipped_period_complete", period=period_db_type, date=str(dt))
                continue

            # Build the rolling summary using existing build_period_snapshot logic
            from .periods import build_period_snapshot
            snapshot = build_period_snapshot(
                conn,
                snapshot_type=period_snapshot_type,
                as_of=str(dt),
                mode="to_date"
            )
        except Exception as exc:
            log.error("rolling_snapshot_build_failed", run_id=run_id, period=period_db_type, err=str(exc))
            continue

        # Validate the snapshot
        ok, reasons = validate_period_snapshot(snapshot)
        if not ok:
            log.error("rolling_snapshot_invalid", run_id=run_id, period=period_db_type, reasons=reasons)
            continue

        # Check if it changed before persisting (check latest rolling snapshot for this period)
        payload_sha = sha256_json(snapshot)
        cur = conn.cursor()
        existing = cur.execute(
            """
            SELECT payload_sha256, period_end_date
            FROM snapshots
            WHERE period_type=? AND period_start_date=?
            ORDER BY period_end_date DESC
            LIMIT 1
            """,
            (period_db_type, str(start)),
        ).fetchone()

        if existing and existing[0] == payload_sha and existing[1] == str(dt):
            log.debug("rolling_snapshot_unchanged", period=period_db_type, start=str(start), end=str(dt))
            continue

        # Delete any existing rolling snapshot for this period (we only keep the latest one)
        deleted = cur.execute(
            """
            DELETE FROM snapshots
            WHERE period_type=? AND period_start_date=?
            """,
            (period_db_type, str(start)),
        )
        if deleted.rowcount > 0:
            log.debug("rolling_snapshot_replaced", period=period_db_type, start=str(start), old_count=deleted.rowcount)

        # Persist the new rolling summary
        cur.execute(
            """
            INSERT INTO snapshots(snapshot_id, period_type, period_start_date, period_end_date, built_from_run_id, payload_json, payload_sha256, created_at_utc)
            VALUES(?,?,?,?,?,?,?,?)
            """,
            (str(uuid.uuid4()), period_db_type, str(start), str(dt), run_id, json.dumps(snapshot), payload_sha, now_utc_iso()),
        )
        conn.commit()
        log.info("rolling_snapshot_persisted", period=period_db_type, start=str(start), end=str(dt))


def backfill_rolling_summaries(conn: sqlite3.Connection, start_date: str | None = None, end_date: str | None = None):
    """
    Backfill rolling summaries for all existing daily snapshots.
    Useful for initial setup or after pulling a fresh DB from server.

    Args:
        conn: Database connection
        start_date: Optional start date (YYYY-MM-DD) to limit backfill range
        end_date: Optional end date (YYYY-MM-DD) to limit backfill range
    """
    log = structlog.get_logger()
    cur = conn.cursor()

    # Get all daily snapshot dates
    query = "SELECT as_of_date_local, payload_json FROM snapshot_daily_current"
    params = []

    if start_date and end_date:
        query += " WHERE as_of_date_local BETWEEN ? AND ?"
        params = [start_date, end_date]
    elif start_date:
        query += " WHERE as_of_date_local >= ?"
        params = [start_date]
    elif end_date:
        query += " WHERE as_of_date_local <= ?"
        params = [end_date]

    query += " ORDER BY as_of_date_local ASC"

    rows = cur.execute(query, params).fetchall()
    total = len(rows)

    if total == 0:
        log.info("backfill_no_daily_snapshots")
        return

    log.info("backfill_rolling_summaries_started", total_dates=total, start_date=start_date, end_date=end_date)

    run_id = "backfill_" + now_utc_iso().replace(":", "").replace("-", "").replace(".", "")[:19]
    processed = 0
    failed = 0

    for as_of_date_local, payload_json in rows:
        try:
            daily = json.loads(payload_json)
            dt = date.fromisoformat(as_of_date_local)

            # Create rolling summaries for this date (skips if period end)
            persist_rolling_summaries(conn, run_id, daily, as_of_date_local)

            # If this is a period end date, clean up rolling snapshots for completed periods
            for period in ["WEEK", "MONTH", "QUARTER", "YEAR"]:
                start, end = _period_bounds(period, dt)
                if dt == end:
                    # Check if final snapshot exists for this period
                    final_exists = cur.execute(
                        "SELECT 1 FROM snapshots WHERE period_type=? AND period_start_date=? AND period_end_date=?",
                        (period, str(start), str(end)),
                    ).fetchone()

                    if final_exists:
                        # Clean up rolling snapshots for this completed period
                        rolling_type = f"{period}_ROLLING"
                        deleted = cur.execute(
                            "DELETE FROM snapshots WHERE period_type=? AND period_start_date=?",
                            (rolling_type, str(start)),
                        )
                        if deleted.rowcount > 0:
                            conn.commit()
                            log.debug("backfill_cleanup_rolling", period=period, start=str(start), deleted=deleted.rowcount)

            processed += 1
            if processed % 10 == 0:
                log.info("backfill_progress", processed=processed, total=total)
        except Exception as exc:
            failed += 1
            log.error("backfill_date_failed", date=as_of_date_local, err=str(exc))
            continue

    log.info("backfill_rolling_summaries_complete", processed=processed, failed=failed, total=total)


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

import calendar
import json
import math
import sqlite3
import statistics
import uuid
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from ..config import settings
import structlog
from ..utils import sha256_json, now_utc_iso, to_local_date, to_local_datetime_iso
from .validation import validate_period_snapshot
from . import metrics

TRADE_TYPES = {"buy", "buy_shares", "reinvest", "reinvestment", "sell", "sell_shares", "redemption"}


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


def _load_provider_dividends(conn: sqlite3.Connection):
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT symbol, ex_date, amount, source
        FROM dividend_events_provider
        """
    ).fetchall()
    out = defaultdict(list)
    seen = set()
    for symbol, ex_date, amount, _source in rows:
        sym = str(symbol).upper() if symbol else None
        dt = _parse_date(ex_date)
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
        out[sym].append({"ex_date": dt, "amount": amt})
    for sym in out:
        out[sym].sort(key=lambda item: item["ex_date"])
    return out


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


def _income_stability_score(div_tx: list[dict], as_of_date: date, window_months: int = 6) -> float:
    totals_by_month: dict[tuple[int, int], float] = defaultdict(float)
    for tx in div_tx:
        dt = tx.get("date")
        if dt is None or dt > as_of_date:
            continue
        totals_by_month[(dt.year, dt.month)] += float(tx.get("amount") or 0.0)

    months = _month_keys(as_of_date, window_months)
    totals = [totals_by_month.get(key, 0.0) for key in months]
    if not totals:
        return 0.0
    mean = sum(totals) / len(totals)
    if mean <= 0:
        return 0.0
    stdev = statistics.pstdev(totals)
    cv = stdev / mean if mean else 0.0
    score = max(0.0, 100.0 - min(100.0, cv * 100.0))
    return round(score, 2)


def _symbol_metrics(series, benchmark_series):
    if series is None or series.empty:
        return {}
    returns = metrics.time_weighted_returns(series)
    benchmark_returns = metrics.time_weighted_returns(benchmark_series) if benchmark_series is not None else None
    beta, corr = metrics.beta_and_corr(returns, benchmark_returns, window_days=365)
    max_dd, dd_dur = metrics.max_drawdown(series, window_days=365)
    var_95, cvar_95 = metrics.var_cvar(returns, alpha=0.05, window_days=365)
    vol_30d = metrics.annualized_volatility(returns, window_days=30)
    vol_90d = metrics.annualized_volatility(returns, window_days=90)
    downside = metrics.downside_deviation(returns, window_days=365)
    sharpe = metrics.sharpe_ratio(returns, window_days=365)
    sortino = metrics.sortino_ratio(returns, window_days=365)
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
        "downside_dev_1y_pct": _round_pct(downside * 100 if downside is not None else None),
        "sharpe_1y": _round_pct(sharpe),
        "calmar_1y": _round_pct(calmar),
        "drawdown_duration_1y_days": dd_dur,
        "var_95_1d_pct": _round_pct(var_95 * 100 if var_95 is not None else None),
        "cvar_95_1d_pct": _round_pct(cvar_95 * 100 if cvar_95 is not None else None),
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
    provider_divs = _load_provider_dividends(conn)
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
        ex_dates = [ev["ex_date"] for ev in div_events]
        trailing_12m_div_ps = sum(
            ev["amount"]
            for ev in div_events
            if ev["ex_date"] >= as_of_date_local - timedelta(days=365)
        )
        last_ex_date = ex_dates[-1] if ex_dates else None
        frequency = _frequency_from_ex_dates(ex_dates)
        next_ex_date_est = _next_ex_date_est(ex_dates)
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
            "downside_dev_1y_pct",
            "calmar_1y",
            "var_95_1d_pct",
            "cvar_95_1d_pct",
            "corr_1y",
            "beta_3y",
            "twr_1m_pct",
            "twr_3m_pct",
            "twr_6m_pct",
            "twr_12m_pct",
        ]:
            ultimate_provenance[key] = _provenance_entry("derived", "internal", "price_history_metrics", [sym], fetched_at)

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
            "dividends_30d": _round_money(div_30d),
            "dividends_qtd": _round_money(div_qtd),
            "dividends_ytd": _round_money(div_ytd),
            "ultimate_provenance": ultimate_provenance,
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
            next_est = _next_ex_date_est([ev["ex_date"] for ev in provider_divs.get(sym, [])])
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
    income_stability_score = _income_stability_score(div_tx, as_of_date_local)
    risk["income_stability_score"] = income_stability_score

    portfolio_rollups = {
        "performance": performance,
        "benchmark": benchmark_symbol,
        "risk": risk,
        "meta": {"version": f"v{as_of_date_str}", "method": "approx-holdings"},
    }

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
        "income_stability_score",
        ["dividend_transactions", "window_months:6"],
        now_utc_iso(),
    )

    income_provenance = income_prov
    holdings_provenance = _provenance_entry("derived", "internal", "reconstruct_holdings", ["investment_transactions"], now_utc_iso())
    dividends_provenance = _provenance_entry("derived", "internal", "normalize_dividends", ["transactions"], now_utc_iso())
    dividends_upcoming_provenance = _provenance_entry("derived", "internal", "project_upcoming_dividends", ["holdings"], now_utc_iso())
    margin_guidance = _build_margin_guidance(total_market_value, margin_loan_balance, income["projected_monthly_income"] or 0.0)
    margin_guidance_provenance = _provenance_entry("derived", "internal", "margin_guidance_calculator", ["totals", "income"], now_utc_iso())

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
        "cache": {
            "pricing": {"ttl_seconds": settings.cache_ttl_hours * 3600, "bypassed": False},
            "yf_dividends": {"ttl_seconds": settings.cache_ttl_hours * 3600, "bypassed": False},
        },
        "snapshot_age_days": 0,
        "schema_version": "2.4",
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
        "coverage": coverage,
        "holdings_provenance": holdings_provenance,
        "totals_provenance": totals_prov,
        "income_provenance": income_provenance,
        "portfolio_rollups_provenance": portfolio_rollups_prov,
        "dividends_provenance": dividends_provenance,
        "dividends_upcoming_provenance": dividends_upcoming_provenance,
        "margin_guidance_provenance": margin_guidance_provenance,
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
            snapshot_type = period.lower()
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

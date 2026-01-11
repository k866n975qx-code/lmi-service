from __future__ import annotations

import json
import hashlib
import sqlite3
from datetime import date, datetime, timedelta
from typing import List, Tuple, Optional

import structlog
from .constants import (
    DIVIDEND_CUT_THRESHOLD,
    EXTENDED_DRAWDOWN_DAYS,
    GOAL_REQUIRED_INVESTMENT_DELTA,
    GOAL_SLIPPAGE_MONTHS,
    HY_SPREAD_CRITICAL,
    INCOME_BUNCHING_WEEK_PCT,
    INCOME_CONCENTRATION_WARNING,
    INCOME_STABILITY_MIN,
    INCOME_MISS_CRITICAL,
    INCOME_MISS_DAY_THRESHOLD,
    INCOME_SINGLE_SOURCE_WARNING,
    INCOME_VOLATILITY_30D_WARN,
    MARGIN_COVERAGE_MIN,
    MARGIN_INTEREST_INCOME_WARN_PCT,
    MARGIN_BUFFER_WARNING,
    MARGIN_BUFFER_CRITICAL,
    MARGIN_LTV_CRITICAL,
    MARGIN_LTV_RED,
    MARGIN_LTV_YELLOW,
    MAX_DRAWDOWN_WARNING,
    MILESTONE_MONTHLY_INCOME,
    MILESTONE_NET_VALUES,
    MILESTONE_PROGRESS_PCT,
    PORTFOLIO_VOL_WARNING,
    PORTFOLIO_SORTINO_MIN,
    PORTFOLIO_SORTINO_DROP,
    PORTFOLIO_SORTINO_SHARPE_GAP,
    POSITION_SORTINO_NEGATIVE,
    POSITION_LOSS_CRITICAL,
    POSITION_LOSS_SEVERE,
    SINGLE_POSITION_MAX,
    TAIL_RISK_CVAR_1D_CRITICAL,
    TAIL_RISK_CVAR_1W_CRITICAL,
    TAIL_RISK_INCOME_RATIO,
    TREASURY_SPIKE,
    VOL_EXPANSION_RATIO,
    VIX_CRITICAL,
    YIELD_COMPRESSION_WARNING,
)

log = structlog.get_logger()

def _latest_daily(conn: sqlite3.Connection) -> Tuple[Optional[str], Optional[dict]]:
    cur = conn.cursor()
    row = cur.execute(
        "SELECT as_of_date_local, payload_json FROM snapshot_daily_current ORDER BY updated_at_utc DESC LIMIT 1"
    ).fetchone()
    if not row:
        return None, None
    try:
        return row[0], json.loads(row[1])
    except json.JSONDecodeError:
        return row[0], None

def _load_daily_by_date(conn: sqlite3.Connection, as_of_date_local: str) -> Optional[dict]:
    cur = conn.cursor()
    row = cur.execute(
        "SELECT payload_json FROM snapshot_daily_current WHERE as_of_date_local=?",
        (as_of_date_local,),
    ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return None

def _previous_daily(conn: sqlite3.Connection, as_of_date_local: str) -> Tuple[Optional[str], Optional[dict]]:
    cur = conn.cursor()
    row = cur.execute(
        "SELECT as_of_date_local, payload_json FROM snapshot_daily_current WHERE as_of_date_local < ? ORDER BY as_of_date_local DESC LIMIT 1",
        (as_of_date_local,),
    ).fetchone()
    if not row:
        return None, None
    try:
        return row[0], json.loads(row[1])
    except json.JSONDecodeError:
        return row[0], None

def _snapshot_on_or_before(conn: sqlite3.Connection, target: date) -> Tuple[Optional[str], Optional[dict]]:
    cur = conn.cursor()
    row = cur.execute(
        "SELECT as_of_date_local, payload_json FROM snapshot_daily_current WHERE as_of_date_local <= ? ORDER BY as_of_date_local DESC LIMIT 1",
        (target.isoformat(),),
    ).fetchone()
    if not row:
        return None, None
    try:
        return row[0], json.loads(row[1])
    except json.JSONDecodeError:
        return row[0], None

def _recent_snapshots(conn: sqlite3.Connection, limit: int = 7) -> list[tuple[str, dict]]:
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT as_of_date_local, payload_json FROM snapshot_daily_current ORDER BY as_of_date_local DESC LIMIT ?",
        (limit,),
    ).fetchall()
    out = []
    for row in rows:
        try:
            out.append((row[0], json.loads(row[1])))
        except json.JSONDecodeError:
            continue
    return out

def _fp(category: str, title: str) -> str:
    h = hashlib.sha256()
    h.update(f"{category}:{title}".encode())
    return h.hexdigest()[:16]

def _mk(category: str, as_of_date_local: str, severity: int, title: str, body_html: str, period_type: str | None = None, details: dict | None = None):
    fingerprint = _fp(category, title)
    alert_id = f"{category}_{fingerprint}_{as_of_date_local}".replace(" ", "_")
    return {
        "id": alert_id,
        "fingerprint": fingerprint,
        "as_of_date_local": as_of_date_local,
        "period_type": period_type,
        "category": category,
        "severity": int(severity),
        "title": title,
        "body_html": body_html,
        "details": details,
    }

def _fmt_money(x):
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "—"

def _fmt_pct(x, precision: int = 2):
    try:
        return f"{float(x):.{precision}f}%"
    except Exception:
        return "—"

def _fmt_ratio(x, precision: int = 2):
    try:
        return f"{float(x):.{precision}f}"
    except Exception:
        return "—"

def _profile_label(profile: str | None) -> str | None:
    if not profile:
        return None
    return profile.replace("_", "-")

def _month_day_info(d: date):
    import calendar
    return d.day, calendar.monthrange(d.year, d.month)[1]

def _frequency_per_year(freq: str | None) -> int:
    if not freq:
        return 12
    freq = freq.lower()
    if "monthly" in freq:
        return 12
    if "quarter" in freq:
        return 4
    if "semi" in freq or "bi" in freq:
        return 2
    if "annual" in freq or "year" in freq:
        return 1
    return 12

def _dividend_last_two(conn: sqlite3.Connection, symbol: str):
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT ex_date, amount FROM dividend_events_provider WHERE symbol=? ORDER BY ex_date DESC LIMIT 5",
        (symbol,),
    ).fetchall()
    vals = [(r[0], r[1]) for r in rows if isinstance(r[1], (int, float))]
    if len(vals) >= 2:
        return vals[0], vals[1]
    return (vals[0] if vals else None), None

def _holdings_map(holdings: list[dict]) -> dict:
    out = {}
    for h in holdings:
        sym = h.get("symbol")
        if sym:
            out[sym] = h
    return out

def _realized_mtd_total(dividends: dict | None):
    if not dividends:
        return None
    val = dividends.get("realized_mtd")
    if isinstance(val, dict):
        return val.get("total_dividends")
    return val

def _income_coverage_now(margin_guidance: dict | None):
    if not margin_guidance:
        return None
    selected = margin_guidance.get("selected_mode")
    modes = margin_guidance.get("modes") or []
    for mode in modes:
        if mode.get("mode") == selected:
            return mode.get("income_interest_coverage_now")
    return modes[0].get("income_interest_coverage_now") if modes else None

def _margin_guidance_selected(margin_guidance: dict | None):
    if not margin_guidance:
        return None
    selected = margin_guidance.get("selected_mode")
    modes = margin_guidance.get("modes") or []
    for mode in modes:
        if mode.get("mode") == selected:
            return mode
    return modes[0] if modes else None

def _top3_income_pct(holdings: list[dict]):
    by_income = []
    for h in holdings:
        inc = h.get("forward_12m_dividend")
        if isinstance(inc, (int, float)):
            by_income.append((h.get("symbol"), float(inc)))
    if not by_income:
        return None, "", None
    by_income.sort(key=lambda x: x[1], reverse=True)
    total = sum(x[1] for x in by_income)
    top3 = by_income[:3]
    pct = (sum(x[1] for x in top3) / total * 100.0) if total else None
    max_single = (by_income[0][1] / total * 100.0) if total else None
    names = ", ".join([t[0] for t in top3])
    return (round(pct, 2) if pct is not None else None), names, (round(max_single, 2) if max_single is not None else None)

def _income_bunching(div_upcoming: dict, as_of_dt: date):
    events = (div_upcoming or {}).get("events") or []
    if not events:
        return None, None, None, 0
    import collections
    by_week = collections.defaultdict(float)
    exdates_count = 0
    for ev in events:
        ex = ev.get("ex_date_est") or ev.get("ex_date")
        amt = ev.get("amount_est")
        if not ex or not isinstance(amt, (int, float)):
            continue
        try:
            ex_dt = date.fromisoformat(ex)
        except Exception:
            continue
        if ex_dt.month != as_of_dt.month or ex_dt.year != as_of_dt.year:
            continue
        exdates_count += 1
        iso = ex_dt.isocalendar()
        key = f"{iso.year}-W{iso.week:02d}"
        by_week[key] += float(amt)
    if not by_week:
        return None, None, None, exdates_count
    total = sum(by_week.values())
    max_week_key, max_week_amt = max(by_week.items(), key=lambda kv: kv[1])
    pct = (max_week_amt / total * 100.0) if total else None
    return (round(pct, 2) if pct is not None else None), max_week_key, (round(max_week_amt, 2) if isinstance(max_week_amt, (int, float)) else None), exdates_count

def _next_exdates(events: list[dict], as_of_dt: date, days: int = 7):
    upcoming = []
    for ev in events:
        ex = ev.get("ex_date_est") or ev.get("ex_date")
        amt = ev.get("amount_est")
        if not ex:
            continue
        try:
            ex_dt = date.fromisoformat(ex)
        except Exception:
            continue
        if 0 <= (ex_dt - as_of_dt).days <= days:
            upcoming.append((ex_dt, ev.get("symbol"), amt))
    upcoming.sort(key=lambda x: x[0])
    return upcoming

def _consecutive_green_days(conn: sqlite3.Connection) -> tuple[int, float | None]:
    snaps = _recent_snapshots(conn, limit=7)
    if len(snaps) < 2:
        return 0, None
    snaps.sort(key=lambda x: x[0])  # oldest to newest
    consecutive = 0
    first_val = None
    last_val = None
    for i in range(1, len(snaps)):
        left = snaps[i - 1][1]
        right = snaps[i][1]
        lval = (left.get("totals") or {}).get("net_liquidation_value")
        rval = (right.get("totals") or {}).get("net_liquidation_value")
        if not isinstance(lval, (int, float)) or not isinstance(rval, (int, float)):
            continue
        if rval > lval:
            if consecutive == 0:
                first_val = lval
            consecutive += 1
            last_val = rval
        else:
            consecutive = 0
            first_val = None
            last_val = None
    gain_pct = None
    if consecutive >= 1 and first_val and last_val:
        gain_pct = ((last_val / first_val) - 1.0) * 100.0
    return consecutive, gain_pct

def evaluate_alerts(conn: sqlite3.Connection) -> List[dict]:
    as_of, snap = _latest_daily(conn)
    if not as_of or not snap:
        return []
    alerts: List[dict] = []
    totals = snap.get("totals") or {}
    income = snap.get("income") or {}
    dividends = snap.get("dividends") or {}
    holdings = snap.get("holdings") or []
    div_upcoming = snap.get("dividends_upcoming") or {}
    portfolio_rollups = snap.get("portfolio_rollups") or {}
    risk = (portfolio_rollups.get("risk") or {})
    performance = (portfolio_rollups.get("performance") or {})
    macro = (snap.get("macro") or {}).get("snapshot") or {}
    goal = snap.get("goal_progress") or {}
    goal_net = snap.get("goal_progress_net") or {}
    margin_guidance = snap.get("margin_guidance") or {}

    try:
        as_of_dt = date.fromisoformat(as_of)
    except Exception:
        as_of_dt = date.today()

    prev_date, prev_snap = _previous_daily(conn, as_of)
    prev_7d_date, prev_7d_snap = _snapshot_on_or_before(conn, as_of_dt - timedelta(days=7))
    prev_30d_date, prev_30d_snap = _snapshot_on_or_before(conn, as_of_dt - timedelta(days=30))

    prev_holdings = _holdings_map((prev_snap or {}).get("holdings") or []) if prev_snap else {}

    # 1) Dividend cut detection
    total_forward_12m = income.get("forward_12m_total")
    for pos in holdings:
        sym = pos.get("symbol")
        if not sym:
            continue
        last, prev = _dividend_last_two(conn, sym)
        if not last or not prev:
            continue
        (_, amt_last) = last
        (_, amt_prev) = prev
        if not isinstance(amt_last, (int, float)) or not isinstance(amt_prev, (int, float)) or amt_prev <= 0:
            continue
        cut_pct = 1.0 - (amt_last / amt_prev)
        if cut_pct < DIVIDEND_CUT_THRESHOLD:
            continue
        shares = pos.get("shares")
        freq = _frequency_per_year((pos.get("ultimate") or {}).get("distribution_frequency"))
        annual_impact = None
        impact_pct = None
        if isinstance(shares, (int, float)):
            annual_impact = (amt_prev - amt_last) * float(shares) * freq
            if isinstance(total_forward_12m, (int, float)) and total_forward_12m:
                impact_pct = (annual_impact / total_forward_12m) * 100.0
        severity = 10 if (impact_pct is not None and impact_pct > 10.0) else 9
        fy = pos.get("current_yield_pct") or (pos.get("ultimate") or {}).get("forward_yield_pct")
        title = f"{sym} dividend cut"
        body = f"🚨 <b>Dividend Cut</b> — <b>{sym}</b><br/>Per-share: {amt_prev:.4f} → {amt_last:.4f} (-{cut_pct*100:.1f}%)."
        if isinstance(fy, (int, float)):
            body += f"<br/>Fwd yield now: {fy:.2f}%."
        if annual_impact is not None:
            body += f"<br/>Impact: {_fmt_money(annual_impact/12.0)}/mo."
        alerts.append(_mk("dividend", as_of, severity, title, body))

    # 2) Margin call risk
    ltv = totals.get("margin_to_portfolio_pct")
    net_change_pct = None
    if prev_snap:
        prev_nlv = (prev_snap.get("totals") or {}).get("net_liquidation_value")
        cur_nlv = totals.get("net_liquidation_value")
        if isinstance(prev_nlv, (int, float)) and isinstance(cur_nlv, (int, float)) and prev_nlv:
            net_change_pct = ((cur_nlv / prev_nlv) - 1.0) * 100.0
    coverage_now = _income_coverage_now(margin_guidance)
    if (
        isinstance(ltv, (int, float))
        and (ltv > MARGIN_LTV_CRITICAL
             or (ltv > MARGIN_LTV_RED and isinstance(net_change_pct, (int, float)) and net_change_pct < -2.0)
             or (isinstance(coverage_now, (int, float)) and coverage_now < MARGIN_COVERAGE_MIN))
    ):
        mode = _margin_guidance_selected(margin_guidance) or {}
        repay = mode.get("amount")
        title = "Margin Call Risk"
        body = f"🚨 <b>MARGIN CRITICAL</b><br/>LTV at <b>{ltv:.1f}%</b>."
        if isinstance(net_change_pct, (int, float)):
            body += f"<br/>Portfolio change today: {net_change_pct:.1f}%."
        if isinstance(coverage_now, (int, float)):
            body += f"<br/>Income coverage: {coverage_now:.2f}x."
        if isinstance(repay, (int, float)):
            body += f"<br/>Action: Repay {_fmt_money(repay)} to reach target."
        alerts.append(_mk("margin", as_of, 10, title, body))

    # 2b) Margin stress buffer
    margin_call = (margin_stress.get("stress_scenarios") or {}).get("margin_call_distance") or {}
    decline_pct = margin_call.get("portfolio_decline_to_call_pct")
    if isinstance(decline_pct, (int, float)):
        buffer_pct = abs(decline_pct)
        if buffer_pct < MARGIN_BUFFER_CRITICAL:
            title = "Margin Buffer Critical"
            body = f"🔴 <b>Margin Buffer</b><br/>Decline to call: {_fmt_pct(decline_pct,1)}."
            alerts.append(_mk("margin", as_of, 9, title, body))
        elif buffer_pct < MARGIN_BUFFER_WARNING:
            title = "Margin Buffer Low"
            body = f"⚠️ <b>Margin Buffer</b><br/>Decline to call: {_fmt_pct(decline_pct,1)}."
            alerts.append(_mk("margin", as_of, 6, title, body))

    shock = (margin_stress.get("stress_scenarios") or {}).get("interest_rate_shock") or {}
    for key, data in shock.items():
        coverage = data.get("income_coverage_ratio")
        if isinstance(coverage, (int, float)) and coverage < MARGIN_COVERAGE_MIN:
            title = "Margin Interest Coverage Risk"
            body = f"⚠️ <b>Rate Shock</b><br/>{key} coverage: {coverage:.2f}x."
            alerts.append(_mk("margin", as_of, 6, title, body))
            break

    # 3) Position blow-up
    for pos in holdings:
        sym = pos.get("symbol") or ""
        if not sym:
            continue
        unreal_pct = pos.get("unrealized_pct")
        mv = pos.get("market_value")
        price_drop = None
        prev_price = (prev_holdings.get(sym) or {}).get("last_price")
        cur_price = pos.get("last_price")
        if isinstance(prev_price, (int, float)) and isinstance(cur_price, (int, float)) and prev_price:
            price_drop = ((cur_price / prev_price) - 1.0) * 100.0
        trigger = False
        if isinstance(unreal_pct, (int, float)) and unreal_pct <= POSITION_LOSS_CRITICAL:
            trigger = True
        if isinstance(price_drop, (int, float)) and price_drop < -10.0 and isinstance(mv, (int, float)) and mv > 1000:
            trigger = True
        if not trigger:
            continue
        severity = 9 if isinstance(unreal_pct, (int, float)) and unreal_pct <= POSITION_LOSS_SEVERE else 8
        title = f"{sym} position blow-up"
        body = f"🚨 <b>{sym}</b> down <b>{_fmt_pct(unreal_pct, 1)}</b> from cost.<br/>Value {_fmt_money(mv)}."
        vol30 = (pos.get("ultimate") or {}).get("vol_30d_pct")
        if isinstance(vol30, (int, float)):
            body += f"<br/>30d vol: {vol30:.1f}%."
        alerts.append(_mk("position", as_of, severity, title, body))

    # 4) Income failure late-month
    day, days_in_month = _month_day_info(as_of_dt)
    proj_monthly = income.get("projected_monthly_income")
    realized_mtd = _realized_mtd_total(dividends)
    expected_mtd = None
    if isinstance(proj_monthly, (int, float)):
        expected_mtd = proj_monthly * (day / float(days_in_month))
    income_failure = False
    failure_lines = []
    if day >= INCOME_MISS_DAY_THRESHOLD and isinstance(expected_mtd, (int, float)) and isinstance(realized_mtd, (int, float)):
        if realized_mtd < expected_mtd * INCOME_MISS_CRITICAL:
            shortfall = expected_mtd - realized_mtd
            income_failure = True
            failure_lines.append(f"Projected by today: {_fmt_money(expected_mtd)}")
            failure_lines.append(f"Received: {_fmt_money(realized_mtd)} (short {_fmt_money(shortfall)}).")

    missing_events = []
    alt = (dividends.get("projected_vs_received") or {}).get("alt") or {}
    expected_events = alt.get("expected_events") or []
    received_by_symbol = (dividends.get("realized_mtd") or {}).get("by_symbol") or {}
    for ev in expected_events:
        sym = ev.get("symbol")
        pay_date_str = ev.get("pay_date_est")
        if not sym or not pay_date_str:
            continue
        try:
            pay_dt = date.fromisoformat(pay_date_str)
        except Exception:
            continue
        if (as_of_dt - pay_dt).days <= 5:
            continue
        if sym not in received_by_symbol:
            missing_events.append(f"{sym} (pay {pay_date_str})")
    if missing_events:
        income_failure = True
        miss_list = ", ".join(missing_events[:6])
        if len(missing_events) > 6:
            miss_list += f" (+{len(missing_events) - 6} more)"
        failure_lines.append(f"Missing payments: {miss_list}")

    if income_failure:
        title = "Income Failure"
        body = "🚨 <b>Income Failure</b>"
        if failure_lines:
            body += "<br/>" + "<br/>".join(failure_lines)
        alerts.append(_mk("income", as_of, 9, title, body))

    # 5) Macro regime shift
    ten_year = macro.get("ten_year_yield")
    vix = macro.get("vix")
    hy_spread = macro.get("hy_spread_bps")
    ten_year_prev = None
    if prev_7d_snap:
        ten_year_prev = ((prev_7d_snap.get("macro") or {}).get("snapshot") or {}).get("ten_year_yield")
    ten_year_delta = None
    if isinstance(ten_year, (int, float)) and isinstance(ten_year_prev, (int, float)):
        ten_year_delta = ten_year - ten_year_prev
    if (
        (isinstance(ten_year_delta, (int, float)) and ten_year_delta > TREASURY_SPIKE)
        or (isinstance(vix, (int, float)) and vix > VIX_CRITICAL)
        or (isinstance(hy_spread, (int, float)) and hy_spread > HY_SPREAD_CRITICAL)
    ):
        severity = 9 if isinstance(vix, (int, float)) and vix > 35 else 8
        title = "Macro Regime Shift"
        vix_s = f"{vix:.2f}" if isinstance(vix, (int, float)) else "—"
        ten_s = f"{ten_year:.2f}%" if isinstance(ten_year, (int, float)) else "—"
        hy_s = f"{hy_spread:.0f}" if isinstance(hy_spread, (int, float)) else "—"
        body = f"🚨 <b>Macro Shift</b><br/>VIX: {vix_s} | 10Y: {ten_s} | HY Spread: {hy_s} bps."
        alerts.append(_mk("macro", as_of, severity, title, body))

    # 6) Margin creep
    monthly_interest = None
    selected_mode = _margin_guidance_selected(margin_guidance)
    if selected_mode:
        monthly_interest = selected_mode.get("monthly_interest_now")
    interest_pct = None
    if isinstance(monthly_interest, (int, float)) and isinstance(proj_monthly, (int, float)) and proj_monthly:
        interest_pct = (monthly_interest / proj_monthly) * 100.0
    if (
        (isinstance(ltv, (int, float)) and ltv > MARGIN_LTV_YELLOW)
        or (isinstance(interest_pct, (int, float)) and interest_pct > MARGIN_INTEREST_INCOME_WARN_PCT)
    ):
        severity = 7 if isinstance(ltv, (int, float)) and ltv > 35 else 6
        title = "Margin Creep"
        body = f"⚠️ <b>Margin Creep</b><br/>LTV: {_fmt_pct(ltv,1)}."
        if isinstance(monthly_interest, (int, float)) and isinstance(interest_pct, (int, float)):
            body += f"<br/>Interest: {_fmt_money(monthly_interest)}/mo ({interest_pct:.1f}% of income)."
        alerts.append(_mk("margin", as_of, severity, title, body))

    # 7) Income concentration
    top3_pct, top3_names, max_single_pct = _top3_income_pct(holdings)
    if (
        isinstance(top3_pct, (int, float)) and top3_pct >= INCOME_CONCENTRATION_WARNING
    ) or (
        isinstance(max_single_pct, (int, float)) and max_single_pct >= INCOME_SINGLE_SOURCE_WARNING
    ):
        title = "Income Concentration"
        body = f"⚠️ <b>Income Concentration</b><br/>Top 3: {top3_names} ({top3_pct:.1f}%)."
        if isinstance(max_single_pct, (int, float)):
            body += f"<br/>Largest single source: {max_single_pct:.1f}%."
        alerts.append(_mk("income", as_of, 6, title, body))

    # 7b) Income stability signals
    stability_score = income_stability.get("stability_score")
    trend_6m = income_stability.get("income_trend_6m")
    income_vol = income_stability.get("income_volatility_30d_pct")
    cuts_12m = income_stability.get("dividend_cut_count_12m")
    if isinstance(stability_score, (int, float)) and stability_score < INCOME_STABILITY_MIN:
        title = "Income Stability Declining"
        body = f"⚠️ <b>Income Stability</b><br/>Score: {stability_score:.2f} (below {INCOME_STABILITY_MIN:.2f})."
        alerts.append(_mk("income", as_of, 6, title, body))
    if trend_6m == "declining":
        title = "Income Trend Declining"
        body = "📉 <b>Income Trend</b><br/>6m trend: declining."
        alerts.append(_mk("income", as_of, 6, title, body))
    if isinstance(income_vol, (int, float)) and income_vol > INCOME_VOLATILITY_30D_WARN:
        title = "Income Volatility Elevated"
        body = f"⚠️ <b>Income Volatility</b><br/>30d volatility: {income_vol:.1f}%."
        alerts.append(_mk("income", as_of, 6, title, body))
    if isinstance(cuts_12m, int) and cuts_12m > 0:
        title = "Portfolio Dividend Cuts"
        body = f"🔴 <b>Dividend Cuts</b><br/>{cuts_12m} cut(s) detected in last 12 months."
        alerts.append(_mk("income", as_of, 8, title, body))

    # 8) Volatility regime change (portfolio)
    vol30 = risk.get("vol_30d_pct")
    vol90 = risk.get("vol_90d_pct")
    if isinstance(vol30, (int, float)) and (vol30 > PORTFOLIO_VOL_WARNING):
        severity = 6 if vol30 > 15 else 5
        title = "Portfolio Volatility Regime"
        body = f"⚠️ <b>Volatility Regime</b><br/>30d vol: {vol30:.1f}% | 90d vol: {_fmt_pct(vol90,1)}."
        alerts.append(_mk("volatility", as_of, severity, title, body))

    # Per-holding volatility expansion
    for pos in holdings:
        sym = pos.get("symbol")
        if not sym:
            continue
        u = pos.get("ultimate") or {}
        hvol30 = u.get("vol_30d_pct")
        hvol90 = u.get("vol_90d_pct")
        if (
            isinstance(hvol30, (int, float))
            and isinstance(hvol90, (int, float))
            and hvol90 > 0
            and (hvol30 / hvol90) >= VOL_EXPANSION_RATIO
            and hvol30 > 15
        ):
            title = f"{sym} volatility expanding"
            body = f"⚠️ <b>{sym}</b> vol expanding<br/>30d {hvol30:.1f}% vs 90d {hvol90:.1f}%."
            alerts.append(_mk("volatility", as_of, 5, title, body))

    # 8b) Sortino risk signals
    sortino_now = risk.get("sortino_1y")
    sharpe_now = risk.get("sharpe_1y")
    sortino_prev = None
    if prev_7d_snap:
        sortino_prev = ((prev_7d_snap.get("portfolio_rollups") or {}).get("risk") or {}).get("sortino_1y")
    if sortino_prev is None and prev_snap:
        sortino_prev = ((prev_snap.get("portfolio_rollups") or {}).get("risk") or {}).get("sortino_1y")
    if isinstance(sortino_now, (int, float)) and sortino_now < PORTFOLIO_SORTINO_MIN:
        title = "Portfolio Sortino Low"
        body = f"⚠️ <b>Portfolio Sortino</b><br/>Sortino: {_fmt_ratio(sortino_now,2)} (below {PORTFOLIO_SORTINO_MIN:.2f})."
        alerts.append(_mk("risk", as_of, 7, title, body))
    if (
        isinstance(sortino_now, (int, float))
        and isinstance(sortino_prev, (int, float))
        and (sortino_prev - sortino_now) >= PORTFOLIO_SORTINO_DROP
    ):
        delta = sortino_now - sortino_prev
        title = "Sortino Declining"
        body = f"⚠️ <b>Sortino Decline</b><br/>{_fmt_ratio(sortino_prev,2)} → {_fmt_ratio(sortino_now,2)} ({delta:+.2f})."
        alerts.append(_mk("risk", as_of, 6, title, body))
    if isinstance(sortino_now, (int, float)) and isinstance(sharpe_now, (int, float)):
        gap = sharpe_now - sortino_now
        if gap > PORTFOLIO_SORTINO_SHARPE_GAP:
            title = "Downside-heavy Volatility"
            body = f"⚠️ <b>Downside Volatility</b><br/>Sortino {_fmt_ratio(sortino_now,2)} vs Sharpe {_fmt_ratio(sharpe_now,2)} (gap {gap:.2f})."
            alerts.append(_mk("risk", as_of, 6, title, body))
    for pos in holdings:
        sym = pos.get("symbol")
        if not sym:
            continue
        sortino_pos = pos.get("sortino_1y")
        if sortino_pos is None:
            sortino_pos = (pos.get("ultimate") or {}).get("sortino_1y")
        if isinstance(sortino_pos, (int, float)) and sortino_pos < POSITION_SORTINO_NEGATIVE:
            max_dd = (pos.get("ultimate") or {}).get("max_drawdown_1y_pct")
            title = f"{sym} Sortino Negative"
            body = f"🔴 <b>{sym}</b> Sortino {sortino_pos:.2f}"
            if isinstance(max_dd, (int, float)):
                body += f"<br/>Max DD: {_fmt_pct(max_dd,1)}."
            alerts.append(_mk("position", as_of, 8, title, body))

    # 8c) Tail risk signals
    cvar_1d = tail_risk.get("cvar_95_1d_pct")
    cvar_1w = tail_risk.get("cvar_95_1w_pct")
    cvar_ratio = tail_risk.get("cvar_to_income_ratio")
    tail_category = tail_risk.get("tail_risk_category")
    if isinstance(cvar_1d, (int, float)) and cvar_1d < TAIL_RISK_CVAR_1D_CRITICAL:
        title = "High Tail Risk (1d)"
        body = f"⚠️ <b>Tail Risk</b><br/>CVaR 1d: {_fmt_pct(cvar_1d,1)}."
        alerts.append(_mk("risk", as_of, 6, title, body))
    if isinstance(cvar_1w, (int, float)) and cvar_1w < TAIL_RISK_CVAR_1W_CRITICAL:
        title = "High Tail Risk (1w)"
        body = f"⚠️ <b>Tail Risk</b><br/>CVaR 1w: {_fmt_pct(cvar_1w,1)}."
        alerts.append(_mk("risk", as_of, 7, title, body))
    if isinstance(cvar_ratio, (int, float)) and cvar_ratio > TAIL_RISK_INCOME_RATIO:
        title = "Tail Risk vs Income"
        body = f"🔴 <b>Tail Risk</b><br/>Loss risk: {cvar_ratio:.1f} months of income."
        alerts.append(_mk("risk", as_of, 8, title, body))
    if tail_category == "severe":
        title = "Severe Tail Risk"
        body = "🔴 <b>Tail Risk</b><br/>Category: severe."
        alerts.append(_mk("risk", as_of, 8, title, body))

    # 9) Yield compression
    current_yield = income.get("portfolio_current_yield_pct")
    prev_yield = None
    if prev_30d_snap:
        prev_yield = (prev_30d_snap.get("income") or {}).get("portfolio_current_yield_pct")
    if (
        isinstance(current_yield, (int, float))
        and isinstance(prev_yield, (int, float))
        and (current_yield < (prev_yield - YIELD_COMPRESSION_WARNING))
    ):
        title = "Yield Compression"
        monthly_income = income.get("projected_monthly_income")
        capital_needed = None
        if isinstance(monthly_income, (int, float)) and current_yield:
            required_now = (monthly_income * 12.0) / (current_yield / 100.0)
            required_prev = (monthly_income * 12.0) / (prev_yield / 100.0)
            capital_needed = required_now - required_prev
        body = f"⚠️ <b>Yield Compression</b><br/>Current: {current_yield:.2f}% (was {prev_yield:.2f}%)."
        if isinstance(capital_needed, (int, float)):
            body += f"<br/>Capital needed for same income: {_fmt_money(capital_needed)}."
        alerts.append(_mk("yield", as_of, 6, title, body))

    # 10) Goal progress stall
    months_to_goal = goal.get("months_to_goal")
    additional_needed = goal.get("additional_investment_needed")
    additional_needed_net = goal_net.get("additional_investment_needed_now")
    prev_months = None
    prev_additional = None
    prev_additional_net = None
    if prev_30d_snap:
        prev_goal = prev_30d_snap.get("goal_progress") or {}
        prev_months = prev_goal.get("months_to_goal")
        prev_additional = prev_goal.get("additional_investment_needed")
        prev_goal_net = prev_30d_snap.get("goal_progress_net") or {}
        prev_additional_net = prev_goal_net.get("additional_investment_needed_now")
    if (
        isinstance(months_to_goal, (int, float))
        and isinstance(prev_months, (int, float))
        and (months_to_goal - prev_months) > GOAL_SLIPPAGE_MONTHS
    ) or (
        isinstance(additional_needed, (int, float))
        and isinstance(prev_additional, (int, float))
        and (additional_needed - prev_additional) > GOAL_REQUIRED_INVESTMENT_DELTA
    ) or (
        isinstance(additional_needed_net, (int, float))
        and isinstance(prev_additional_net, (int, float))
        and (additional_needed_net - prev_additional_net) > GOAL_REQUIRED_INVESTMENT_DELTA
    ):
        title = "Goal Progress Stall"
        body = f"⚠️ <b>Goal Progress</b><br/>Months to goal: {months_to_goal} (was {prev_months})."
        if isinstance(additional_needed, (int, float)) and isinstance(prev_additional, (int, float)):
            body += f"<br/>Required investment: {_fmt_money(additional_needed)} (was {_fmt_money(prev_additional)})."
        if isinstance(additional_needed_net, (int, float)) and isinstance(prev_additional_net, (int, float)):
            body += f"<br/>Net required investment: {_fmt_money(additional_needed_net)} (was {_fmt_money(prev_additional_net)})."
        alerts.append(_mk("goal", as_of, 6, title, body))

    # 11) Ex-date clustering / income bunching
    week_pct, week_key, week_amt, exdates_count = _income_bunching(div_upcoming, as_of_dt)
    if (
        isinstance(week_pct, (int, float)) and week_pct >= INCOME_BUNCHING_WEEK_PCT
    ) or (
        isinstance(exdates_count, int) and exdates_count < 3
    ):
        title = "Income Bunching"
        body = "⚠️ <b>Income Bunching</b>"
        if isinstance(week_pct, (int, float)) and week_key:
            body += f"<br/>{week_pct:.0f}% in {week_key} ({_fmt_money(week_amt)})."
        if isinstance(exdates_count, int):
            body += f"<br/>Ex-dates this month: {exdates_count}."
        alerts.append(_mk("income", as_of, 5, title, body))

    # 12) Extended drawdown
    dd_days = risk.get("drawdown_duration_1y_days")
    max_dd = risk.get("max_drawdown_1y_pct")
    if (
        isinstance(dd_days, (int, float)) and dd_days > EXTENDED_DRAWDOWN_DAYS
    ) or (
        isinstance(max_dd, (int, float)) and max_dd < MAX_DRAWDOWN_WARNING
    ):
        title = "Extended Drawdown"
        body = f"⚠️ <b>Extended Drawdown</b><br/>Duration: {dd_days} days | Max DD: {_fmt_pct(max_dd,1)}."
        alerts.append(_mk("drawdown", as_of, 5, title, body))

    # 13) Dividend pipeline
    events = div_upcoming.get("events") or []
    next_7d = _next_exdates(events, as_of_dt, days=7)
    if next_7d:
        total = sum(x[2] for x in next_7d if isinstance(x[2], (int, float)))
        lines = [f"{sym} {dt.isoformat()} ~{_fmt_money(amt)}" for dt, sym, amt in next_7d]
        title = "Dividend Pipeline"
        body = "📅 <b>Dividend Pipeline</b><br/>" + "<br/>".join(lines)
        body += f"<br/>Total: {_fmt_money(total)} expected."
        alerts.append(_mk("income", as_of, 3, title, body))

    # 14) Milestones
    prev_totals = (prev_snap.get("totals") or {}) if prev_snap else {}
    cur_net = totals.get("net_liquidation_value")
    prev_net = prev_totals.get("net_liquidation_value")
    for threshold in MILESTONE_NET_VALUES:
        if isinstance(cur_net, (int, float)) and isinstance(prev_net, (int, float)) and prev_net < threshold <= cur_net:
            title = f"Milestone: Net value {_fmt_money(threshold)}"
            body = f"🎉 <b>Milestone</b><br/>Net value: {_fmt_money(cur_net)}."
            alerts.append(_mk("milestone", as_of, 4, title, body))
    cur_monthly = income.get("projected_monthly_income")
    prev_monthly = (prev_snap.get("income") or {}).get("projected_monthly_income") if prev_snap else None
    for threshold in MILESTONE_MONTHLY_INCOME:
        if isinstance(cur_monthly, (int, float)) and isinstance(prev_monthly, (int, float)) and prev_monthly < threshold <= cur_monthly:
            title = f"Milestone: Monthly income {_fmt_money(threshold)}"
            body = f"🎉 <b>Milestone</b><br/>Projected monthly income: {_fmt_money(cur_monthly)}."
            alerts.append(_mk("milestone", as_of, 4, title, body))
    cur_progress = goal.get("progress_pct")
    prev_progress = (prev_snap.get("goal_progress") or {}).get("progress_pct") if prev_snap else None
    for threshold in MILESTONE_PROGRESS_PCT:
        if isinstance(cur_progress, (int, float)) and isinstance(prev_progress, (int, float)) and prev_progress < threshold <= cur_progress:
            title = f"Milestone: Goal progress {threshold}%"
            body = f"🎉 <b>Milestone</b><br/>Progress: {cur_progress:.1f}%."
            alerts.append(_mk("milestone", as_of, 4, title, body))

    # 15) Positive momentum
    green_days, gain_pct = _consecutive_green_days(conn)
    if green_days >= 5:
        title = "Positive Momentum"
        body = f"✅ <b>Momentum</b><br/>{green_days} green days"
        if isinstance(gain_pct, (int, float)):
            body += f" (+{gain_pct:.1f}%)."
        alerts.append(_mk("momentum", as_of, 2, title, body))

    # Position concentration warning (single position max)
    for pos in holdings:
        weight = pos.get("weight_pct")
        sym = pos.get("symbol")
        if sym and isinstance(weight, (int, float)) and weight >= SINGLE_POSITION_MAX:
            title = f"{sym} position concentration"
            body = f"⚠️ <b>Concentration</b><br/>{sym} weight at {weight:.1f}%."
            alerts.append(_mk("position", as_of, 6, title, body))

    return alerts

def evaluate_immediate_daily(conn: sqlite3.Connection) -> List[dict]:
    return [a for a in evaluate_alerts(conn) if int(a.get("severity", 0)) >= 8]

def build_daily_report_html(conn: sqlite3.Connection):
    from .storage import list_open_alerts

    as_of, snap = _latest_daily(conn)
    if not as_of or not snap:
        return None, "No daily snapshot available."

    totals = snap.get("totals") or {}
    income = snap.get("income") or {}
    dividends = snap.get("dividends") or {}
    div_upcoming = snap.get("dividends_upcoming") or {}
    rollups = snap.get("portfolio_rollups") or {}
    performance = rollups.get("performance") or {}
    risk = rollups.get("risk") or {}
    income_stability = rollups.get("income_stability") or {}
    tail_risk = rollups.get("tail_risk") or {}
    goal = snap.get("goal_progress") or {}
    goal_net = snap.get("goal_progress_net") or {}
    macro = (snap.get("macro") or {}).get("snapshot") or {}
    margin_guidance = snap.get("margin_guidance") or {}
    margin_stress = snap.get("margin_stress") or {}
    holdings = snap.get("holdings") or []

    try:
        as_of_dt = date.fromisoformat(as_of)
    except Exception:
        as_of_dt = date.today()

    prev_7d_date, prev_7d_snap = _snapshot_on_or_before(conn, as_of_dt - timedelta(days=7))
    _prev_date, prev_snap = _previous_daily(conn, as_of)
    nlv = totals.get("net_liquidation_value")
    prev_nlv = (prev_7d_snap or {}).get("totals", {}).get("net_liquidation_value") if prev_7d_snap else None
    week_delta = None
    week_delta_pct = None
    if isinstance(nlv, (int, float)) and isinstance(prev_nlv, (int, float)) and prev_nlv:
        week_delta = nlv - prev_nlv
        week_delta_pct = (nlv / prev_nlv - 1.0) * 100.0

    parts = []
    parts.append(f"📊 <b>Portfolio Daily Review</b> | {as_of}")

    parts.append("")
    parts.append("<b>NET VALUE</b>")
    parts.append(f"• Net: {_fmt_money(nlv)} ({_fmt_money(week_delta)} / {_fmt_pct(week_delta_pct,1)} W/W)")
    parts.append(f"• Market Value: {_fmt_money(totals.get('market_value'))}")
    parts.append(f"• Margin Loan: {_fmt_money(totals.get('margin_loan_balance'))} (LTV {_fmt_pct(totals.get('margin_to_portfolio_pct'),1)})")

    open_crit = list_open_alerts(conn, min_severity=8)
    if open_crit:
        parts.append("")
        parts.append("🚨 <b>CRITICAL ALERTS</b>")
        for a in open_crit[:5]:
            parts.append(f"[{a['severity']}] {a['title']}")

    open_warn = list_open_alerts(conn, min_severity=5, max_severity=7)
    if open_warn:
        parts.append("")
        parts.append("⚠️ <b>WARNINGS</b>")
        for a in open_warn[:5]:
            parts.append(f"[{a['severity']}] {a['title']}")

    proj_monthly = income.get("projected_monthly_income")
    mtd_realized = _realized_mtd_total(dividends)
    proj_vs = (dividends.get("projected_vs_received") or {})
    parts.append("")
    parts.append("<b>💰 INCOME UPDATE</b>")
    parts.append(f"• MTD: {_fmt_money(mtd_realized)} / {_fmt_money(proj_monthly)} projected")
    if isinstance(proj_vs.get("pct_of_projection"), (int, float)):
        parts.append(f"• MTD % of projection: {proj_vs.get('pct_of_projection'):.1f}%")
    parts.append(f"• Last 30d: {_fmt_money((dividends.get('windows') or {}).get('30d', {}).get('total_dividends'))}")

    events = div_upcoming.get("events") or []
    next_7d = _next_exdates(events, as_of_dt, days=7)
    if next_7d:
        parts.append("")
        parts.append("<b>Next 7 Days</b>")
        total = 0.0
        for dt, sym, amt in next_7d:
            if isinstance(amt, (int, float)):
                total += float(amt)
            parts.append(f"• {dt.isoformat()}: {sym} ~{_fmt_money(amt)}")
        parts.append(f"Total expected: {_fmt_money(total)}")

    parts.append("")
    parts.append("<b>📈 PERFORMANCE</b>")
    parts.append(f"• 1M: {_fmt_pct(performance.get('twr_1m_pct'),2)} | 3M: {_fmt_pct(performance.get('twr_3m_pct'),2)}")
    parts.append(f"• 6M: {_fmt_pct(performance.get('twr_6m_pct'),2)} | 1Y: {_fmt_pct(performance.get('twr_12m_pct'),2)}")

    parts.append("")
    parts.append("<b>📉 RISK</b>")
    sortino = risk.get("sortino_1y")
    sharpe = risk.get("sharpe_1y")
    prev_sortino = None
    if prev_snap:
        prev_sortino = ((prev_snap.get("portfolio_rollups") or {}).get("risk") or {}).get("sortino_1y")
    sortino_delta = None
    if isinstance(sortino, (int, float)) and isinstance(prev_sortino, (int, float)):
        sortino_delta = sortino - prev_sortino
    profile = None
    if isinstance(sortino, (int, float)) and isinstance(sharpe, (int, float)):
        if sortino > sharpe:
            profile = "upside_biased"
        elif sortino < sharpe:
            profile = "downside_biased"
        else:
            profile = "balanced"
    sortino_delta_text = f" ({sortino_delta:+.2f})" if isinstance(sortino_delta, (int, float)) else ""
    parts.append(
        f"• 30d Vol: {_fmt_pct(risk.get('vol_30d_pct'),2)} | Sharpe: {_fmt_ratio(sharpe,2)} | Sortino: {_fmt_ratio(sortino,2)}{sortino_delta_text}"
    )
    profile_label = _profile_label(profile)
    if profile_label:
        parts.append(f"• Volatility Profile: {profile_label}")
    parts.append(f"• Max DD: {_fmt_pct(risk.get('max_drawdown_1y_pct'),2)} | DD Days: {risk.get('drawdown_duration_1y_days', '—')}")
    stability_score = (rollups.get("income_stability") or {}).get("stability_score")
    stability_trend = (rollups.get("income_stability") or {}).get("income_trend_6m")
    if isinstance(stability_score, (int, float)):
        trend_text = stability_trend or "—"
        parts.append(f"• Income Stability: {_fmt_ratio(stability_score,2)} | Trend: {trend_text}")
    tail = rollups.get("tail_risk") or {}
    cvar_1d = tail.get("cvar_95_1d_pct")
    tail_cat = tail.get("tail_risk_category")
    if isinstance(cvar_1d, (int, float)) or tail_cat:
        parts.append(f"• CVaR 1d: {_fmt_pct(cvar_1d,1)} | Tail Risk: {tail_cat or '—'}")

    sortino_rows = []
    for h in holdings:
        sortino_val = h.get("sortino_1y")
        if sortino_val is None:
            sortino_val = (h.get("ultimate") or {}).get("sortino_1y")
        if isinstance(sortino_val, (int, float)):
            sortino_rows.append((h, float(sortino_val)))
    if sortino_rows:
        sortino_rows.sort(key=lambda item: item[1], reverse=True)
        parts.append("")
        parts.append("<b>🧭 SORTINO SNAPSHOT</b>")
        prev_7d_sortino = None
        if prev_7d_snap:
            prev_7d_sortino = ((prev_7d_snap.get("portfolio_rollups") or {}).get("risk") or {}).get("sortino_1y")
        if isinstance(sortino, (int, float)) and isinstance(prev_7d_sortino, (int, float)):
            trend_delta = sortino - prev_7d_sortino
            parts.append(f"7-Day Trend: {prev_7d_sortino:.2f} → {sortino:.2f} ({trend_delta:+.2f})")
        top = sortino_rows[:3]
        if top:
            parts.append("Top performers (Sortino):")
            for h, val in top:
                category = h.get("risk_quality_category") or (h.get("ultimate") or {}).get("risk_quality_category")
                label = f" ({category})" if category else ""
                parts.append(f"• {h.get('symbol')}: {val:.2f}{label}")
        low = [item for item in sortino_rows[::-1] if item[1] < PORTFOLIO_SORTINO_MIN]
        if not low:
            low = sortino_rows[-3:][::-1]
        if low:
            parts.append("Watch list (low Sortino):")
            for h, val in low[:3]:
                profile = h.get("volatility_profile") or (h.get("ultimate") or {}).get("volatility_profile")
                profile_label = _profile_label(profile)
                max_dd = (h.get("ultimate") or {}).get("max_drawdown_1y_pct")
                extra = []
                if profile_label:
                    extra.append(profile_label)
                if isinstance(max_dd, (int, float)):
                    extra.append(f"max DD {_fmt_pct(max_dd,1)}")
                extra_text = f" ({', '.join(extra)})" if extra else ""
                parts.append(f"• {h.get('symbol')}: {val:.2f}{extra_text}")

    parts.append("")
    parts.append("<b>🎯 GOAL PROGRESS</b>")
    parts.append(f"• Target: {_fmt_money(goal.get('target_monthly'))}/mo")
    parts.append(f"• Current: {_fmt_money(goal.get('current_projected_monthly'))}/mo ({goal.get('progress_pct', '—')}%)")
    parts.append(f"• Timeline: {goal.get('months_to_goal', '—')} months (ETA {goal.get('estimated_goal_date', '—')})")
    parts.append(f"• Net Reality: {_fmt_money(goal_net.get('current_projected_monthly_net'))}/mo after interest")

    parts.append("")
    parts.append("<b>📊 MACRO ENVIRONMENT</b>")
    vix = macro.get("vix")
    ten_year = macro.get("ten_year_yield")
    spread = macro.get("yield_spread_10y_2y")
    hy_spread = macro.get("hy_spread_bps")
    parts.append(f"• VIX: {vix if vix is not None else '—'} | 10Y: {ten_year if ten_year is not None else '—'}% | Spread: {spread if spread is not None else '—'}")
    parts.append(f"• HY Spread: {hy_spread if hy_spread is not None else '—'} bps | Stress: {macro.get('macro_stress_score', '—')}")

    # Position highlights
    top_weight = None
    top_yield = None
    top_vol = None
    top_perf = None
    for h in holdings:
        w = h.get("weight_pct")
        y = h.get("current_yield_pct")
        u = h.get("ultimate") or {}
        vol = u.get("vol_30d_pct")
        perf = u.get("twr_1m_pct")
        if isinstance(w, (int, float)) and (top_weight is None or w > top_weight[1]):
            top_weight = (h.get("symbol"), w)
        if isinstance(y, (int, float)) and (top_yield is None or y > top_yield[1]):
            top_yield = (h.get("symbol"), y)
        if isinstance(vol, (int, float)) and (top_vol is None or vol > top_vol[1]):
            top_vol = (h.get("symbol"), vol)
        if isinstance(perf, (int, float)) and (top_perf is None or perf > top_perf[1]):
            top_perf = (h.get("symbol"), perf)

    parts.append("")
    parts.append("<b>💼 POSITION HIGHLIGHTS</b>")
    if top_perf:
        parts.append(f"• Top Performer (1M): {top_perf[0]} {top_perf[1]:.1f}%")
    if top_yield:
        parts.append(f"• Top Yielder: {top_yield[0]} {top_yield[1]:.2f}%")
    if top_weight:
        parts.append(f"• Largest Weight: {top_weight[0]} {top_weight[1]:.1f}%")
    if top_vol:
        parts.append(f"• Most Volatile: {top_vol[0]} {top_vol[1]:.1f}%")

    high_vol = [h for h in holdings if isinstance((h.get("ultimate") or {}).get("vol_30d_pct"), (int, float)) and (h.get("ultimate") or {}).get("vol_30d_pct") > 15]
    if high_vol:
        parts.append("High Vol Watch:")
        for h in high_vol[:5]:
            parts.append(f"• {h.get('symbol')} {h.get('ultimate', {}).get('vol_30d_pct'):.1f}%")

    return as_of, "\n".join(parts)

def build_daily_digest_html(conn: sqlite3.Connection):
    return build_daily_report_html(conn)

def build_period_report_html(conn: sqlite3.Connection, period: str):
    from ..pipeline.periods import build_period_snapshot

    try:
        snap = build_period_snapshot(conn, snapshot_type=period, mode="to_date")
    except Exception as exc:
        log.warning("period_snapshot_failed", period=period, err=str(exc))
        as_of, daily = _latest_daily(conn)
        if not as_of or not daily:
            return None, "No daily snapshot available."
        return build_daily_report_html(conn)

    summary = snap.get("period_summary") or {}
    totals = summary.get("totals") or {}
    income = summary.get("income") or {}
    performance = summary.get("performance") or {}
    risk = summary.get("risk") or {}
    macro = summary.get("macro") or {}
    goal = summary.get("goal_progress") or {}
    goal_net = summary.get("goal_progress_net") or {}
    period_label = (snap.get("period") or {}).get("label") or period

    header = {
        "weekly": "📈 <b>Weekly Recap</b>",
        "monthly": "📅 <b>Monthly Recap</b>",
        "quarterly": "📣 <b>Quarterly Review</b>",
        "yearly": "📣 <b>Yearly Review</b>",
    }.get(period, "📣 <b>Period Recap</b>")
    parts = [f"{header} — {period_label}", ""]

    parts.append("<b>💼 Totals</b>")
    parts.append(f"• MV {_fmt_money(totals.get('end', {}).get('total_market_value'))}")
    parts.append(f"• Net {_fmt_money(totals.get('end', {}).get('net_liquidation_value'))}")
    parts.append(f"• Δ Net {_fmt_money(totals.get('delta', {}).get('net_liquidation_value'))}")

    parts.append("")
    parts.append("<b>💰 Income</b>")
    parts.append(f"• Projected Monthly: {_fmt_money(income.get('end', {}).get('projected_monthly_income'))}")
    parts.append(f"• Fwd 12m: {_fmt_money(income.get('end', {}).get('forward_12m_total'))}")

    parts.append("")
    parts.append("<b>📈 Performance</b>")
    period_perf = (performance.get("period") or {})
    parts.append(f"• Period return: {_fmt_pct(period_perf.get('twr_period_pct'),2)}")
    twr = (performance.get("twr_windows") or {})
    parts.append(f"• 1M Δ: {_fmt_pct(twr.get('twr_1m_pct_delta'),2)} | 3M Δ: {_fmt_pct(twr.get('twr_3m_pct_delta'),2)}")
    parts.append(f"• 6M Δ: {_fmt_pct(twr.get('twr_6m_pct_delta'),2)} | 1Y Δ: {_fmt_pct(twr.get('twr_12m_pct_delta'),2)}")

    parts.append("")
    parts.append("<b>📉 Risk</b>")
    risk_end = risk.get("end", {}) if isinstance(risk, dict) else {}
    parts.append(
        f"• 30d Vol: {_fmt_pct(risk_end.get('vol_30d_pct'),2)} | Sharpe: {_fmt_ratio(risk_end.get('sharpe_1y'),2)} | Sortino: {_fmt_ratio(risk_end.get('sortino_1y'),2)}"
    )
    parts.append(f"• Max DD: {_fmt_pct(risk_end.get('max_drawdown_1y_pct'),2)}")

    parts.append("")
    parts.append("<b>🎯 Goal</b>")
    parts.append(f"• Progress: {goal.get('end', {}).get('progress_pct', '—')}%")
    parts.append(f"• Months to goal: {goal.get('end', {}).get('months_to_goal', '—')}")
    parts.append(f"• Net monthly: {_fmt_money(goal_net.get('end', {}).get('current_projected_monthly_net'))}")

    parts.append("")
    parts.append("<b>📊 Macro</b>")
    parts.append(f"• VIX: {macro.get('end', {}).get('vix', '—')} | 10Y: {macro.get('end', {}).get('ten_year_yield', '—')}")

    return snap.get("as_of"), "\n".join(parts)

def build_period_digest_html(conn: sqlite3.Connection, period: str):
    return build_period_report_html(conn, period)

from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import settings
from ..db import get_conn
from ..alerts.evaluator import evaluate_alerts, build_daily_report_html, build_period_report_html
from ..alerts.storage import (
    migrate_alerts,
    list_open_alerts,
    ack_alert,
    get_setting,
)
from ..alerts.notifier import (
    send_alerts,
    send_digest,
    set_silence,
    clear_silence,
    set_min_severity,
)
from ..pipeline.diff_daily import diff_daily_from_db
from ..services.telegram import TelegramClient

router = APIRouter()

class EvaluateRequest(BaseModel):
    period: str = "daily"  # daily|weekly|monthly|quarterly
    send: bool = False

def _telegram_ready() -> bool:
    return bool(getattr(settings, "telegram_bot_token", None))

def _chat_allowed(chat_id: int | str) -> bool:
    allowed = getattr(settings, "telegram_chat_id", None)
    if not allowed:
        return True
    return str(chat_id) == str(allowed)

def _format_alerts_html(alerts: list[dict]) -> str:
    if not alerts:
        return "✅ No active alerts."
    lines = ["<b>Active Alerts</b>"]
    for a in alerts[:20]:
        lines.append(f"• <code>{a['id']}</code> ({a['severity']}) {a['title']}")
    if len(alerts) > 20:
        lines.append(f"…and {len(alerts) - 20} more.")
    lines.append("Use /ack &lt;id&gt; to acknowledge.")
    return "\n".join(lines)

def _help_text() -> str:
    return (
        "<b>Alert Bot Commands</b>\n"
        "/status - quick daily summary\n"
        "/alerts - list active alerts\n"
        "/ack &lt;id&gt; - acknowledge alert\n"
        "/ackall - acknowledge all\n"
        "/full - full daily digest\n"
        "/income - upcoming dividends\n"
        "/mtd - month-to-date income\n"
        "/received - last 30d dividend breakdown\n"
        "/perf - performance summary\n"
        "/holdings - top holdings by weight\n"
        "/risk - risk metrics\n"
        "/goal - goal progress\n"
        "/projection - time to goal\n"
        "/settings - current bot settings\n"
        "/silence &lt;hours&gt; - pause non-critical alerts\n"
        "/resume - resume alerts\n"
        "/threshold &lt;1-10&gt; - minimum severity\n"
        "/snapshot - latest snapshot metadata\n"
        "/compare - today vs yesterday\n"
        "/macro - macro environment\n"
        "/help - show this help"
    )

def _latest_snapshot(conn):
    cur = conn.cursor()
    row = cur.execute(
        "SELECT as_of_date_local, payload_json FROM snapshot_daily_current ORDER BY as_of_date_local DESC LIMIT 1"
    ).fetchone()
    if not row:
        return None, None
    import json
    try:
        return row[0], json.loads(row[1])
    except json.JSONDecodeError:
        return row[0], None

def _prev_snapshot_date(conn, as_of_date_local: str):
    cur = conn.cursor()
    row = cur.execute(
        "SELECT as_of_date_local FROM snapshot_daily_current WHERE as_of_date_local < ? ORDER BY as_of_date_local DESC LIMIT 1",
        (as_of_date_local,),
    ).fetchone()
    return row[0] if row else None

def _fmt_money(val):
    try:
        return f"${float(val):,.2f}"
    except Exception:
        return "—"

def _fmt_pct(val, precision: int = 2):
    try:
        return f"{float(val):.{precision}f}%"
    except Exception:
        return "—"

def _status_text(snap: dict) -> str:
    totals = snap.get("totals") or {}
    income = snap.get("income") or {}
    nlv = totals.get("net_liquidation_value")
    ltv = totals.get("margin_to_portfolio_pct")
    proj = income.get("projected_monthly_income")
    return (
        "<b>Quick Status</b>\n"
        f"Net: {_fmt_money(nlv)}\n"
        f"LTV: {_fmt_pct(ltv,1)}\n"
        f"Projected Monthly: {_fmt_money(proj)}"
    )

def _income_text(snap: dict) -> str:
    divs = snap.get("dividends_upcoming") or {}
    events = divs.get("events") or []
    if not events:
        return "No upcoming dividends in the current window."
    lines = ["<b>Upcoming Dividends</b>"]
    for ev in events[:10]:
        sym = ev.get("symbol")
        ex = ev.get("ex_date_est") or ev.get("ex_date")
        amt = ev.get("amount_est")
        lines.append(f"• {sym} {ex} ~{_fmt_money(amt)}")
    total = sum(ev.get("amount_est") or 0 for ev in events if isinstance(ev.get("amount_est"), (int, float)))
    lines.append(f"Total projected: {_fmt_money(total)}")
    return "\n".join(lines)

def _mtd_text(snap: dict) -> str:
    divs = snap.get("dividends") or {}
    proj_vs = divs.get("projected_vs_received") or {}
    projected = proj_vs.get("projected")
    received = proj_vs.get("received")
    pct = proj_vs.get("pct_of_projection")
    return (
        "<b>Month-to-Date Income</b>\n"
        f"Received: {_fmt_money(received)}\n"
        f"Projected: {_fmt_money(projected)}\n"
        f"% of projection: {_fmt_pct(pct,1)}"
    )

def _received_text(snap: dict) -> str:
    divs = snap.get("dividends") or {}
    window = (divs.get("windows") or {}).get("30d") or {}
    by_symbol = window.get("by_symbol") or {}
    lines = ["<b>Last 30 Days (by symbol)</b>"]
    for sym, info in sorted(by_symbol.items()):
        amt = info.get("amount")
        lines.append(f"• {sym}: {_fmt_money(amt)} ({info.get('status', 'n/a')})")
    lines.append(f"Total: {_fmt_money(window.get('total_dividends'))}")
    return "\n".join(lines)

def _perf_text(snap: dict) -> str:
    perf = (snap.get("portfolio_rollups") or {}).get("performance") or {}
    return (
        "<b>Performance</b>\n"
        f"1M: {_fmt_pct(perf.get('twr_1m_pct'),2)}\n"
        f"3M: {_fmt_pct(perf.get('twr_3m_pct'),2)}\n"
        f"6M: {_fmt_pct(perf.get('twr_6m_pct'),2)}\n"
        f"1Y: {_fmt_pct(perf.get('twr_12m_pct'),2)}"
    )

def _holdings_text(snap: dict) -> str:
    holdings = snap.get("holdings") or []
    holdings = [h for h in holdings if isinstance(h.get("weight_pct"), (int, float))]
    holdings.sort(key=lambda h: h.get("weight_pct") or 0.0, reverse=True)
    lines = ["<b>Top Holdings</b>"]
    for h in holdings[:10]:
        lines.append(f"• {h.get('symbol')}: {h.get('weight_pct'):.1f}% ({_fmt_money(h.get('market_value'))})")
    return "\n".join(lines)

def _risk_text(snap: dict) -> str:
    risk = (snap.get("portfolio_rollups") or {}).get("risk") or {}
    return (
        "<b>Risk</b>\n"
        f"30d Vol: {_fmt_pct(risk.get('vol_30d_pct'),2)}\n"
        f"90d Vol: {_fmt_pct(risk.get('vol_90d_pct'),2)}\n"
        f"Sharpe: {risk.get('sharpe_1y', '—')}\n"
        f"Max DD: {_fmt_pct(risk.get('max_drawdown_1y_pct'),2)}"
    )

def _goal_text(snap: dict) -> str:
    goal = snap.get("goal_progress") or {}
    goal_net = snap.get("goal_progress_net") or {}
    return (
        "<b>Goal Progress</b>\n"
        f"Target: {_fmt_money(goal.get('target_monthly'))}/mo\n"
        f"Current: {_fmt_money(goal.get('current_projected_monthly'))}/mo\n"
        f"Progress: {goal.get('progress_pct', '—')}%\n"
        f"Months to goal: {goal.get('months_to_goal', '—')}\n"
        f"Net: {_fmt_money(goal_net.get('current_projected_monthly_net'))}/mo"
    )

def _projection_text(snap: dict) -> str:
    goal = snap.get("goal_progress") or {}
    return (
        "<b>Projection</b>\n"
        f"Months to goal: {goal.get('months_to_goal', '—')}\n"
        f"Estimated date: {goal.get('estimated_goal_date', '—')}\n"
        f"Required investment: {_fmt_money(goal.get('additional_investment_needed'))}"
    )

def _settings_text(conn) -> str:
    min_sev = get_setting(conn, "min_severity", "1")
    silenced = get_setting(conn, "silenced_until_utc", "")
    return (
        "<b>Settings</b>\n"
        f"Daily digest: {getattr(settings, 'alerts_daily_hour', 7):02d}:{getattr(settings, 'alerts_daily_minute', 30):02d}\n"
        f"Min severity: {min_sev}\n"
        f"Silenced until: {silenced or 'none'}"
    )

def _snapshot_text(snap: dict) -> str:
    meta = snap.get("meta") or {}
    return (
        "<b>Snapshot</b>\n"
        f"As of: {snap.get('as_of_date_local', '—')}\n"
        f"Created: {meta.get('snapshot_created_at', '—')}\n"
        f"Schema: {meta.get('schema_version', '—')}\n"
        f"Age days: {meta.get('snapshot_age_days', '—')}"
    )

def _macro_text(snap: dict) -> str:
    macro = (snap.get("macro") or {}).get("snapshot") or {}
    return (
        "<b>Macro</b>\n"
        f"VIX: {macro.get('vix', '—')}\n"
        f"10Y: {macro.get('ten_year_yield', '—')}%\n"
        f"HY Spread: {macro.get('hy_spread_bps', '—')} bps\n"
        f"Stress: {macro.get('macro_stress_score', '—')}"
    )

@router.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    conn = get_conn(settings.db_path)
    migrate_alerts(conn)
    if req.period == "daily":
        alerts = evaluate_alerts(conn)
        as_of, html = build_daily_report_html(conn)
        result = {"as_of": as_of, "alerts": alerts, "digest_html": html}
        if req.send and getattr(settings, "telegram_bot_token", None) and getattr(settings, "telegram_chat_id", None):
            tg = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)
            if alerts:
                await send_alerts(conn, alerts, tg)
            open_warn = list_open_alerts(conn, min_severity=5, max_severity=7)
            if html:
                await send_digest(conn, open_warn, html, tg, severity_hint=5)
        return result
    elif req.period in {"weekly", "monthly", "quarterly"}:
        as_of, html = build_period_report_html(conn, req.period)
        if req.send and getattr(settings, "telegram_bot_token", None) and getattr(settings, "telegram_chat_id", None) and html:
            tg = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)
            open_info = list_open_alerts(conn, min_severity=1, max_severity=4)
            await send_digest(conn, open_info, html, tg, severity_hint=3)
        return {"as_of": as_of, "digest_html": html}
    else:
        raise HTTPException(400, "invalid period")

@router.get("/open")
def list_open():
    conn = get_conn(settings.db_path)
    migrate_alerts(conn)
    return list_open_alerts(conn)

@router.get("/active")
def list_active():
    return list_open()

@router.post("/ack/{alert_id}")
def ack(alert_id: str):
    conn = get_conn(settings.db_path)
    migrate_alerts(conn)
    ok = ack_alert(conn, alert_id, who="telegram")
    if not ok:
        raise HTTPException(404, "not found or already acked")
    return {"status": "ok"}

@router.post("/{alert_id}/acknowledge")
def acknowledge(alert_id: str):
    return ack(alert_id)

@router.get("/reports/daily")
def report_daily():
    conn = get_conn(settings.db_path)
    migrate_alerts(conn)
    as_of, html = build_daily_report_html(conn)
    return {"as_of": as_of, "digest_html": html}

@router.post("/telegram/webhook")
async def telegram_webhook(update: dict):
    if not _telegram_ready():
        raise HTTPException(400, "telegram_not_configured")
    msg = update.get("message") or update.get("edited_message") or {}
    text = (msg.get("text") or "").strip()
    chat_id = (msg.get("chat") or {}).get("id")
    if not text or not chat_id:
        return {"ok": True, "ignored": True}
    if not _chat_allowed(chat_id):
        return {"ok": True, "ignored": True}

    parts = text.split()
    cmd = parts[0].split("@")[0].lstrip("/").lower()
    args = parts[1:]

    conn = get_conn(settings.db_path)
    migrate_alerts(conn)

    if cmd in {"start", "help"}:
        reply = _help_text()
    elif cmd in {"alerts", "active"}:
        reply = _format_alerts_html(list_open_alerts(conn))
    elif cmd == "ack":
        if not args:
            reply = "Usage: /ack <id>"
        else:
            ok = ack_alert(conn, args[0], who="telegram")
            reply = "✅ Acknowledged." if ok else "⚠️ Not found or already acknowledged."
    elif cmd == "ackall":
        open_alerts = list_open_alerts(conn)
        count = 0
        for a in open_alerts:
            if ack_alert(conn, a["id"], who="telegram"):
                count += 1
        reply = f"✅ Acknowledged {count} alert(s)."
    elif cmd == "status":
        _, snap = _latest_snapshot(conn)
        reply = _status_text(snap) if snap else "No daily snapshot available."
    elif cmd == "full":
        _, html = build_daily_report_html(conn)
        reply = html or "No daily snapshot available."
    elif cmd == "income":
        _, snap = _latest_snapshot(conn)
        reply = _income_text(snap) if snap else "No daily snapshot available."
    elif cmd == "mtd":
        _, snap = _latest_snapshot(conn)
        reply = _mtd_text(snap) if snap else "No daily snapshot available."
    elif cmd == "received":
        _, snap = _latest_snapshot(conn)
        reply = _received_text(snap) if snap else "No daily snapshot available."
    elif cmd == "perf":
        _, snap = _latest_snapshot(conn)
        reply = _perf_text(snap) if snap else "No daily snapshot available."
    elif cmd == "holdings":
        _, snap = _latest_snapshot(conn)
        reply = _holdings_text(snap) if snap else "No daily snapshot available."
    elif cmd == "risk":
        _, snap = _latest_snapshot(conn)
        reply = _risk_text(snap) if snap else "No daily snapshot available."
    elif cmd == "goal":
        _, snap = _latest_snapshot(conn)
        reply = _goal_text(snap) if snap else "No daily snapshot available."
    elif cmd == "projection":
        _, snap = _latest_snapshot(conn)
        reply = _projection_text(snap) if snap else "No daily snapshot available."
    elif cmd == "settings":
        reply = _settings_text(conn)
    elif cmd == "silence":
        hours = 24
        if args and args[0].isdigit():
            hours = max(1, min(int(args[0]), 168))
        set_silence(conn, hours)
        reply = f"🔕 Alerts silenced for {hours} hours."
    elif cmd == "resume":
        clear_silence(conn)
        reply = "🔔 Alerts resumed."
    elif cmd == "threshold":
        if not args or not args[0].isdigit():
            reply = "Usage: /threshold <1-10>"
        else:
            sev = max(1, min(int(args[0]), 10))
            set_min_severity(conn, sev)
            reply = f"✅ Minimum severity set to {sev}."
    elif cmd == "snapshot":
        _, snap = _latest_snapshot(conn)
        reply = _snapshot_text(snap) if snap else "No daily snapshot available."
    elif cmd == "compare":
        as_of, snap = _latest_snapshot(conn)
        if not as_of or not snap:
            reply = "No daily snapshot available."
        else:
            prev_date = _prev_snapshot_date(conn, as_of)
            if not prev_date:
                reply = "No prior snapshot available."
            else:
                try:
                    diff = diff_daily_from_db(conn, prev_date, as_of)
                    nlv = ((diff.get("portfolio_metrics") or {}).get("totals") or {}).get("net_liquidation_value") or {}
                    left = nlv.get("left")
                    right = nlv.get("right")
                    delta = nlv.get("delta")
                    delta_pct = None
                    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and left:
                        delta_pct = (right / left - 1.0) * 100.0
                    reply = (
                        "<b>Daily Compare</b>\n"
                        f"{prev_date} → {as_of}\n"
                        f"Net Δ: {_fmt_money(delta)} ({_fmt_pct(delta_pct,2)})"
                    )
                except Exception:
                    reply = "Unable to compute daily comparison."
    elif cmd == "macro":
        _, snap = _latest_snapshot(conn)
        reply = _macro_text(snap) if snap else "No daily snapshot available."
    elif cmd == "about":
        _, snap = _latest_snapshot(conn)
        meta = snap.get("meta") if snap else {}
        reply = f"<b>Alert Bot</b>\nSchema: {meta.get('schema_version', '—')}\nAs of: {snap.get('as_of_date_local', '—') if snap else '—'}"
    else:
        reply = _help_text()

    tg = TelegramClient(settings.telegram_bot_token, chat_id)
    await tg.send_message_html(reply)
    return {"ok": True}

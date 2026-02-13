from __future__ import annotations
import json
import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import settings
from ..db import get_conn
from ..pipeline.snapshot_views import assemble_daily_snapshot, assemble_period_snapshot_target
from ..pipeline.diff_daily import (
    _totals,
    _income,
    _dividends,
    _dividends_upcoming,
    _perf,
    _risk_flat,
    _rollups,
    _goal_tiers,
    _goal_pace,
    _holdings_flat,
    _macro_snapshot,
)
from ..alerts.evaluator import (
    evaluate_alerts,
    build_daily_report_html,
    build_period_report_html,
    build_morning_brief_html,
    build_evening_recap_html,
    DIGEST_SECTIONS,
    _get_digest_sections,
    _holding_ultimate,
)
from ..alerts.storage import (
    migrate_alerts,
    list_open_alerts,
    ack_alert,
    get_setting,
    set_setting,
    get_alert_by_id,
)
from ..alerts.notifier import (
    send_alerts,
    send_digest,
    set_silence,
    clear_silence,
    set_min_severity,
)
from ..pipeline.diff_daily import diff_daily_from_db
from ..services.telegram import (
    TelegramClient,
    format_goal_tiers_html,
    build_inline_keyboard,
    format_period_holdings_html,
    format_period_trades_html,
    format_period_activity_html,
    format_period_risk_html,
    build_period_insight_keyboard,
)

router = APIRouter()

# NLP pattern matching for natural language queries
_NLP_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"how much dividend|dividends?\s+(this|for)\s+month", re.I), "mtd"),
    (re.compile(r"next payment|when\s+is\s+.*payment|upcoming\s+dividend", re.I), "income"),
    (re.compile(r"am i on track|on pace|pace\s+check", re.I), "pace"),
    (re.compile(r"how am i doing|performance|how.?s my portfolio", re.I), "perf"),
    (re.compile(r"portfolio|summary|status|overview", re.I), "status"),
    (re.compile(r"what if|whatif|scenario", re.I), "whatif"),
    (re.compile(r"simulat", re.I), "simulate"),
    (re.compile(r"\brisk\b|volatility|vol\b", re.I), "risk"),
    (re.compile(r"\bgoals?\b|target", re.I), "goal"),
    (re.compile(r"rebalanc|optimiz", re.I), "rebalance"),
    (re.compile(r"\btrend\b|alert\s+history|frequency", re.I), "trend"),
]

def _nlp_match(text: str) -> str | None:
    """Return matched command string or None."""
    for pattern, cmd in _NLP_PATTERNS:
        if pattern.search(text):
            return cmd
    return None

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


def _build_alert_buttons(alert_id: str) -> dict:
    """Build inline keyboard buttons for an alert."""
    return build_inline_keyboard([
        [
            {"text": "✓ Acknowledge", "callback_data": f"ack:{alert_id[:32]}"},
            {"text": "📊 Details", "callback_data": f"details:{alert_id[:32]}"},
        ],
        [
            {"text": "🔇 Silence 1h", "callback_data": "silence:1"},
            {"text": "🔇 Silence 24h", "callback_data": "silence:24"},
        ],
    ])

def _help_text() -> str:
    return (
        "<b>Alert Bot Commands</b>\n\n"
        "<b>📊 Portfolio</b>\n"
        "/status - quick daily summary\n"
        "/holdings - top holdings by weight\n"
        "/position &lt;symbol&gt; - detailed position info\n"
        "/perf - performance summary\n"
        "/risk - risk metrics\n\n"
        "<b>💰 Income</b>\n"
        "/income - upcoming dividends\n"
        "/mtd - month-to-date income\n"
        "/received - last 30d dividend breakdown\n\n"
        "<b>🎯 Goals</b>\n"
        "/goal - goal progress\n"
        "/goals - dividend goal tiers (6 scenarios)\n"
        "/pace - goal pace tracking\n"
        "/projection - time to goal\n"
        "/simulate - compare all tier strategies\n"
        "/simulate &lt;1-6&gt; - detailed tier simulation\n\n"
        "<b>🔔 Alerts</b>\n"
        "/alerts - list active alerts\n"
        "/ack &lt;id&gt; - acknowledge alert\n"
        "/ackall - acknowledge all\n"
        "/silence &lt;hours&gt; - pause non-critical alerts\n"
        "/resume - resume alerts\n"
        "/threshold &lt;1-10&gt; - minimum severity\n\n"
        "<b>📈 Reports</b>\n"
        "/full - full daily digest\n"
        "/weekly - weekly period summary\n"
        "/monthly - monthly period summary\n"
        "/quarterly - quarterly period summary\n"
        "/yearly - year-to-date recap\n"
        "/compare - today vs yesterday\n"
        "/macro - macro environment\n"
        "/snapshot - latest snapshot metadata\n"
        "/health - system health check\n"
        "/settings - current bot settings\n\n"
        "<b>📉 Charts</b>\n"
        "/chart pace - pace vs expected\n"
        "/chart income - monthly income trend\n"
        "/chart performance - NLV over time\n"
        "/chart attribution - income by position\n"
        "/chart yield - yield &amp; YoC trend\n"
        "/chart risk - vol + Sharpe/Sortino\n"
        "/chart drawdown - drawdown from peak\n"
        "/chart allocation - position weights\n"
        "/chart margin - LTV &amp; margin balance\n"
        "/chart dividends - upcoming div calendar\n"
        "Add days: /chart performance 365\n\n"
        "<b>🤖 AI</b>\n"
        "/insights - AI portfolio analysis (Sonnet)\n"
        "/insights deep - deeper analysis (Opus)\n"
        "/pinsights <weekly|monthly|quarterly|yearly> - period AI insight\n"
        "/pinsights monthly deep - deeper period analysis\n\n"
        "<b>🔮 Analysis</b>\n"
        "/whatif &lt;param&gt; &lt;value&gt; - scenario planning\n"
        "/trend [category] - alert frequency trends\n"
        "/rebalance - portfolio optimization suggestions\n\n"
        "<b>📅 Scheduled</b>\n"
        "/morning - pre-market brief\n"
        "/evening - post-close recap\n"
        "/digest - customize digest sections\n\n"
        "/menu - quick actions buttons\n"
        "/help - show this help\n\n"
        "<i>Or ask in plain English, e.g. \"how much dividend this month?\"</i>"
    )

def _as_of(snap: dict | None) -> str | None:
    if not snap:
        return None
    return snap.get("as_of_date_local") or (snap.get("timestamps") or {}).get("portfolio_data_as_of_local")


def _latest_snapshot(conn):
    snap = assemble_daily_snapshot(conn, as_of_date=None)
    if not snap:
        return None, None
    as_of = _as_of(snap)
    return as_of, snap


def _prev_snapshot_date(conn, as_of_date_local: str):
    conn.row_factory = None
    row = conn.execute(
        "SELECT as_of_date_local FROM daily_portfolio WHERE as_of_date_local < ? ORDER BY as_of_date_local DESC LIMIT 1",
        (as_of_date_local,),
    ).fetchone()
    return row[0] if row else None


_PERIOD_KIND_TO_DB = {
    "weekly": "WEEK",
    "monthly": "MONTH",
    "quarterly": "QUARTER",
    "yearly": "YEAR",
}
_PERIOD_DB_TO_KIND = {v: k for k, v in _PERIOD_KIND_TO_DB.items()}


def _normalize_period_kind(kind: str | None) -> str | None:
    if not kind:
        return None
    raw = str(kind).strip().lower()
    upper = str(kind).strip().upper()
    if raw in _PERIOD_KIND_TO_DB:
        return raw
    if upper in _PERIOD_DB_TO_KIND:
        return _PERIOD_DB_TO_KIND[upper]
    aliases = {
        "week": "weekly",
        "weeks": "weekly",
        "month": "monthly",
        "months": "monthly",
        "quarter": "quarterly",
        "quarters": "quarterly",
        "year": "yearly",
        "years": "yearly",
    }
    if raw in aliases:
        return aliases[raw]
    return None


def _latest_period_target_snapshot(conn, kind: str, prefer_rolling: bool = False):
    """Load latest period snapshot in target schema for kind."""
    normalized = _normalize_period_kind(kind)
    if not normalized:
        return None
    db_period_type = _PERIOD_KIND_TO_DB[normalized]
    rolling_order = (1, 0) if prefer_rolling else (0, 1)
    for is_rolling in rolling_order:
        row = conn.execute(
            """
            SELECT period_start_date, period_end_date
            FROM period_summary
            WHERE period_type = ? AND is_rolling = ?
            ORDER BY period_end_date DESC
            LIMIT 1
            """,
            (db_period_type, is_rolling),
        ).fetchone()
        if not row:
            continue
        period_start_date, period_end_date = row[0], row[1]
        snap = assemble_period_snapshot_target(
            conn,
            db_period_type,
            period_end_date,
            period_start_date=period_start_date,
            rolling=bool(is_rolling),
        )
        if snap:
            return snap
    return None


def _target_period_snapshot_from_callback(conn, param: str):
    """
    Parse callback param `kind:start:end` and load exact period snapshot.
    Falls back to latest for kind when dates are missing.
    """
    parts = (param or "").split(":")
    kind = _normalize_period_kind(parts[0] if parts else None)
    if not kind:
        return None
    db_period_type = _PERIOD_KIND_TO_DB[kind]
    if len(parts) >= 3 and parts[1] and parts[2]:
        snap = assemble_period_snapshot_target(
            conn,
            db_period_type,
            parts[2],
            period_start_date=parts[1],
            rolling=False,
        )
        if snap:
            return snap
    return _latest_period_target_snapshot(conn, kind, prefer_rolling=True)


async def _send_period_ai_insight(tg: TelegramClient, conn, kind: str, deep: bool = False) -> bool:
    """Generate and send period AI insight with detail buttons."""
    if not settings.anthropic_api_key:
        await tg.send_message_html("ANTHROPIC_API_KEY not configured")
        return False

    normalized = _normalize_period_kind(kind)
    if not normalized:
        await tg.send_message_html("Usage: /pinsights <weekly|monthly|quarterly|yearly> [deep]")
        return False

    period_snap = _latest_period_target_snapshot(conn, normalized, prefer_rolling=True)
    if not period_snap:
        await tg.send_message_html(f"No {normalized} period summary available.")
        return False

    try:
        from ..services.ai_insights import generate_period_insight

        model = "claude-opus-4-20250514" if deep else "claude-sonnet-4-20250514"
        label = "Deep Analysis" if deep else "Quick Insight"
        await tg.send_message_html(f"🤖 Generating {normalized} {label.lower()}...")
        insight = generate_period_insight(period_snap, settings.anthropic_api_key, model=model)
        if not insight:
            await tg.send_message_html("Failed to generate period AI insight.")
            return False

        reply_markup = build_period_insight_keyboard(period_snap)
        title = f"🤖 <b>{normalized.title()} AI {label}</b>"
        await tg.send_message_html(f"{title}\n\n{insight}", reply_markup=reply_markup)
        return True
    except ImportError:
        await tg.send_message_html("AI insights not available (anthropic package not installed).")
        return False
    except Exception:
        await tg.send_message_html("Error generating period AI insight.")
        return False

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

def _fmt_ratio(val, precision: int = 2):
    try:
        return f"{float(val):.{precision}f}"
    except Exception:
        return "—"

def _status_text(snap: dict) -> str:
    totals = _totals(snap)
    income = _income(snap)
    nlv = totals.get("net_liquidation_value")
    ltv = totals.get("margin_to_portfolio_pct") or totals.get("ltv_pct")
    proj = income.get("projected_monthly_income")
    return (
        "<b>Quick Status</b>\n"
        f"Net: {_fmt_money(nlv)}\n"
        f"LTV: {_fmt_pct(ltv,1)}\n"
        f"Projected Monthly: {_fmt_money(proj)}"
    )

def _income_text(snap: dict) -> str:
    divs = _dividends_upcoming(snap)
    events = divs.get("events") or []
    if not events:
        return "No upcoming dividends in the current window."
    lines = ["<b>Upcoming Dividends</b>"]
    for ev in events[:10]:
        sym = ev.get("symbol")
        ex = ev.get("pay_date_est") or ev.get("pay_date") or ev.get("ex_date_est") or ev.get("ex_date")
        amt = ev.get("amount_est")
        lines.append(f"• {sym} {ex} ~{_fmt_money(amt)}")
    total = sum(ev.get("amount_est") or 0 for ev in events if isinstance(ev.get("amount_est"), (int, float)))
    lines.append(f"Total projected: {_fmt_money(total)}")
    return "\n".join(lines)

def _mtd_text(snap: dict) -> str:
    divs = _dividends(snap)
    proj_vs = (snap.get("dividends") or {}).get("projected_vs_received") or divs.get("projected_vs_received") or {}
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
    divs = _dividends(snap)
    window = (divs.get("windows") or {}).get("30d") or {}
    by_symbol = window.get("by_symbol") or {}
    lines = ["<b>Last 30 Days (by symbol)</b>"]
    for sym, info in sorted(by_symbol.items()):
        amt = info.get("amount")
        lines.append(f"• {sym}: {_fmt_money(amt)} ({info.get('status', 'n/a')})")
    lines.append(f"Total: {_fmt_money(window.get('total_dividends'))}")
    return "\n".join(lines)

def _perf_text(snap: dict) -> str:
    perf = _perf(snap)
    return (
        "<b>Performance</b>\n"
        f"1M: {_fmt_pct(perf.get('twr_1m_pct'),2)}\n"
        f"3M: {_fmt_pct(perf.get('twr_3m_pct'),2)}\n"
        f"6M: {_fmt_pct(perf.get('twr_6m_pct'),2)}\n"
        f"1Y: {_fmt_pct(perf.get('twr_12m_pct'),2)}"
    )

def _holdings_text(snap: dict) -> str:
    holdings = _holdings_flat(snap)
    weighted = [(h, _holding_ultimate(h).get("weight_pct") or (h.get("valuation") or {}).get("portfolio_weight_pct")) for h in holdings]
    weighted = [(h, w) for h, w in weighted if isinstance(w, (int, float))]
    weighted.sort(key=lambda x: x[1] or 0.0, reverse=True)
    lines = ["<b>Top Holdings</b>"]
    for h, w in weighted[:10]:
        mv = _holding_ultimate(h).get("market_value") or (h.get("valuation") or {}).get("market_value")
        lines.append(f"• {h.get('symbol')}: {w:.1f}% ({_fmt_money(mv)})")
    return "\n".join(lines)

def _risk_text(snap: dict) -> str:
    risk = _risk_flat(snap)
    rollups = _rollups(snap)
    stability = rollups.get("income_stability") or {}
    tail = rollups.get("tail_risk") or {}
    # Fallbacks when assembly has score/cvar in flat risk but not in nested rollups
    stability_score = stability.get("stability_score") if isinstance(stability, dict) else None
    if stability_score is None:
        stability_score = risk.get("income_stability_score")
    cvar_1d = tail.get("cvar_95_1d_pct") if isinstance(tail, dict) else None
    if cvar_1d is None:
        cvar_1d = risk.get("cvar_95_1d_pct") or risk.get("cvar_90_1d_pct")

    # Add calculation method indicator if not using realized data
    calc_method = stability.get("calculation_method") if isinstance(stability, dict) else None
    stability_label = "Income Stability"
    if calc_method == "projected":
        stability_label = "Income Stability (projected)"
    elif calc_method == "blended":
        stability_label = "Income Stability (blended)"

    return (
        "<b>Risk</b>\n"
        f"30d Vol: {_fmt_pct(risk.get('vol_30d_pct'),2)}\n"
        f"90d Vol: {_fmt_pct(risk.get('vol_90d_pct'),2)}\n"
        f"Sharpe: {_fmt_ratio(risk.get('sharpe_1y'),2)}\n"
        f"Sortino: {_fmt_ratio(risk.get('sortino_1y'),2)}\n"
        f"Sortino/Sharpe: {_fmt_ratio(risk.get('sortino_sharpe_ratio'),2)}\n"
        f"Max DD: {_fmt_pct(risk.get('max_drawdown_1y_pct'),2)}\n"
        f"{stability_label}: {_fmt_ratio(stability_score,2)}\n"
        f"CVaR 1d: {_fmt_pct(cvar_1d,1)}"
    )

def _goal_text(snap: dict) -> str:
    goal_tiers = _goal_tiers(snap)
    goal_pace = _goal_pace(snap)
    current_state = goal_tiers.get("current_state") or {}
    tiers = goal_tiers.get("tiers") or []
    likely = goal_pace.get("likely_tier") or {}
    current_pace = goal_pace.get("current_pace") or {}

    target = current_state.get("target_monthly", 0)
    current = current_state.get("projected_monthly_income", 0)
    progress = round(current / target * 100, 1) if target > 0 else 0

    lines = ["<b>🎯 Goal Progress</b>\n"]
    lines.append(f"Target: {_fmt_money(target)}/mo")
    lines.append(f"Current: {_fmt_money(current)}/mo ({progress:.0f}%)")

    bar_len = 20
    filled = int(min(progress, 100) / 100 * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    lines.append(f"[{bar}] {progress:.0f}%")

    tier_emojis = {1: "🐌", 2: "🚶", 3: "🏃", 4: "🚀", 5: "🌟", 6: "⚡"}
    tier_num = likely.get("tier", 0)
    if tier_num:
        lines.append("")
        lines.append(f"{tier_emojis.get(tier_num, '')} <b>Tier {tier_num}: {likely.get('name', '—')}</b> ({likely.get('confidence', '—')} confidence)")

    pace_cat = current_pace.get("pace_category", "unknown")
    months_delta = current_pace.get("months_ahead_behind", 0)
    pace_icons = {"ahead": "✅", "on_track": "✓", "behind": "⚠️", "off_track": "🚨"}
    if months_delta > 0:
        lines.append(f"{pace_icons.get(pace_cat, '')} Ahead of Schedule (+{months_delta:.1f} months)")
    elif months_delta < 0:
        lines.append(f"{pace_icons.get(pace_cat, '')} Behind Schedule ({months_delta:.1f} months)")
    elif pace_cat != "unknown":
        lines.append(f"{pace_icons.get(pace_cat, '')} On Track")

    revised = current_pace.get("revised_goal_date")
    if revised:
        lines.append(f"Revised ETA: {revised}")

    achievable = [t for t in tiers if t.get("months_to_goal") is not None]
    if achievable:
        lines.append("")
        lines.append("<b>Top Strategies:</b>")
        parts = []
        for t in achievable[:3]:
            tn = t.get("tier", 0)
            months = t.get("months_to_goal", 0)
            y, m = divmod(months, 12)
            time_str = f"{y}y {m}m" if y > 0 else f"{m}m"
            parts.append(f"{tier_emojis.get(tn, '')} T{tn}: {time_str}")
        lines.append(" | ".join(parts))

    return "\n".join(lines)

def _projection_text(snap: dict) -> str:
    goal_tiers = _goal_tiers(snap)
    goal_pace = _goal_pace(snap)
    tiers = goal_tiers.get("tiers") or []
    likely = goal_pace.get("likely_tier") or {}

    if not tiers:
        return "Goal tiers not available. Run a sync to generate tier data."

    tier_emojis = {1: "🐌", 2: "🚶", 3: "🏃", 4: "🚀", 5: "🌟", 6: "⚡"}
    tier_num = likely.get("tier", 0)

    lines = ["<b>📐 Goal Projections</b>\n"]
    if tier_num:
        lines.append(f"Currently on: {tier_emojis.get(tier_num, '')} Tier {tier_num} ({likely.get('name', '—')})\n")

    for t in tiers:
        tn = t.get("tier", 0)
        emoji = tier_emojis.get(tn, "📌")
        name = t.get("name", "—")
        months = t.get("months_to_goal")
        goal_date = t.get("estimated_goal_date", "—")
        assumptions = t.get("assumptions") or {}

        desc_parts = []
        if assumptions.get("monthly_contribution", 0) > 0:
            desc_parts.append(f"${assumptions['monthly_contribution']:,.0f}/mo")
        if assumptions.get("drip_enabled"):
            desc_parts.append("DRIP")
        if assumptions.get("annual_appreciation_pct", 0) > 0:
            desc_parts.append(f"{assumptions['annual_appreciation_pct']:.0f}% growth")
        if assumptions.get("ltv_maintained"):
            desc_parts.append(f"{assumptions.get('target_ltv_pct', 0):.0f}% LTV")
        if desc_parts:
            desc = " + ".join(desc_parts)
        else:
            desc = (t.get("description") or "").strip() or "Hold only; no contributions or leverage"

        if months is not None:
            y, m = divmod(months, 12)
            time_str = f"{y}y {m}m" if y > 0 else f"{m}m"
            marker = " ◀" if tn == tier_num else ""
            lines.append(f"{emoji} <b>T{tn}:</b> {time_str} ({goal_date}) — {desc}{marker}")
        else:
            lines.append(f"{emoji} <b>T{tn}:</b> ❌ Not achievable — {desc}")

    return "\n".join(lines)

def _settings_text(conn) -> str:
    min_sev = get_setting(conn, "min_severity", "1")
    silenced = get_setting(conn, "silenced_until_utc", "")
    enabled_sections = _get_digest_sections(conn)
    morning_on = "on" if getattr(settings, "alerts_morning_enabled", 1) else "off"
    evening_on = "on" if getattr(settings, "alerts_evening_enabled", 1) else "off"
    return (
        "<b>Settings</b>\n"
        f"Daily digest: {getattr(settings, 'alerts_daily_hour', 7):02d}:{getattr(settings, 'alerts_daily_minute', 30):02d}\n"
        f"Morning brief: {morning_on}\n"
        f"Evening recap: {evening_on}\n"
        f"Digest sections: {len(enabled_sections)}/{len(DIGEST_SECTIONS)}\n"
        f"Min severity: {min_sev}\n"
        f"Silenced until: {silenced or 'none'}"
    )

def _snapshot_text(snap: dict) -> str:
    meta = snap.get("meta") or {}
    as_of = _as_of(snap)
    # V5 schema: timestamps.snapshot_created_utc
    created_at = meta.get("snapshot_created_at") or meta.get("created_at") or ""
    # V5 schema: timestamps.macro_data_as_of_date
    timestamps = snap.get("timestamps") or {}
    as_of_date = timestamps.get("portfolio_data_as_of_local") or as_of
    schema_version = meta.get("schema_version") or "V4"
    return (
        "<b>Snapshot</b>\n"
        f"As of: {as_of_date or '—'}\n"
        f"Created: {created_at or '—'}\n"
        f"Schema: {schema_version}\n"
    )

def _macro_text(snap: dict) -> str:
    macro = _macro_snapshot(snap)
    return (
        "<b>Macro</b>\n"
        f"VIX: {macro.get('vix', '—')}\n"
        f"10Y: {macro.get('ten_year_yield', '—')}%\n"
        f"HY Spread: {macro.get('hy_spread_bps', '—')} bps\n"
        f"Stress: {macro.get('macro_stress_score', '—')}"
    )


def _delta_icon(delta, inverse: bool = False) -> str:
    """Return arrow icon for a delta value. inverse=True means negative is good (e.g. volatility down)."""
    if delta is None or delta == 0:
        return "➡️"
    if inverse:
        return "📉" if delta > 0 else "📈"
    return "📈" if delta > 0 else "📉"


def _delta_str(delta, fmt: str = "money") -> str:
    """Format a delta value with +/- sign."""
    if delta is None:
        return "—"
    if fmt == "money":
        return f"${delta:+,.2f}"
    elif fmt == "pct":
        return f"{delta:+.2f}%"
    elif fmt == "pct1":
        return f"{delta:+.1f}%"
    elif fmt == "ratio":
        return f"{delta:+.3f}"
    elif fmt == "bps":
        return f"{delta:+.0f} bps"
    elif fmt == "int":
        return f"{delta:+.0f}"
    return f"{delta:+.2f}"


def _compare_text(diff: dict) -> str:
    """Format an expanded daily comparison as HTML."""
    lines = ["<b>🔄 Daily Comparison</b>\n"]

    # Headline
    summary = diff.get("summary") or {}
    comp = diff.get("comparison") or {}
    left_date = comp.get("left_date", "?")
    right_date = comp.get("right_date", "?")
    direction = summary.get("direction", "neutral")
    dir_icons = {"positive": "🟢", "negative": "🔴", "neutral": "⚪", "mixed": "🟡"}
    lines.append(f"{dir_icons.get(direction, '⚪')} {left_date} → {right_date}")
    if summary.get("headline"):
        lines.append(f"<i>{summary['headline']}</i>")
    lines.append("")

    # Range return
    rng = diff.get("range_metrics") or {}
    ret_pct = rng.get("simple_return_pct")
    ret_pnl = rng.get("simple_pnl")
    if ret_pct is not None or ret_pnl is not None:
        icon = _delta_icon(ret_pnl)
        lines.append(f"{icon} <b>Day Return:</b> {_delta_str(ret_pnl)} ({_delta_str(ret_pct, 'pct')})")
        # Two different dates but zero return → flat tables may have same data for both (need separate sync runs per date)
        if left_date != right_date and (ret_pnl is None or abs(float(ret_pnl)) < 0.01) and (ret_pct is None or abs(float(ret_pct)) < 0.001):
            lines.append("<i>Identical values for both dates; run sync on each day to get deltas.</i>")
        lines.append("")

    # Portfolio totals
    pm = diff.get("portfolio_metrics") or {}
    totals = pm.get("totals") or {}
    nlv = totals.get("net_liquidation_value") or {}
    mv = totals.get("market_value") or {}
    upnl = totals.get("unrealized_pnl") or {}
    upct = totals.get("unrealized_pct") or {}
    ltv = totals.get("margin_to_portfolio_pct") or {}
    margin = totals.get("margin_loan_balance") or {}

    lines.append("<b>Portfolio:</b>")
    lines.append(f"  Net Liq: {_fmt_money(nlv.get('right'))} ({_delta_str(nlv.get('delta'))})")
    lines.append(f"  Mkt Val: {_fmt_money(mv.get('right'))} ({_delta_str(mv.get('delta'))})")
    lines.append(f"  Unreal P&L: {_fmt_money(upnl.get('right'))} ({_delta_str(upnl.get('delta'))})")
    lines.append(f"  LTV: {_fmt_pct(ltv.get('right'), 1)} ({_delta_str(ltv.get('delta'), 'pct1')})")
    lines.append(f"  Margin: {_fmt_money(margin.get('right'))} ({_delta_str(margin.get('delta'))})")
    lines.append("")

    # Income
    income = pm.get("income") or {}
    proj_mo = income.get("projected_monthly_income") or {}
    fwd_12m = income.get("forward_12m_total") or {}
    yld = income.get("portfolio_current_yield_pct") or {}
    yoc = income.get("portfolio_yield_on_cost_pct") or {}

    lines.append("<b>Income:</b>")
    lines.append(f"  Monthly: {_fmt_money(proj_mo.get('right'))} ({_delta_str(proj_mo.get('delta'))})")
    lines.append(f"  Fwd 12m: {_fmt_money(fwd_12m.get('right'))} ({_delta_str(fwd_12m.get('delta'))})")
    lines.append(f"  Yield: {_fmt_pct(yld.get('right'), 2)} ({_delta_str(yld.get('delta'), 'pct')})")
    lines.append(f"  YoC: {_fmt_pct(yoc.get('right'), 2)} ({_delta_str(yoc.get('delta'), 'pct')})")
    lines.append("")

    # Dividends
    divs = diff.get("dividends") or {}
    mtd_real = divs.get("realized_mtd_total") or {}
    next_30 = divs.get("next_30d_total") or {}
    if mtd_real or next_30:
        lines.append("<b>Dividends:</b>")
        if mtd_real:
            lines.append(f"  MTD Realized: {_fmt_money(mtd_real.get('right'))} ({_delta_str(mtd_real.get('delta'))})")
        if next_30:
            lines.append(f"  Next 30d: {_fmt_money(next_30.get('right'))} ({_delta_str(next_30.get('delta'))})")
        lines.append("")

    # Goal progress
    goal = pm.get("goal_progress") or {}
    goal_pct = goal.get("progress_pct") or {}
    goal_mo = goal.get("current_projected_monthly") or {}
    goal_months = goal.get("months_to_goal") or {}
    if goal_pct:
        lines.append("<b>Goal:</b>")
        lines.append(f"  Progress: {_fmt_pct(goal_pct.get('right'), 1)} ({_delta_str(goal_pct.get('delta'), 'pct1')})")
        lines.append(f"  Proj Monthly: {_fmt_money(goal_mo.get('right'))} ({_delta_str(goal_mo.get('delta'))})")
        lines.append(f"  Months Left: {goal_months.get('right', '—')} ({_delta_str(goal_months.get('delta'), 'int')})")
        lines.append("")

    # Performance rollups
    rollups = pm.get("rollups") or {}
    perf = rollups.get("performance") or {}
    twr_1m = perf.get("twr_1m_pct") or {}
    twr_3m = perf.get("twr_3m_pct") or {}
    if twr_1m:
        lines.append("<b>Performance:</b>")
        lines.append(f"  1M TWR: {_fmt_pct(twr_1m.get('right'), 2)} ({_delta_str(twr_1m.get('delta'), 'pct')})")
        lines.append(f"  3M TWR: {_fmt_pct(twr_3m.get('right'), 2)} ({_delta_str(twr_3m.get('delta'), 'pct')})")
        lines.append("")

    # Risk
    risk = rollups.get("risk") or {}
    vol30 = risk.get("vol_30d_pct") or {}
    sortino = risk.get("sortino_1y") or {}
    max_dd = risk.get("max_drawdown_1y_pct") or {}
    sharpe = risk.get("sharpe_1y") or {}
    if vol30:
        lines.append("<b>Risk:</b>")
        lines.append(f"  30d Vol: {_fmt_pct(vol30.get('right'), 2)} ({_delta_str(vol30.get('delta'), 'pct')})")
        lines.append(f"  Sharpe: {_fmt_ratio(sharpe.get('right'))} ({_delta_str(sharpe.get('delta'), 'ratio')})")
        lines.append(f"  Sortino: {_fmt_ratio(sortino.get('right'))} ({_delta_str(sortino.get('delta'), 'ratio')})")
        lines.append(f"  Max DD: {_fmt_pct(max_dd.get('right'), 2)} ({_delta_str(max_dd.get('delta'), 'pct')})")
        lines.append("")

    # Macro
    macro_snap = (diff.get("macro") or {}).get("snapshot") or {}
    vix = macro_snap.get("vix") or {}
    ten_y = macro_snap.get("ten_year_yield") or {}
    hy = macro_snap.get("hy_spread_bps") or {}
    stress = macro_snap.get("macro_stress_score") or {}
    if vix:
        lines.append("<b>Macro:</b>")
        lines.append(f"  VIX: {vix.get('right', '—')} ({_delta_str(vix.get('delta'), 'ratio')})")
        lines.append(f"  10Y: {ten_y.get('right', '—')}% ({_delta_str(ten_y.get('delta'), 'pct')})")
        lines.append(f"  HY Spread: {hy.get('right', '—')} bps ({_delta_str(hy.get('delta'), 'bps')})")
        lines.append(f"  Stress: {stress.get('right', '—')} ({_delta_str(stress.get('delta'), 'ratio')})")
        lines.append("")

    # Holdings changes
    holdings = diff.get("holdings") or {}
    added = holdings.get("added") or []
    removed = holdings.get("removed") or []
    changed = holdings.get("changed") or []
    if added or removed or changed:
        lines.append("<b>Holdings Changes:</b>")
        for h in added[:5]:
            sym = h.get("symbol", "?")
            fields = h.get("fields") or {}
            val = (fields.get("market_value") or {}).get("right")
            lines.append(f"  ➕ {sym} {_fmt_money(val) if val else ''}")
        for h in removed[:5]:
            sym = h.get("symbol", "?")
            fields = h.get("fields") or {}
            val = (fields.get("market_value") or {}).get("left")
            lines.append(f"  ➖ {sym} {_fmt_money(val) if val else ''}")
        for h in changed[:5]:
            sym = h.get("symbol", "?")
            ctype = h.get("change_type", "")
            impact = h.get("impact_on_income_monthly")
            detail = f" ({ctype})" if ctype else ""
            income_str = f" income {_delta_str(impact)}/mo" if impact and impact != 0 else ""
            lines.append(f"  🔀 {sym}{detail}{income_str}")
        if len(added) + len(removed) + len(changed) > 15:
            lines.append(f"  …and more")
    else:
        lines.append("No holdings changes.")

    return "\n".join(lines)


def _pace_text(snap: dict) -> str:
    """Format goal pace tracking data as HTML."""
    pace = _goal_pace(snap)
    if not pace:
        return "Goal pace tracking not available. Run a sync to generate pace data."

    lines = ["<b>🎯 Goal Pace Tracking</b>\n"]

    # Likely tier info
    likely = pace.get("likely_tier") or {}
    tier_emojis = {1: "🐌", 2: "🚶", 3: "🏃", 4: "🚀", 5: "🌟", 6: "⚡"}
    tier_num = likely.get("tier", 0)
    tier_emoji = tier_emojis.get(tier_num, "📌")

    lines.append(f"<b>Detected Strategy:</b> {tier_emoji} Tier {tier_num}: {likely.get('name', '—')}")
    if likely.get("reason"):
        lines.append(f"<i>{likely.get('reason')}</i>")
    lines.append("")

    # Current pace summary
    current = pace.get("current_pace") or {}
    pace_cat = current.get("pace_category", "unknown")
    pace_icons = {"ahead": "✅", "on_track": "✓", "behind": "⚠️", "off_track": "🚨"}
    pace_icon = pace_icons.get(pace_cat, "⚪")

    months_delta = current.get("months_ahead_behind", 0)
    if months_delta > 0:
        pace_desc = f"{abs(months_delta):.1f} months ahead"
    elif months_delta < 0:
        pace_desc = f"{abs(months_delta):.1f} months behind"
    else:
        pace_desc = "On track"

    lines.append(f"<b>Current Pace:</b> {pace_icon} {pace_desc}")
    if current.get("revised_goal_date"):
        lines.append(f"Revised Goal Date: {current.get('revised_goal_date')}")
    lines.append("")

    # Windows summary
    windows = pace.get("windows") or {}
    if windows:
        lines.append("<b>Time Windows:</b>")
        window_order = ["mtd", "30d", "60d", "90d", "qtd", "ytd", "since_inception"]
        window_labels = {
            "mtd": "MTD",
            "30d": "30D",
            "60d": "60D",
            "90d": "90D",
            "qtd": "QTD",
            "ytd": "YTD",
            "since_inception": "All Time"
        }
        for wkey in window_order:
            w = windows.get(wkey)
            if not w:
                continue
            label = window_labels.get(wkey, wkey.upper())
            actual = w.get("actual", {})
            expected = w.get("expected", {})
            delta = w.get("delta", {})

            mv_actual = actual.get("portfolio_value", 0)
            mv_expected = expected.get("portfolio_value", 0)
            mv_delta = delta.get("portfolio_value", 0)

            if mv_delta > 0:
                mv_icon = "📈"
            elif mv_delta < 0:
                mv_icon = "📉"
            else:
                mv_icon = "➡️"

            lines.append(f"  {label}: {mv_icon} ${mv_delta:+,.0f} vs expected")
        lines.append("")

    # Baseline projection
    baseline = pace.get("baseline_projection") or {}
    if baseline.get("original_months_to_goal"):
        lines.append(f"<b>Baseline:</b> {baseline.get('original_months_to_goal')} months to goal ({baseline.get('original_goal_date', '—')})")

    return "\n".join(lines)


def _health_text(conn, snap: dict) -> str:
    """Format system health check as HTML."""
    lines = ["<b>🏥 System Health</b>\n"]

    # Snapshot freshness
    meta = snap.get("meta") or {} if snap else {}
    as_of = _as_of(snap) or "—" if snap else "—"
    age_days = meta.get("snapshot_age_days", "—")
    schema = meta.get("schema_version", "—")

    if snap:
        age_status = "🟢" if age_days in (0, "—") or (isinstance(age_days, (int, float)) and age_days <= 1) else "🟡" if isinstance(age_days, (int, float)) and age_days <= 3 else "🔴"
    else:
        age_status = "🔴"

    lines.append(f"{age_status} <b>Data Freshness:</b> {as_of} (age: {age_days} days)")
    lines.append(f"⚪ <b>Schema Version:</b> {schema}")

    # Price completeness
    if snap:
        coverage = snap.get("coverage") or {}
        filled_pct = coverage.get("filled_pct", 0)
        missing = len(coverage.get("missing_paths", []))
        price_status = "🟢" if filled_pct >= 95 else "🟡" if filled_pct >= 80 else "🔴"
        lines.append(f"{price_status} <b>Price Coverage:</b> {filled_pct:.1f}% ({missing} missing)")

    # Open alerts count
    from ..alerts.storage import list_open_alerts
    open_alerts = list_open_alerts(conn)
    critical = len([a for a in open_alerts if a.get("severity", 0) >= 8])
    warnings = len([a for a in open_alerts if 5 <= a.get("severity", 0) < 8])
    info = len([a for a in open_alerts if a.get("severity", 0) < 5])

    alert_status = "🟢" if critical == 0 else "🔴"
    lines.append(f"{alert_status} <b>Open Alerts:</b> {len(open_alerts)} ({critical} critical, {warnings} warnings, {info} info)")

    # Scheduler status (check last sync)
    if snap:
        created = meta.get("snapshot_created_at", "—")
        lines.append(f"🔵 <b>Last Sync:</b> {created}")

    return "\n".join(lines)


def _position_text(snap: dict, symbol: str) -> str:
    """Format detailed position info for a single holding."""
    holdings = _holdings_flat(snap)
    symbol_upper = symbol.upper()

    holding = next((h for h in holdings if (h.get("symbol") or "").upper() == symbol_upper), None)
    if not holding:
        return f"Position <code>{symbol_upper}</code> not found in holdings."

    lines = [f"<b>📊 {symbol_upper} Position Details</b>\n"]

    # Flatten V5 structure to access nested fields
    ultimate = _holding_ultimate(holding)

    # Core position info
    shares = ultimate.get("shares") or ultimate.get("quantity") or 0
    lines.append("<b>Position:</b>")
    lines.append(f"  Shares: {shares:,.2f}")
    lines.append(f"  Price: {_fmt_money(ultimate.get('last_price'))}")
    lines.append(f"  Value: {_fmt_money(ultimate.get('market_value'))}")
    lines.append(f"  Weight: {_fmt_pct(ultimate.get('weight_pct'))}")
    lines.append("")

    # Cost basis if available
    cost_basis = ultimate.get("cost_basis")
    if cost_basis:
        avg_cost = ultimate.get("avg_cost") or ultimate.get("avg_cost_per_share")
        gain = ultimate.get("unrealized_pnl") or ultimate.get("unrealized_gain")
        gain_pct = ultimate.get("unrealized_pct") or ultimate.get("unrealized_gain_pct")
        lines.append("<b>Cost Basis:</b>")
        lines.append(f"  Avg Cost: {_fmt_money(avg_cost)}")
        lines.append(f"  Cost Basis: {_fmt_money(cost_basis)}")
        if gain is not None:
            gain_icon = "📈" if gain >= 0 else "📉"
            lines.append(f"  {gain_icon} Unrealized: {_fmt_money(gain)} ({_fmt_pct(gain_pct)})")
        lines.append("")

    # Dividend info
    div_yield = ultimate.get("current_yield_pct") or ultimate.get("dividend_yield_pct")
    annual_div = ultimate.get("forward_12m_dividend") or ultimate.get("annual_dividend")
    monthly_div = ultimate.get("projected_monthly_dividend")
    if div_yield or annual_div:
        lines.append("<b>Dividends:</b>")
        if div_yield:
            lines.append(f"  Yield: {_fmt_pct(div_yield)}")
        if annual_div:
            lines.append(f"  Annual: {_fmt_money(annual_div)}")
        if monthly_div:
            lines.append(f"  Monthly: {_fmt_money(monthly_div)}")
        elif annual_div:
            lines.append(f"  Monthly: {_fmt_money(annual_div / 12)}")
        lines.append("")

    # Risk metrics if available
    if ultimate.get("sortino_1y") or ultimate.get("vol_30d_pct"):
        lines.append("<b>Risk:</b>")
        if ultimate.get("vol_30d_pct"):
            lines.append(f"  30d Vol: {_fmt_pct(ultimate.get('vol_30d_pct'))}")
        if ultimate.get("sortino_1y"):
            lines.append(f"  Sortino: {_fmt_ratio(ultimate.get('sortino_1y'))}")
        if ultimate.get("max_drawdown_1y_pct"):
            lines.append(f"  Max DD: {_fmt_pct(ultimate.get('max_drawdown_1y_pct'))}")

    return "\n".join(lines)


def _menu_markup() -> dict:
    """Build the /menu inline keyboard."""
    return build_inline_keyboard([
        [
            {"text": "📊 Status", "callback_data": "cmd:status"},
            {"text": "💰 Income", "callback_data": "cmd:income"},
            {"text": "🎯 Goal", "callback_data": "cmd:goal"},
        ],
        [
            {"text": "📈 Perf", "callback_data": "cmd:perf"},
            {"text": "📉 Risk", "callback_data": "cmd:risk"},
            {"text": "🏃 Pace", "callback_data": "cmd:pace"},
        ],
        [
            {"text": "🔔 Alerts", "callback_data": "cmd:alerts"},
            {"text": "🏥 Health", "callback_data": "cmd:health"},
            {"text": "🌐 Macro", "callback_data": "cmd:macro"},
        ],
        [
            {"text": "💵 MTD", "callback_data": "cmd:mtd"},
            {"text": "🏦 Holdings", "callback_data": "cmd:holdings"},
            {"text": "📋 Goals", "callback_data": "cmd:goals"},
        ],
        [
            {"text": "💸 Received", "callback_data": "cmd:received"},
            {"text": "🔮 Simulate", "callback_data": "cmd:simulate"},
            {"text": "📐 Projection", "callback_data": "cmd:projection"},
        ],
        [
            {"text": "📷 Snapshot", "callback_data": "cmd:snapshot"},
            {"text": "⚙️ Settings", "callback_data": "cmd:settings"},
            {"text": "🔄 Compare", "callback_data": "cmd:compare"},
        ],
        [
            {"text": "📜 Full Digest", "callback_data": "cmd:full"},
        ],
        [
            {"text": "📈 Pace", "callback_data": "chart:pace"},
            {"text": "💰 Income", "callback_data": "chart:income"},
            {"text": "📊 NLV", "callback_data": "chart:performance"},
        ],
        [
            {"text": "📈 Yield", "callback_data": "chart:yield"},
            {"text": "⚡ Risk", "callback_data": "chart:risk"},
            {"text": "📉 Drawdown", "callback_data": "chart:drawdown"},
        ],
        [
            {"text": "🏗️ Alloc", "callback_data": "chart:allocation"},
            {"text": "🏦 Margin", "callback_data": "chart:margin"},
            {"text": "🥧 Attrib", "callback_data": "chart:attribution"},
        ],
        [
            {"text": "📅 Div Calendar", "callback_data": "chart:dividends"},
        ],
        # Period summaries
        [
            {"text": "📅 Weekly", "callback_data": "period:weekly"},
            {"text": "📅 Monthly", "callback_data": "period:monthly"},
            {"text": "📅 Quarterly", "callback_data": "period:quarterly"},
            {"text": "📅 Yearly", "callback_data": "period:yearly"},
        ],
        # 30-day chart windows
        [
            {"text": "30d NLV", "callback_data": "chart:performance:30"},
            {"text": "30d Income", "callback_data": "chart:income:30"},
            {"text": "30d Yield", "callback_data": "chart:yield:30"},
        ],
        # 1-year chart windows
        [
            {"text": "1Y NLV", "callback_data": "chart:performance:365"},
            {"text": "1Y Yield", "callback_data": "chart:yield:365"},
            {"text": "1Y Drawdown", "callback_data": "chart:drawdown:365"},
        ],
        # AI insights
        [
            {"text": "🤖 AI Insight", "callback_data": "insights:quick"},
            {"text": "🧠 Deep Analysis", "callback_data": "insights:deep"},
        ],
        [
            {"text": "🤖 Weekly AI", "callback_data": "period_insights:weekly:quick"},
            {"text": "🧠 Weekly Deep", "callback_data": "period_insights:weekly:deep"},
        ],
        # Analysis
        [
            {"text": "🔮 What-If", "callback_data": "cmd:whatif"},
            {"text": "📊 Trends", "callback_data": "cmd:trend"},
            {"text": "🔄 Rebalance", "callback_data": "cmd:rebalance"},
        ],
    ])


def _digest_keyboard_and_text(conn) -> tuple[str, dict]:
    """Build digest section toggle message and inline keyboard."""
    enabled = _get_digest_sections(conn)
    lines = ["<b>📋 Digest Sections</b>", "Tap to toggle on/off:", ""]
    rows = []
    for key, label in DIGEST_SECTIONS.items():
        on = key in enabled
        icon = "✅" if on else "❌"
        lines.append(f"{icon} {label}")
        rows.append([{"text": f"{icon} {label}", "callback_data": f"digest:{key}"}])
    text = "\n".join(lines)
    markup = build_inline_keyboard(rows)
    return text, markup


def _simulate_text(snap: dict, tier_num: int | None = None) -> str:
    """Format simulation output comparing tiers."""
    goal_tiers = _goal_tiers(snap)
    if not goal_tiers or not goal_tiers.get("tiers"):
        return "Goal tiers not available. Run a sync to generate tier data."

    tiers = goal_tiers.get("tiers", [])
    current_state = goal_tiers.get("current_state", {})
    current_income = current_state.get("projected_monthly_income", 0)
    target = current_state.get("target_monthly", 0)

    # If a specific tier was requested, show just that one in detail
    if tier_num is not None:
        tier = next((t for t in tiers if t.get("tier") == tier_num), None)
        if not tier:
            return f"Tier {tier_num} not found. Available: 1-6"

        tier_emojis = {1: "🐌", 2: "🚶", 3: "🏃", 4: "🚀", 5: "🌟", 6: "⚡"}
        emoji = tier_emojis.get(tier_num, "📌")
        assumptions = tier.get("assumptions", {})
        months = tier.get("months_to_goal")
        goal_date = tier.get("estimated_goal_date", "—")
        req_value = tier.get("required_portfolio_value", 0)
        final_value = tier.get("final_portfolio_value", 0)

        lines = [f"<b>{emoji} Simulate: Tier {tier_num} - {tier.get('name', '—')}</b>\n"]
        lines.append(f"<i>{tier.get('description', '')}</i>\n")

        lines.append("<b>Assumptions:</b>")
        if assumptions.get("monthly_contribution", 0) > 0:
            lines.append(f"  Contribution: {_fmt_money(assumptions['monthly_contribution'])}/mo")
        if assumptions.get("drip_enabled"):
            lines.append("  DRIP: Enabled")
        if assumptions.get("annual_appreciation_pct", 0) > 0:
            lines.append(f"  Growth: {assumptions['annual_appreciation_pct']:.0f}%/yr")
        if assumptions.get("ltv_maintained"):
            lines.append(f"  Leverage: {assumptions.get('target_ltv_pct', 0):.0f}% LTV maintained")
        lines.append("")

        lines.append("<b>Projection:</b>")
        if months is not None:
            years = months // 12
            rem = months % 12
            time_str = f"{years}y {rem}m" if years > 0 else f"{rem}m"
            lines.append(f"  Time to goal: {time_str}")
            lines.append(f"  Goal date: {goal_date}")
        else:
            lines.append("  Goal not achievable with these assumptions")
        if req_value:
            lines.append(f"  Required portfolio: {_fmt_money(req_value)}")
        if final_value:
            lines.append(f"  Final portfolio: {_fmt_money(final_value)}")

        return "\n".join(lines)

    # Overview: compare all tiers
    lines = ["<b>🔮 Strategy Simulation</b>\n"]
    lines.append(f"Current: {_fmt_money(current_income)}/mo → Target: {_fmt_money(target)}/mo\n")

    tier_emojis = {1: "🐌", 2: "🚶", 3: "🏃", 4: "🚀", 5: "🌟", 6: "⚡"}

    for tier in tiers:
        tn = tier.get("tier", 0)
        emoji = tier_emojis.get(tn, "📌")
        name = tier.get("name", "—")
        months = tier.get("months_to_goal")
        goal_date = tier.get("estimated_goal_date", "—")

        if months is not None:
            years = months // 12
            rem = months % 12
            time_str = f"{years}y {rem}m" if years > 0 else f"{rem}m"
            lines.append(f"{emoji} <b>T{tn}:</b> {name} → {time_str} ({goal_date})")
        else:
            lines.append(f"{emoji} <b>T{tn}:</b> {name} → ❌ Not achievable")

    lines.append("")
    lines.append("Use /simulate &lt;1-6&gt; for detailed view")

    return "\n".join(lines)


def _simple_projection(
    portfolio_value: float,
    yield_pct: float,
    monthly_contribution: float,
    growth_pct: float,
    target_monthly: float,
    max_months: int = 360,
) -> int | None:
    """Estimate months to reach target monthly income."""
    if yield_pct <= 0 or target_monthly <= 0 or portfolio_value <= 0:
        return None
    monthly_yield = yield_pct / 100 / 12
    monthly_growth = growth_pct / 100 / 12
    pv = portfolio_value
    for m in range(1, max_months + 1):
        income = pv * monthly_yield
        if income >= target_monthly:
            return m
        pv = (pv + monthly_contribution + income) * (1 + monthly_growth)
    return None


def _whatif_text(snap: dict, args: list[str]) -> str:
    """Format what-if scenario analysis."""
    goal_tiers = _goal_tiers(snap)
    current_state = goal_tiers.get("current_state") or {}

    target_monthly = current_state.get("target_monthly", 0)
    current_monthly = current_state.get("projected_monthly_income", 0)
    portfolio_value = current_state.get("total_market_value", 0) or _totals(snap).get("market_value", 0)
    yield_pct = current_state.get("portfolio_yield_pct", 0)

    if not target_monthly:
        return "Goal data not available. Set GOAL_TARGET_MONTHLY in your config."

    if not args:
        lines = [
            "<b>🔮 What-If Scenarios</b>\n",
            "Usage: /whatif &lt;param&gt; &lt;value&gt;\n",
            "<b>Parameters:</b>",
            "• /whatif contribution 5000",
            "• /whatif yield 5.0",
            "• /whatif target 3000",
            "• /whatif growth 12\n",
            "<b>Current State:</b>",
            f"• Portfolio: {_fmt_money(portfolio_value)}",
            f"• Yield: {_fmt_pct(yield_pct)}",
            f"• Monthly Income: {_fmt_money(current_monthly)}",
            f"• Target: {_fmt_money(target_monthly)}/mo",
        ]
        return "\n".join(lines)

    param_type = args[0].lower()
    try:
        value = float(args[1]) if len(args) > 1 else None
    except (ValueError, IndexError):
        return f"Invalid value. Usage: /whatif {param_type} &lt;number&gt;"
    if value is None:
        return f"Usage: /whatif {param_type} &lt;number&gt;"

    goal_pace = _goal_pace(snap)
    likely_tier = goal_pace.get("likely_tier") or {}
    tiers = goal_tiers.get("tiers") or []
    current_tier = next((t for t in tiers if t.get("tier") == likely_tier.get("tier")), {})

    goal_pace = _goal_pace(snap)
    likely_tier = goal_pace.get("likely_tier") or {}
    tiers = goal_tiers.get("tiers") or []
    current_tier = next((t for t in tiers if t.get("tier") == likely_tier.get("tier")), {})
    current_assumptions = current_tier.get("assumptions") or {}

    base_contribution = current_assumptions.get("monthly_contribution", settings.goal_monthly_contribution)
    base_growth = current_assumptions.get("annual_appreciation_pct", 8.0)
    base_yield = yield_pct if isinstance(yield_pct, (int, float)) and yield_pct > 0 else 4.0
    base_target = target_monthly

    baseline_months = _simple_projection(portfolio_value, base_yield, base_contribution, base_growth, base_target)

    if param_type == "contribution":
        scenario_months = _simple_projection(portfolio_value, base_yield, value, base_growth, base_target)
        change_label = f"Contribution: {_fmt_money(base_contribution)}/mo → {_fmt_money(value)}/mo"
    elif param_type == "yield":
        scenario_months = _simple_projection(portfolio_value, value, base_contribution, base_growth, base_target)
        change_label = f"Yield: {_fmt_pct(base_yield)} → {_fmt_pct(value)}"
    elif param_type == "target":
        scenario_months = _simple_projection(portfolio_value, base_yield, base_contribution, base_growth, value)
        change_label = f"Target: {_fmt_money(base_target)}/mo → {_fmt_money(value)}/mo"
    elif param_type == "growth":
        scenario_months = _simple_projection(portfolio_value, base_yield, base_contribution, value, base_target)
        change_label = f"Growth: {base_growth:.0f}%/yr → {value:.0f}%/yr"
    else:
        return f"Unknown parameter: {param_type}. Try: contribution, yield, target, growth"

    lines = ["<b>🔮 What-If Analysis</b>\n"]
    lines.append(f"<b>Change:</b> {change_label}\n")

    lines.append("<b>Baseline:</b>")
    if baseline_months is not None:
        by, bm = divmod(baseline_months, 12)
        lines.append(f"• Time to goal: {by}y {bm}m" if by > 0 else f"• Time to goal: {bm}m")
    else:
        lines.append("• Goal not achievable with current assumptions")

    lines.append("")
    lines.append("<b>Scenario:</b>")
    if scenario_months is not None:
        sy, sm = divmod(scenario_months, 12)
        lines.append(f"• Time to goal: {sy}y {sm}m" if sy > 0 else f"• Time to goal: {sm}m")
        if baseline_months is not None:
            diff = baseline_months - scenario_months
            if diff > 0:
                lines.append(f"• ✅ {diff} months faster!")
            elif diff < 0:
                lines.append(f"• ⚠️ {abs(diff)} months slower")
            else:
                lines.append("• ➡️ No change")
    else:
        lines.append("• Goal not achievable with these assumptions")

    return "\n".join(lines)


def _trend_text(conn, category: str | None = None) -> str:
    """Format alert trend analysis."""
    from datetime import date as _date
    from collections import defaultdict
    from ..alerts.storage import alert_trend_data

    data = alert_trend_data(conn, days=30, category=category)
    if not data:
        cat_label = f" for {category}" if category else ""
        return f"No alert data in the last 30 days{cat_label}."

    cat_label = f" — {category}" if category else ""
    lines = [f"<b>📊 Alert Trends</b> (30 days){cat_label}\n"]

    by_week: dict[str, dict] = {}
    by_category: dict[str, int] = defaultdict(int)

    for d in data:
        try:
            dt = _date.fromisoformat(d["date"])
            iso = dt.isocalendar()
            week_key = f"W{iso.week:02d}"
        except Exception:
            week_key = "W??"
        if week_key not in by_week:
            by_week[week_key] = {"count": 0, "max_sev": 0}
        by_week[week_key]["count"] += d["count"]
        by_week[week_key]["max_sev"] = max(by_week[week_key]["max_sev"], d["max_severity"])
        by_category[d["category"]] += d["count"]

    max_count = max(w["count"] for w in by_week.values()) if by_week else 1
    for week in sorted(by_week):
        info = by_week[week]
        bar_len = int(info["count"] / max_count * 10) if max_count > 0 else 0
        bar = "█" * bar_len + "░" * (10 - bar_len)
        lines.append(f"{week}: {bar} {info['count']}")

    total = sum(d["count"] for d in data)
    weeks = sorted(by_week.keys())
    if len(weeks) >= 2:
        mid = len(weeks) // 2
        first_half = sum(by_week[w]["count"] for w in weeks[:mid])
        second_half = sum(by_week[w]["count"] for w in weeks[mid:])
        if first_half > 0:
            change_pct = ((second_half - first_half) / first_half) * 100
            trend = "📈 Increasing" if change_pct > 10 else "📉 Decreasing" if change_pct < -10 else "➡️ Stable"
            lines.append(f"\n<b>Trend:</b> {trend} ({change_pct:+.0f}%)")

    lines.append(f"<b>Total:</b> {total} alerts\n")

    if not category:
        lines.append("<b>Top Categories:</b>")
        for cat, cnt in sorted(by_category.items(), key=lambda x: x[1], reverse=True)[:5]:
            lines.append(f"• {cat}: {cnt}")

    return "\n".join(lines)


def _rebalance_text(snap: dict) -> str:
    """Format portfolio rebalancing suggestions."""
    holdings = _holdings_flat(snap)
    if not holdings:
        return "No holdings data available."

    rollups = _rollups(snap)
    stability = rollups.get("income_stability") or {}
    risk = _risk_flat(snap)

    over_weight = []
    low_sortino = []
    high_yield_good_risk = []
    low_yield = []

    for h in holdings:
        sym = h.get("symbol")
        if not sym:
            continue
        weight = h.get("weight_pct", 0)
        u = _holding_ultimate(h)
        sortino = u.get("sortino_1y")
        yield_pct = h.get("income", {}).get("current_yield_pct", 0)

        if isinstance(weight, (int, float)) and weight > 10:
            over_weight.append((sym, weight))
        if isinstance(sortino, (int, float)) and sortino < 0.5:
            low_sortino.append((sym, sortino, yield_pct, weight or 0))
        if (isinstance(yield_pct, (int, float)) and yield_pct > 3.0
                and isinstance(sortino, (int, float)) and sortino > 1.0):
            high_yield_good_risk.append((sym, yield_pct, sortino))
        if isinstance(yield_pct, (int, float)) and yield_pct < 1.0 and isinstance(weight, (int, float)) and weight > 2:
            low_yield.append((sym, yield_pct, weight))

    lines = ["<b>🔄 Rebalancing Analysis</b>\n"]

    if over_weight:
        lines.append("<b>⚠️ Over-Concentrated (&gt;10%)</b>")
        for sym, w in sorted(over_weight, key=lambda x: x[1], reverse=True):
            lines.append(f"• {sym}: {w:.1f}% — consider trimming")
        lines.append("")

    if low_sortino:
        lines.append("<b>📉 Low Risk-Adjusted Return</b>")
        for sym, sort_val, yld, w in sorted(low_sortino, key=lambda x: x[1])[:5]:
            lines.append(f"• {sym}: Sortino {sort_val:.2f} | Yield {yld:.1f}% | {w:.1f}%")
        lines.append("")

    if low_yield:
        lines.append("<b>💤 Low Yield (&gt;2% weight)</b>")
        for sym, yld, w in sorted(low_yield, key=lambda x: x[1])[:5]:
            lines.append(f"• {sym}: Yield {yld:.1f}% | Weight {w:.1f}%")
        lines.append("")

    if high_yield_good_risk:
        lines.append("<b>✅ Strong Positions (yield &gt;3% + Sortino &gt;1)</b>")
        for sym, yld, sort_val in sorted(high_yield_good_risk, key=lambda x: x[1], reverse=True)[:5]:
            lines.append(f"• {sym}: Yield {yld:.1f}% | Sortino {sort_val:.2f}")
        lines.append("")

    score = stability.get("stability_score")
    calc_method = stability.get("calculation_method")
    sortino_port = risk.get("sortino_1y")
    lines.append("<b>Portfolio Metrics:</b>")
    if isinstance(score, (int, float)):
        stability_label = "Income Stability"
        if calc_method == "projected":
            stability_label = "Income Stability (projected)"
        elif calc_method == "blended":
            stability_label = "Income Stability (blended)"
        lines.append(f"• {stability_label}: {score:.2f}")
    if isinstance(sortino_port, (int, float)):
        lines.append(f"• Portfolio Sortino: {sortino_port:.2f}")

    if not over_weight and not low_sortino and not low_yield:
        lines.append("\n✅ Portfolio looks well-balanced.")

    if over_weight or low_sortino:
        lines.append("\n<b>Suggestions:</b>")
        if over_weight:
            lines.append("• Trim largest positions to improve diversification")
        if low_sortino:
            lines.append("• Review low-Sortino holdings for swap candidates")
        if high_yield_good_risk:
            lines.append("• Consider adding to strong yield+risk positions")

    return "\n".join(lines)


def _quick_summary_text(snap: dict, conn) -> str:
    """Build compact daily summary for collapsible view."""
    totals = _totals(snap)
    income = _income(snap)
    goal_tiers = _goal_tiers(snap)
    goal_pace = _goal_pace(snap)
    current_state = goal_tiers.get("current_state") or {}
    rollups = _rollups(snap)
    perf = rollups.get("performance") or {}
    risk = rollups.get("risk") or {}

    target = current_state.get("target_monthly", 0)
    current = current_state.get("projected_monthly_income", 0)
    progress = round(current / target * 100, 1) if target > 0 else 0

    pace_cat = (goal_pace.get("current_pace") or {}).get("pace_category", "")
    pace_icons = {"ahead": "✅", "on_track": "✓", "behind": "⚠️", "off_track": "🚨"}
    pace_label = pace_icons.get(pace_cat, "")

    open_crit = list_open_alerts(conn, min_severity=8)
    open_warn = list_open_alerts(conn, min_severity=5, max_severity=7)

    as_of = _as_of(snap)
    lines = [
        f"📊 <b>Daily Summary</b> | {as_of or '—'}\n",
        f"• Net: {_fmt_money(totals.get('net_liquidation_value'))}",
        f"• Monthly Income: {_fmt_money(income.get('projected_monthly_income'))}",
        f"• Goal: {progress:.0f}% {pace_label}",
        f"• LTV: {_fmt_pct(totals.get('margin_to_portfolio_pct'), 1)}",
        f"• 1M Return: {_fmt_pct(perf.get('twr_1m_pct'), 2)} | Sortino: {_fmt_ratio(risk.get('sortino_1y'), 2)}",
        f"\n🔴 {len(open_crit)} critical | 🟡 {len(open_warn)} warnings",
        "\nTap below for the full report ↓",
    ]
    return "\n".join(lines)


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


async def _handle_callback_query(callback_query: dict):
    """Handle inline keyboard button presses."""
    query_id = callback_query.get("id")
    data = callback_query.get("data", "")
    chat_id = (callback_query.get("message", {}).get("chat", {}) or {}).get("id")
    message_id = callback_query.get("message", {}).get("message_id")

    if not chat_id:
        return {"ok": True, "ignored": True}
    if not _chat_allowed(chat_id):
        return {"ok": True, "ignored": True}

    tg = TelegramClient(settings.telegram_bot_token, chat_id)

    # Answer callback immediately to prevent Telegram timeout & retries
    await tg.answer_callback_query(query_id)

    conn = get_conn(settings.db_path)
    migrate_alerts(conn)

    # Parse callback data: format is "action:param"
    parts = data.split(":", 1)
    action = parts[0] if parts else ""
    param = parts[1] if len(parts) > 1 else ""

    if action == "ack":
        ok = ack_alert(conn, param, who="telegram")
        if ok:
            await tg.edit_message_reply_markup(chat_id, message_id, None)
        else:
            await tg.send_message_html("⚠️ Already acknowledged or not found")

    elif action == "silence":
        try:
            hours = int(param) if param else 24
            hours = max(1, min(hours, 168))
            set_silence(conn, hours)
            await tg.send_message_html(f"🔕 Alerts silenced for {hours} hours")
        except ValueError:
            await tg.send_message_html("⚠️ Invalid silence duration")

    elif action == "details":
        alert = get_alert_by_id(conn, param)
        if alert:
            # Show full alert details
            title = alert.get("title", "Unknown Alert")
            body = alert.get("body_html", "No details available")
            as_of = alert.get("as_of_date_local", "")
            severity = alert.get("severity", "?")
            severity_emoji = "🔴" if severity >= 8 else "🟠" if severity >= 5 else "🟡"
            details_text = alert.get("details")
            if details_text and isinstance(details_text, dict):
                # If details is a dict, format it nicely
                import json
                details = "\n".join(f"<b>{k}:</b> {v}" for k, v in details_text.items())
            else:
                details = ""

            full_html = f"{severity_emoji} <b>{title}</b>\n\n"
            full_html += f"<b>Severity:</b> {severity}\n"
            full_html += f"<b>As of:</b> {as_of}\n\n"
            full_html += f"<b>Details:</b>\n{body}"
            if details:
                full_html += f"\n\n{details}"

            await tg.send_message_html(full_html)
        else:
            await tg.send_message_html("⚠️ Alert not found")

    elif action == "ackall":
        open_alerts = list_open_alerts(conn)
        count = 0
        for a in open_alerts:
            if ack_alert(conn, a["id"], who="telegram"):
                count += 1
        await tg.send_message_html(f"✅ Acknowledged {count} alert(s)")

    elif action == "ackcat":
        open_alerts = list_open_alerts(conn)
        count = 0
        for a in open_alerts:
            if a.get("category") == param and ack_alert(conn, a["id"], who="telegram"):
                count += 1
        await tg.send_message_html(f"✅ Acknowledged {count} {param} alert(s)")
        if message_id:
            await tg.edit_message_reply_markup(chat_id, message_id, None)

    elif action == "cmd":
        # Menu button - execute a command and send the result
        as_of, snap = _latest_snapshot(conn)
        no_snap = "No snapshot."

        def _cmd_compare():
            if not as_of or not snap:
                return no_snap
            prev_date = _prev_snapshot_date(conn, as_of)
            if not prev_date:
                return "No prior snapshot available."
            try:
                diff = diff_daily_from_db(conn, prev_date, as_of)
                return _compare_text(diff)
            except Exception:
                return "Unable to compute daily comparison."

        def _cmd_full():
            _, html = build_daily_report_html(conn)
            return html or no_snap

        cmd_map = {
            "status": lambda: _status_text(snap) if snap else no_snap,
            "income": lambda: _income_text(snap) if snap else no_snap,
            "goal": lambda: _goal_text(snap) if snap else no_snap,
            "perf": lambda: _perf_text(snap) if snap else no_snap,
            "risk": lambda: _risk_text(snap) if snap else no_snap,
            "pace": lambda: _pace_text(snap) if snap else no_snap,
            "alerts": lambda: _format_alerts_html(list_open_alerts(conn)),
            "health": lambda: _health_text(conn, snap),
            "macro": lambda: _macro_text(snap) if snap else no_snap,
            "mtd": lambda: _mtd_text(snap) if snap else no_snap,
            "holdings": lambda: _holdings_text(snap) if snap else no_snap,
            "goals": lambda: format_goal_tiers_html(_goal_tiers(snap)) if snap and _goal_tiers(snap) else "No tier data.",
            "received": lambda: _received_text(snap) if snap else no_snap,
            "simulate": lambda: _simulate_text(snap) if snap else no_snap,
            "projection": lambda: _projection_text(snap) if snap else no_snap,
            "snapshot": lambda: _snapshot_text(snap) if snap else no_snap,
            "settings": lambda: _settings_text(conn),
            "compare": _cmd_compare,
            "full": _cmd_full,
            "whatif": lambda: _whatif_text(snap, []) if snap else no_snap,
            "trend": lambda: _trend_text(conn),
            "rebalance": lambda: _rebalance_text(snap) if snap else no_snap,
        }
        handler = cmd_map.get(param)
        if handler:
            await tg.send_message_html(handler())

    elif action == "chart":
        try:
            from ..services.charts import (
                generate_pace_chart,
                generate_income_chart,
                generate_performance_chart,
                generate_attribution_chart,
                generate_yield_chart,
                generate_risk_chart,
                generate_drawdown_chart,
                generate_allocation_chart,
                generate_margin_chart,
                generate_dividend_calendar_chart,
            )
            chart_parts = param.split(":", 1)
            chart_type = chart_parts[0]
            chart_days = int(chart_parts[1]) if len(chart_parts) > 1 and chart_parts[1].isdigit() else 90

            _, snap = _latest_snapshot(conn)
            chart_gen = {
                "pace": lambda: generate_pace_chart(conn, days=chart_days),
                "income": lambda: generate_income_chart(conn, days=chart_days),
                "performance": lambda: generate_performance_chart(conn, days=chart_days),
                "attribution": lambda: generate_attribution_chart(snap) if snap else None,
                "yield": lambda: generate_yield_chart(conn, days=chart_days),
                "risk": lambda: generate_risk_chart(conn, days=chart_days),
                "drawdown": lambda: generate_drawdown_chart(conn, days=chart_days),
                "allocation": lambda: generate_allocation_chart(snap) if snap else None,
                "margin": lambda: generate_margin_chart(conn, days=chart_days),
                "dividends": lambda: generate_dividend_calendar_chart(snap) if snap else None,
            }
            gen = chart_gen.get(chart_type)
            if gen:
                img = gen()
                if img:
                    captions = {
                        "pace": "📈 Goal Pace Tracking",
                        "income": "💰 Projected Monthly Income",
                        "performance": "📊 Net Liquidation Value",
                        "attribution": "🥧 Income Attribution",
                        "yield": "📈 Portfolio Yield",
                        "risk": "⚡ Risk Dashboard",
                        "drawdown": "📉 Drawdown from Peak",
                        "allocation": "🏗️ Portfolio Allocation",
                        "margin": "🏦 Margin Utilization",
                        "dividends": "📅 Dividend Calendar",
                    }
                    days_label = f" ({chart_days}d)" if chart_days != 90 else ""
                    await tg.send_photo(img, caption=captions.get(chart_type, "Chart") + days_label)
                else:
                    await tg.send_message_html("Not enough data for chart")
            else:
                await tg.send_message_html("Unknown chart type")
        except ImportError:
            await tg.send_message_html("Charts not available (matplotlib missing)")
        except Exception:
            await tg.send_message_html("Error generating chart")

    elif action == "period":
        period_kind = _normalize_period_kind(param)
        if period_kind in {"weekly", "monthly", "quarterly", "yearly"}:
            try:
                _as_of, html = build_period_report_html(conn, period_kind)
                if html:
                    period_menu = build_inline_keyboard(
                        [
                            [
                                {"text": "🤖 AI Insight", "callback_data": f"period_insights:{period_kind}:quick"},
                                {"text": "🧠 Deep Analysis", "callback_data": f"period_insights:{period_kind}:deep"},
                            ]
                        ]
                    )
                    await tg.send_message_html(html, reply_markup=period_menu)
                else:
                    await tg.send_message_html(f"No {period_kind} report available")
            except Exception:
                await tg.send_message_html(f"Error generating {period_kind} report")

    elif action in {"period_holdings", "period_trades", "period_activity", "period_risk"}:
        period_snap = _target_period_snapshot_from_callback(conn, param)
        if not period_snap:
            await tg.send_message_html("No period snapshot found for this action.")
        else:
            formatter_map = {
                "period_holdings": format_period_holdings_html,
                "period_trades": format_period_trades_html,
                "period_activity": format_period_activity_html,
                "period_risk": format_period_risk_html,
            }
            formatter = formatter_map.get(action)
            html = formatter(period_snap) if formatter else "No period action available."
            await tg.send_message_html(html)

    elif action == "period_dismiss":
        if message_id:
            await tg.edit_message_reply_markup(chat_id, message_id, None)

    elif action == "period_insights":
        param_parts = (param or "").split(":")
        period_kind = _normalize_period_kind(param_parts[0] if param_parts else None)
        depth = (param_parts[1] if len(param_parts) > 1 else "quick").lower()
        deep = depth == "deep"
        if not period_kind:
            await tg.send_message_html("Invalid period insight request.")
        else:
            await _send_period_ai_insight(tg, conn, period_kind, deep=deep)

    elif action == "insights":
        if not settings.anthropic_api_key:
            await tg.send_message_html("ANTHROPIC_API_KEY not configured")
        else:
            try:
                from ..services.ai_insights import generate_insight
                _, snap = _latest_snapshot(conn)
                if not snap:
                    await tg.send_message_html("No snapshot available")
                else:
                    model = "claude-opus-4-20250514" if param == "deep" else "claude-sonnet-4-20250514"
                    label = "Deep Analysis" if param == "deep" else "Quick Insight"
                    insight = generate_insight(snap, settings.anthropic_api_key, model=model)
                    if insight:
                        await tg.send_message_html(f"🤖 <b>AI {label}</b>\n\n{insight}")
                    else:
                        await tg.send_message_html("Failed to generate insight")
            except Exception:
                await tg.send_message_html("Error generating AI insight")

    elif action == "digest":
        section_key = param
        if section_key in DIGEST_SECTIONS:
            enabled = _get_digest_sections(conn)
            if section_key in enabled:
                enabled.discard(section_key)
            else:
                enabled.add(section_key)
            set_setting(conn, "digest_sections", json.dumps({k: (k in enabled) for k in DIGEST_SECTIONS}))
            text, markup = _digest_keyboard_and_text(conn)
            if message_id:
                await tg.edit_message_text(chat_id, message_id, text, reply_markup=markup)

    elif action == "expand":
        if param == "full":
            _, html = build_daily_report_html(conn)
            if html:
                await tg.send_message_html(html)
            else:
                await tg.send_message_html("No report available")
        elif param in {"weekly", "monthly", "quarterly", "yearly"}:
            try:
                _, html = build_period_report_html(conn, param)
                if html:
                    await tg.send_message_html(html)
                else:
                    await tg.send_message_html(f"No {param} report available")
            except Exception:
                await tg.send_message_html(f"Error generating {param} report")

    return {"ok": True, "action": action, "param": param}

@router.post("/telegram/webhook")
async def telegram_webhook(update: dict):
    if not _telegram_ready():
        raise HTTPException(400, "telegram_not_configured")

    # Handle callback queries (inline button presses)
    callback_query = update.get("callback_query")
    if callback_query:
        return await _handle_callback_query(callback_query)

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
        _, snap = _latest_snapshot(conn)
        if not snap:
            reply = "No daily snapshot available."
        else:
            summary = _quick_summary_text(snap, conn)
            tg = TelegramClient(settings.telegram_bot_token, chat_id)
            markup = build_inline_keyboard([
                [{"text": "📖 Show Full Report", "callback_data": "expand:full"}],
            ])
            await tg.send_message_html(summary, reply_markup=markup)
            return {"ok": True}
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
    elif cmd == "goals":
        _, snap = _latest_snapshot(conn)
        if not snap:
            reply = "No daily snapshot available."
        else:
            goal_tiers = _goal_tiers(snap)
            if goal_tiers:
                reply = format_goal_tiers_html(goal_tiers)
            else:
                reply = "Goal tiers not available. Run a sync to generate tier data."
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
                    reply = _compare_text(diff)
                except Exception:
                    reply = "Unable to compute daily comparison."
    elif cmd == "macro":
        _, snap = _latest_snapshot(conn)
        reply = _macro_text(snap) if snap else "No daily snapshot available."
    elif cmd == "pace":
        _, snap = _latest_snapshot(conn)
        reply = _pace_text(snap) if snap else "No daily snapshot available."
    elif cmd == "health":
        _, snap = _latest_snapshot(conn)
        reply = _health_text(conn, snap)
    elif cmd == "position":
        if not args:
            reply = "Usage: /position &lt;symbol&gt;\nExample: /position SCHD"
        else:
            _, snap = _latest_snapshot(conn)
            if not snap:
                reply = "No daily snapshot available."
            else:
                reply = _position_text(snap, args[0])
    elif cmd == "menu":
        tg = TelegramClient(settings.telegram_bot_token, chat_id)
        await tg.send_message_html("<b>📱 Quick Menu</b>\nTap a button:", reply_markup=_menu_markup())
        return {"ok": True}
    elif cmd == "simulate":
        _, snap = _latest_snapshot(conn)
        if not snap:
            reply = "No daily snapshot available."
        elif args and args[0].isdigit():
            reply = _simulate_text(snap, int(args[0]))
        else:
            reply = _simulate_text(snap)
    elif cmd == "chart":
        chart_type = args[0].lower() if args else "pace"
        chart_days = int(args[1]) if len(args) > 1 and args[1].isdigit() else 90
        tg = TelegramClient(settings.telegram_bot_token, chat_id)
        try:
            from ..services.charts import (
                generate_pace_chart,
                generate_income_chart,
                generate_performance_chart,
                generate_attribution_chart,
                generate_yield_chart,
                generate_risk_chart,
                generate_drawdown_chart,
                generate_allocation_chart,
                generate_margin_chart,
                generate_dividend_calendar_chart,
            )
            _, snap = _latest_snapshot(conn)
            chart_map = {
                "pace": lambda: generate_pace_chart(conn, days=chart_days),
                "income": lambda: generate_income_chart(conn, days=chart_days),
                "performance": lambda: generate_performance_chart(conn, days=chart_days),
                "perf": lambda: generate_performance_chart(conn, days=chart_days),
                "attribution": lambda: generate_attribution_chart(snap) if snap else None,
                "yield": lambda: generate_yield_chart(conn, days=chart_days),
                "risk": lambda: generate_risk_chart(conn, days=chart_days),
                "drawdown": lambda: generate_drawdown_chart(conn, days=chart_days),
                "dd": lambda: generate_drawdown_chart(conn, days=chart_days),
                "allocation": lambda: generate_allocation_chart(snap) if snap else None,
                "alloc": lambda: generate_allocation_chart(snap) if snap else None,
                "margin": lambda: generate_margin_chart(conn, days=chart_days),
                "dividends": lambda: generate_dividend_calendar_chart(snap) if snap else None,
                "divs": lambda: generate_dividend_calendar_chart(snap) if snap else None,
            }
            gen = chart_map.get(chart_type)
            if not gen:
                await tg.send_message_html(
                    "Usage: /chart &lt;type&gt; [days]\n"
                    "Types: pace, income, performance, attribution,\n"
                    "yield, risk, drawdown, allocation, margin, dividends\n"
                    "Example: /chart performance 365"
                )
                return {"ok": True}
            img = gen()
            if img:
                captions = {
                    "pace": "📈 Goal Pace Tracking",
                    "income": "💰 Projected Monthly Income",
                    "performance": "📊 Net Liquidation Value",
                    "perf": "📊 Net Liquidation Value",
                    "attribution": "🥧 Income Attribution by Position",
                    "yield": "📈 Portfolio Yield",
                    "risk": "⚡ Risk Dashboard",
                    "drawdown": "📉 Drawdown from Peak",
                    "dd": "📉 Drawdown from Peak",
                    "allocation": "🏗️ Portfolio Allocation",
                    "alloc": "🏗️ Portfolio Allocation",
                    "margin": "🏦 Margin Utilization",
                    "dividends": "📅 Dividend Calendar",
                    "divs": "📅 Dividend Calendar",
                }
                days_label = f" ({chart_days}d)" if chart_days != 90 else ""
                await tg.send_photo(img, caption=captions.get(chart_type, "Chart") + days_label)
            else:
                await tg.send_message_html("Not enough data to generate chart.")
        except ImportError:
            await tg.send_message_html("Chart generation not available (matplotlib not installed).")
        except Exception:
            await tg.send_message_html("Error generating chart.")
        return {"ok": True}
    elif cmd in {"weekly", "monthly", "quarterly", "yearly"}:
        try:
            as_of, html = build_period_report_html(conn, cmd)
            reply = html or f"No {cmd} report available."
        except Exception:
            reply = f"Error generating {cmd} report."
    elif cmd == "insights":
        if not settings.anthropic_api_key:
            reply = "ANTHROPIC_API_KEY not configured. Set it in your .env file."
        else:
            try:
                from ..services.ai_insights import generate_insight
                _, snap = _latest_snapshot(conn)
                if not snap:
                    reply = "No snapshot available."
                else:
                    use_deep = args and args[0].lower() == "deep"
                    model = "claude-opus-4-20250514" if use_deep else "claude-sonnet-4-20250514"
                    label = "Deep Analysis" if use_deep else "Quick Insight"
                    tg = TelegramClient(settings.telegram_bot_token, chat_id)
                    await tg.send_message_html("🤖 Generating AI insight...")
                    insight = generate_insight(snap, settings.anthropic_api_key, model=model)
                    if insight:
                        reply = f"🤖 <b>AI {label}</b>\n\n{insight}"
                    else:
                        reply = "Failed to generate AI insight. Check logs."
            except ImportError:
                reply = "AI insights not available (anthropic package not installed)."
            except Exception:
                reply = "Error generating AI insight."
    elif cmd == "pinsights":
        if not args:
            period_kind = "weekly"
            use_deep = False
        else:
            first = args[0].lower()
            if first == "deep":
                period_kind = "weekly"
                use_deep = True
            else:
                period_kind = _normalize_period_kind(first)
                use_deep = any(arg.lower() == "deep" for arg in args[1:])
        if not period_kind:
            reply = "Usage: /pinsights <weekly|monthly|quarterly|yearly> [deep]"
        else:
            tg = TelegramClient(settings.telegram_bot_token, chat_id)
            await _send_period_ai_insight(tg, conn, period_kind, deep=use_deep)
            return {"ok": True}
    elif cmd == "morning":
        as_of, html = build_morning_brief_html(conn)
        reply = html if html else "No snapshot available."
    elif cmd == "evening":
        as_of, html = build_evening_recap_html(conn)
        reply = html if html else "No snapshot available."
    elif cmd == "digest":
        text, markup = _digest_keyboard_and_text(conn)
        tg = TelegramClient(settings.telegram_bot_token, chat_id)
        await tg.send_message_html(text, reply_markup=markup)
        return {"ok": True}
    elif cmd == "whatif":
        _, snap = _latest_snapshot(conn)
        if not snap:
            reply = "No daily snapshot available."
        else:
            reply = _whatif_text(snap, args)
    elif cmd == "trend":
        category = args[0] if args else None
        reply = _trend_text(conn, category)
    elif cmd == "rebalance":
        _, snap = _latest_snapshot(conn)
        reply = _rebalance_text(snap) if snap else "No daily snapshot available."
    elif cmd == "about":
        _, snap = _latest_snapshot(conn)
        meta = snap.get("meta") if snap else {}
        reply = f"<b>Alert Bot</b>\nSchema: {meta.get('schema_version', '—')}\nAs of: {snap.get('as_of_date_local', '—') if snap else '—'}"
    else:
        # NLP: try matching plain text to a known command
        if not text.startswith("/"):
            matched_cmd = _nlp_match(text)
            if matched_cmd:
                # Dispatch to matched command by re-routing
                _, snap = _latest_snapshot(conn)
                nlp_handlers = {
                    "mtd": lambda: _mtd_text(snap) if snap else "No snapshot available.",
                    "income": lambda: _income_text(snap) if snap else "No snapshot available.",
                    "pace": lambda: _pace_text(snap) if snap else "No snapshot available.",
                    "perf": lambda: _perf_text(snap) if snap else "No snapshot available.",
                    "status": lambda: _status_text(snap) if snap else "No snapshot available.",
                    "simulate": lambda: _simulate_text(snap) if snap else "No snapshot available.",
                    "risk": lambda: _risk_text(snap) if snap else "No snapshot available.",
                    "goal": lambda: _goal_text(snap) if snap else "No snapshot available.",
                    "whatif": lambda: _whatif_text(snap, []) if snap else "No snapshot available.",
                    "rebalance": lambda: _rebalance_text(snap) if snap else "No snapshot available.",
                    "trend": lambda: _trend_text(conn),
                }
                handler = nlp_handlers.get(matched_cmd)
                reply = handler() if handler else _help_text()
            else:
                reply = _help_text()
        else:
            reply = _help_text()

    tg = TelegramClient(settings.telegram_bot_token, chat_id)
    await tg.send_message_html(reply)
    return {"ok": True}

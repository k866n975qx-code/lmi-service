from __future__ import annotations
import json
import re
import threading
from datetime import date, timedelta
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
    format_period_pnl_html,
    format_period_goals_html,
    format_period_margin_html,
    format_period_risk_html,
    build_period_insight_keyboard,
)
from ..services.runtime_cache import TTLCacheStore, db_namespace

router = APIRouter()
_TELEGRAM_RUNTIME_CACHE = TTLCacheStore(maxsize=2048)
_TELEGRAM_VIEW_TTL_SECONDS = 86400
_TELEGRAM_CHART_TTL_SECONDS = 86400
_WARM_STATE_LOCK = threading.Lock()
_LAST_WARMED_SIGNATURE: tuple | None = None
_WARM_IN_FLIGHT = False

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
        "<b>Telegram Portfolio Bot</b>\n\n"
        "<b>Primary Navigation</b>\n"
        "/menu [portfolio|income|risk|history|positions|changes|periods|dividends|transactions|planning|system|charts|ai] - nested browse menu\n"
        "/history [metric] [days] - NLV, market, unrealized, income, yield, LTV, buffer\n"
        "/position &lt;symbol&gt; [days] - symbol hub or direct history window\n"
        "/changes [weekly|monthly|quarterly|yearly] [rolling] - change attribution\n"
        "/calendar [7|30|recent|symbol] - dividend calendar views\n"
        "/transactions [days] [all|trade|dividend|cash|margin] - transaction drilldown\n\n"
        "/pnl [mtd|30d|qtd|ytd|ltd] - realized capital gains/losses\n"
        "/cashflow [mtd|30d|qtd|ytd|ltd] - normalized portfolio cashflow\n"
        "/sales [count] - recent realized sales ledger\n\n"
        "<b>Core Snapshot Views</b>\n"
        "/status, /balance, /holdings, /allocation, /benchmark\n"
        "/income, /incomeprofile, /received, /mtd, /perf, /risk, /riskdetail, /margin, /rateshock, /pace\n"
        "/compare, /snapshot, /macro, /health, /coverage, /settings\n\n"
        "<b>Reports And Planning</b>\n"
        "/full, /weekly, /monthly, /quarterly, /yearly\n"
        "/goal, /goalnet, /goals, /pacewindows, /projection, /simulate, /whatif, /rebalance\n\n"
        "<b>Alerts</b>\n"
        "/alerts, /ack &lt;id&gt;, /ackall, /silence &lt;hours&gt;, /resume, /threshold &lt;1-10&gt;\n\n"
        "<b>Charts And AI</b>\n"
        "/chart &lt;type&gt; [days]\n"
        "/insights [deep], /pinsights &lt;period&gt; [deep]\n\n"
        "<i>You can also ask in plain English, for example \"how much dividend this month?\"</i>"
    )

def _as_of(snap: dict | None) -> str | None:
    if not snap:
        return None
    return snap.get("as_of_date_local") or (snap.get("timestamps") or {}).get("portfolio_data_as_of_local")


def _cache_key(*parts) -> tuple:
    return tuple(parts)


def _daily_signature(conn) -> tuple:
    try:
        row = conn.execute(
            "SELECT MAX(as_of_date_local), MAX(created_at_utc), COUNT(*) FROM daily_portfolio"
        ).fetchone()
    except Exception:
        row = conn.execute(
            "SELECT MAX(as_of_date_local), NULL, COUNT(*) FROM daily_portfolio"
        ).fetchone()
    row_tuple = tuple(row) if row else (None, None, 0)
    return (db_namespace(conn), *row_tuple)


def _holdings_signature(conn) -> tuple:
    row = conn.execute(
        "SELECT MAX(as_of_date_local), COUNT(*) FROM daily_holdings"
    ).fetchone()
    row_tuple = tuple(row) if row else (None, 0)
    return (db_namespace(conn), *row_tuple)


def _transactions_signature(conn) -> tuple:
    try:
        row = conn.execute(
            "SELECT MAX(date), MAX(pulled_at_utc), COUNT(*) FROM investment_transactions"
        ).fetchone()
    except Exception:
        row = conn.execute(
            "SELECT MAX(date), NULL, COUNT(*) FROM investment_transactions"
        ).fetchone()
    row_tuple = tuple(row) if row else (None, None, 0)
    return (db_namespace(conn), *row_tuple)


def _dividends_signature(conn) -> tuple:
    row = conn.execute(
        "SELECT MAX(as_of_date_local), COUNT(*) FROM daily_dividends_upcoming"
    ).fetchone()
    row_tuple = tuple(row) if row else (None, 0)
    return (db_namespace(conn), *row_tuple)


def _period_signature(conn, kind: str, rolling: bool = False) -> tuple:
    normalized = _normalize_period_kind(kind)
    if not normalized:
        return (db_namespace(conn), normalized, rolling, None, None, None)
    period_type = _PERIOD_KIND_TO_DB[normalized]
    try:
        row = conn.execute(
            """
            SELECT MAX(period_end_date), MAX(created_at_utc), COUNT(*)
            FROM period_summary
            WHERE period_type = ? AND is_rolling = ?
            """,
            (period_type, 1 if rolling else 0),
        ).fetchone()
    except Exception:
        row = conn.execute(
            """
            SELECT MAX(period_end_date), NULL, COUNT(*)
            FROM period_summary
            WHERE period_type = ? AND is_rolling = ?
            """,
            (period_type, 1 if rolling else 0),
        ).fetchone()
    row_tuple = tuple(row) if row else (None, None, 0)
    return (db_namespace(conn), normalized, rolling, *row_tuple)


def _cached_runtime_artifact(key: tuple, producer, *, ttl_seconds: float = 180):
    return _TELEGRAM_RUNTIME_CACHE.get_or_set(key, producer, ttl_seconds=ttl_seconds)


def _alerts_signature(conn) -> tuple:
    try:
        row = conn.execute(
            """
            SELECT
                SUM(CASE WHEN status='open' THEN 1 ELSE 0 END),
                MAX(updated_at_utc),
                MAX(last_triggered_at_utc),
                MAX(last_notified_at_utc),
                COUNT(*)
            FROM alert_messages
            """
        ).fetchone()
    except Exception:
        row = (0, None, None, None, 0)
    row_tuple = tuple(row) if row else (0, None, None, None, 0)
    return (db_namespace(conn), "alerts", *row_tuple)


def _settings_signature(conn) -> tuple:
    try:
        rows = conn.execute(
            "SELECT key, value FROM alert_settings ORDER BY key"
        ).fetchall()
    except Exception:
        rows = []
    return (db_namespace(conn), "settings", tuple((row[0], row[1]) for row in rows))


def _telegram_surface_signature(conn) -> tuple:
    return (
        _daily_signature(conn),
        _holdings_signature(conn),
        _transactions_signature(conn),
        _dividends_signature(conn),
        _period_signature(conn, "weekly", False),
        _period_signature(conn, "weekly", True),
        _period_signature(conn, "monthly", False),
        _period_signature(conn, "monthly", True),
        _period_signature(conn, "quarterly", False),
        _period_signature(conn, "quarterly", True),
        _period_signature(conn, "yearly", False),
        _period_signature(conn, "yearly", True),
        _alerts_signature(conn),
        _settings_signature(conn),
    )


def _cached_surface_artifact(conn, name: str, producer, *key_parts, ttl_seconds: float = _TELEGRAM_VIEW_TTL_SECONDS):
    return _cached_runtime_artifact(
        _cache_key(name, _telegram_surface_signature(conn), *key_parts),
        producer,
        ttl_seconds=ttl_seconds,
    )


def _latest_snapshot_uncached(conn):
    snap = assemble_daily_snapshot(conn, as_of_date=None)
    if not snap:
        return None, None
    as_of = _as_of(snap)
    return as_of, snap


def _latest_snapshot(conn):
    return _cached_runtime_artifact(
        _cache_key("latest_snapshot", _daily_signature(conn)),
        lambda: _latest_snapshot_uncached(conn),
        ttl_seconds=_TELEGRAM_VIEW_TTL_SECONDS,
    )


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


def _benchmark_text(snap: dict) -> str:
    perf = _perf(snap)
    vs = perf.get("vs_benchmark") or {}
    return (
        "<b>📎 Benchmark Relative</b>\n"
        f"Benchmark: {vs.get('symbol', '—')}\n"
        f"Portfolio 1Y: {_fmt_pct(perf.get('twr_12m_pct'), 2)}\n"
        f"Benchmark 1Y: {_fmt_pct(vs.get('benchmark_twr_1y_pct'), 2)}\n"
        f"Excess 1Y: {_fmt_pct(vs.get('excess_1y_pct'), 2)}\n"
        f"Correlation: {_fmt_ratio(vs.get('corr_1y'), 3)}"
    )


def _allocation_text(snap: dict) -> str:
    totals = _totals(snap)
    allocation = ((snap.get("portfolio") or {}).get("allocation") or {}).get("concentration") or {}
    holdings = _holdings_flat(snap)
    weighted = []
    for holding in holdings:
        ultimate = _holding_ultimate(holding)
        weight = ultimate.get("weight_pct")
        if isinstance(weight, (int, float)):
            weighted.append((holding.get("symbol", "?"), float(weight), ultimate.get("market_value")))
    weighted.sort(key=lambda item: item[1], reverse=True)

    lines = ["<b>🏗️ Allocation & Concentration</b>\n"]
    lines.append(f"Holdings: {int(totals.get('holdings_count') or 0)}")
    lines.append(f"Profitable/Losing: {int(totals.get('positions_profitable') or 0)}/{int(totals.get('positions_losing') or 0)}")
    lines.append(f"Category: {allocation.get('category', '—')}")
    lines.append(f"Top 3 Weight: {_fmt_pct(allocation.get('top3_weight_pct'))}")
    lines.append(f"Top 5 Weight: {_fmt_pct(allocation.get('top5_weight_pct'))}")
    lines.append(f"Herfindahl: {_fmt_ratio(allocation.get('herfindahl_index'), 3)}")
    if weighted:
        lines.append("")
        lines.append("<b>Largest Positions:</b>")
        for symbol, weight, market_value in weighted[:5]:
            lines.append(f"  {symbol}: {_fmt_pct(weight)} | {_fmt_money(market_value)}")
    return "\n".join(lines)


def _margin_text(snap: dict) -> str:
    margin = snap.get("margin") or {}
    current = margin.get("current") or {}
    history = margin.get("history_90d") or {}
    stress = (margin.get("stress") or {}).get("margin_call_risk") or {}
    guidance = margin.get("guidance") or {}
    selected_mode = guidance.get("selected_mode")
    selected = next(
        (mode for mode in (guidance.get("modes") or []) if mode.get("mode") == selected_mode),
        None,
    )

    lines = ["<b>🏦 Margin & Safety</b>\n"]
    lines.append(f"Monthly Interest: {_fmt_money(current.get('monthly_interest'))}")
    lines.append(f"Annual Interest: {_fmt_money(current.get('annual_interest'))}")
    lines.append(f"Income Coverage: {_fmt_ratio(current.get('income_interest_coverage'), 2)}")
    lines.append(f"Interest / Income: {_fmt_pct(current.get('interest_to_income_pct'), 1)}")
    lines.append("")
    lines.append("<b>Call Buffer:</b>")
    lines.append(f"  Status: {stress.get('buffer_status', '—')}")
    lines.append(f"  Buffer to Call: {_fmt_pct(stress.get('buffer_to_margin_call_pct'), 1)}")
    lines.append(f"  Dollar Decline to Call: {_fmt_money(stress.get('dollar_decline_to_call'))}")
    lines.append(f"  Days at Current Vol: {stress.get('days_at_current_volatility', '—')}")
    lines.append("")
    lines.append("<b>90D History:</b>")
    lines.append(f"  Avg LTV: {_fmt_pct(history.get('ltv_avg'), 1)}")
    lines.append(f"  LTV Range: {_fmt_pct(history.get('ltv_min'), 1)} → {_fmt_pct(history.get('ltv_max'), 1)}")
    lines.append(f"  Interest Trend: {history.get('interest_expense_trend', '—')}")
    if selected:
        lines.append("")
        lines.append("<b>Guidance:</b>")
        lines.append(f"  Selected Mode: {selected.get('mode', '—')}")
        lines.append(f"  Action: {selected.get('action', '—')}")
        lines.append(f"  Amount: {_fmt_money(selected.get('amount'))}")
        lines.append(f"  Stress LTV: {_fmt_pct(selected.get('ltv_stress_pct'), 1)}")
    return "\n".join(lines)


def _coverage_text(snap: dict) -> str:
    meta = snap.get("meta") or {}
    quality = meta.get("data_quality") or {}
    missing_paths = quality.get("missing_paths") or []
    lines = ["<b>🧪 Data Coverage</b>\n"]
    lines.append(f"Filled: {_fmt_pct(quality.get('filled_pct'), 1)}")
    lines.append(f"Derived: {_fmt_pct(quality.get('derived_pct'), 1)}")
    lines.append(f"Pulled: {_fmt_pct(quality.get('pulled_pct'), 1)}")
    lines.append(f"Missing: {_fmt_pct(quality.get('missing_pct'), 1)}")
    lines.append(f"Health Status: {meta.get('health_status') or '—'}")
    lines.append(f"Schema: {meta.get('schema_version') or '—'}")
    if missing_paths:
        lines.append("")
        lines.append("<b>Missing Paths:</b>")
        for path in missing_paths[:5]:
            lines.append(f"  {path}")
    return "\n".join(lines)


def _balance_text(snap: dict) -> str:
    totals = _totals(snap)
    income = _income(snap)
    lines = ["<b>🧾 Portfolio Balance</b>\n"]
    lines.append(f"Market Value: {_fmt_money(totals.get('market_value'))}")
    lines.append(f"Cost Basis: {_fmt_money(totals.get('cost_basis'))}")
    lines.append(f"Net Liq: {_fmt_money(totals.get('net_liquidation_value'))}")
    lines.append(f"Margin Loan: {_fmt_money(totals.get('margin_loan_balance'))}")
    lines.append(f"Unrealized: {_fmt_money(totals.get('unrealized_pnl'))} ({_fmt_pct(totals.get('unrealized_pct'))})")
    lines.append(f"Projected Monthly: {_fmt_money(income.get('projected_monthly_income'))}")
    lines.append(f"Yield / YOC: {_fmt_pct(income.get('portfolio_current_yield_pct'))} / {_fmt_pct(income.get('portfolio_yield_on_cost_pct'))}")
    lines.append(f"Holdings: {int(totals.get('holdings_count') or 0)}")
    lines.append(
        f"Profitable / Losing: {int(totals.get('positions_profitable') or 0)} / {int(totals.get('positions_losing') or 0)}"
    )
    return "\n".join(lines)


def _income_profile_text(snap: dict) -> str:
    income = _income(snap)
    divs = _dividends(snap)
    growth = income.get("income_growth") or {}
    realized = divs.get("windows") or (snap.get("dividends") or {}).get("realized") or {}
    proj_vs = ((snap.get("dividends") or {}).get("projected_vs_received") or divs.get("projected_vs_received") or {})
    upcoming = ((snap.get("dividends") or {}).get("upcoming_this_month") or {}).get("events") or []

    realized_30d = (realized.get("30d") or {}).get("total_dividends")
    realized_qtd = (realized.get("qtd") or {}).get("total_dividends")
    realized_ytd = (realized.get("ytd") or {}).get("total_dividends")
    realized_mtd = (realized.get("mtd") or divs.get("realized_mtd") or {}).get("total_dividends")

    lines = ["<b>💵 Income Profile</b>\n"]
    lines.append(f"Projected Monthly: {_fmt_money(income.get('projected_monthly_income'))}")
    lines.append(f"Forward 12M: {_fmt_money(income.get('forward_12m_total'))}")
    lines.append(f"Current Yield: {_fmt_pct(income.get('portfolio_current_yield_pct'))}")
    lines.append(f"Yield On Cost: {_fmt_pct(income.get('portfolio_yield_on_cost_pct'))}")
    lines.append("")
    lines.append("<b>Realized:</b>")
    lines.append(f"  MTD: {_fmt_money(realized_mtd)}")
    lines.append(f"  30D: {_fmt_money(realized_30d)}")
    lines.append(f"  QTD: {_fmt_money(realized_qtd)}")
    lines.append(f"  YTD: {_fmt_money(realized_ytd)}")
    lines.append("")
    lines.append("<b>Projection Tracking:</b>")
    lines.append(f"  Projected This Month: {_fmt_money(proj_vs.get('projected'))}")
    lines.append(f"  Received This Month: {_fmt_money(proj_vs.get('received'))}")
    lines.append(f"  % Of Projection: {_fmt_pct(proj_vs.get('pct_of_projection'), 1)}")
    lines.append(f"  Upcoming Events This Month: {len(upcoming)}")

    if growth:
        lines.append("")
        lines.append("<b>Growth:</b>")
        lines.append(f"  MoM: {_delta_str(growth.get('mom_absolute'))} ({_fmt_pct(growth.get('mom_pct'))})")
        lines.append(f"  QoQ: {_fmt_pct(growth.get('qoq_pct'))}")
        lines.append(f"  YoY: {_fmt_pct(growth.get('yoy_pct'))}")
        lines.append(f"  12M Trend: {growth.get('trend_12m') or '—'}")
        lines.append(f"  6M CAGR: {_fmt_pct(growth.get('cagr_6m_pct'))}")

    return "\n".join(lines)


def _risk_detail_text(snap: dict) -> str:
    risk = _risk_flat(snap)
    drawdown = (((snap.get("portfolio") or {}).get("risk") or {}).get("drawdown") or {})
    tail = (((snap.get("portfolio") or {}).get("risk") or {}).get("tail_risk") or {})
    var = (((snap.get("portfolio") or {}).get("risk") or {}).get("var") or {})

    lines = ["<b>🧭 Risk Detail</b>\n"]
    lines.append(f"Risk Quality: {risk.get('portfolio_risk_quality') or '—'}")
    lines.append(f"Beta: {_fmt_ratio(risk.get('beta_portfolio'), 2)}")
    lines.append(f"Tail Category: {tail.get('tail_risk_category') or risk.get('tail_risk_category') or '—'}")
    lines.append(f"CVaR / Income: {_fmt_ratio(tail.get('cvar_to_income_ratio') or risk.get('cvar_to_income_ratio'), 2)}")
    lines.append("")
    lines.append("<b>Drawdown:</b>")
    lines.append(f"  Current Depth: {_fmt_pct(drawdown.get('current_drawdown_depth_pct'), 2)}")
    lines.append(f"  Current Duration: {drawdown.get('current_drawdown_duration_days') or '—'} days")
    lines.append(f"  Current State: {'In drawdown' if drawdown.get('currently_in_drawdown') else 'Recovered'}")
    lines.append(f"  Peak Date: {drawdown.get('peak_date') or '—'}")
    lines.append(f"  Peak Value: {_fmt_money(drawdown.get('peak_value'))}")
    lines.append(f"  Max DD 1Y: {_fmt_pct(drawdown.get('max_drawdown_1y_pct'), 2)}")
    lines.append(f"  Max DD Duration: {drawdown.get('drawdown_duration_1y_days') or '—'} days")
    lines.append("")
    lines.append("<b>Tail Loss:</b>")
    lines.append(f"  VaR 95 1D: {_fmt_pct(var.get('var_95_1d_pct') or risk.get('var_95_1d_pct'), 2)}")
    lines.append(f"  CVaR 95 1D: {_fmt_pct(var.get('cvar_95_1d_pct') or risk.get('cvar_95_1d_pct'), 2)}")
    lines.append(f"  VaR 95 1W: {_fmt_pct(var.get('var_95_1w_pct') or risk.get('var_95_1w_pct'), 2)}")
    lines.append(f"  VaR 95 1M: {_fmt_pct(var.get('var_95_1m_pct') or risk.get('var_95_1m_pct'), 2)}")
    return "\n".join(lines)


def _rate_shock_text(snap: dict) -> str:
    margin = snap.get("margin") or {}
    current = margin.get("current") or {}
    stress = (margin.get("stress") or {}).get("rate_shock_scenarios") or {}
    guidance = margin.get("guidance") or {}
    rates = guidance.get("rates") or {}

    if not stress:
        return "No margin rate shock scenarios available."

    monthly_now = current.get("monthly_interest")
    lines = ["<b>📈 Margin Rate Shock</b>\n"]
    lines.append(f"Current APR: {_fmt_pct(rates.get('apr_current_pct'), 2)}")
    lines.append(f"Current Monthly Cost: {_fmt_money(monthly_now)}")
    if rates.get("apr_future_pct"):
        lines.append(
            f"Configured Future APR: {_fmt_pct(rates.get('apr_future_pct'), 2)} on {rates.get('apr_future_date') or '—'}"
        )
    lines.append("")
    lines.append("<b>Scenarios:</b>")
    for label in sorted(stress.keys()):
        scenario = stress.get(label) or {}
        new_cost = scenario.get("new_monthly_cost")
        delta = None
        if isinstance(new_cost, (int, float)) and isinstance(monthly_now, (int, float)):
            delta = float(new_cost) - float(monthly_now)
        lines.append(
            f"  {label}: {_fmt_pct(scenario.get('new_rate_pct'), 2)} | "
            f"{_fmt_money(new_cost)} ({_delta_str(delta)})"
        )
        lines.append(
            f"    Coverage: {_fmt_ratio(scenario.get('income_coverage_ratio'), 2)} | "
            f"Interest / Income: {_fmt_pct(scenario.get('margin_impact_pct'), 2)}"
        )
    return "\n".join(lines)


def _goal_net_text(snap: dict) -> str:
    goals = snap.get("goals") or {}
    baseline = goals.get("baseline") or {}
    net = goals.get("net_of_interest") or {}
    target = goals.get("current_state", {}).get("target_monthly")
    gross = baseline.get("current_projected_monthly")
    net_income = net.get("current_projected_monthly_net")
    drag = None
    if isinstance(gross, (int, float)) and isinstance(net_income, (int, float)):
        drag = float(gross) - float(net_income)

    lines = ["<b>🧮 Net Goal Progress</b>\n"]
    lines.append(f"Target: {_fmt_money(target)}/mo")
    lines.append(f"Gross Projected: {_fmt_money(gross)}/mo")
    lines.append(f"Net Of Interest: {_fmt_money(net_income)}/mo")
    lines.append(f"Interest Drag: {_fmt_money(drag)}/mo")
    lines.append("")
    lines.append("<b>Progress:</b>")
    lines.append(f"  Gross: {_fmt_pct(baseline.get('progress_pct'), 2)} | {baseline.get('months_to_goal') or '—'} months")
    lines.append(f"  Net: {_fmt_pct(net.get('progress_pct'), 2)} | {net.get('months_to_goal') or '—'} months")
    lines.append(f"  Gross Goal Date: {baseline.get('estimated_goal_date') or '—'}")
    return "\n".join(lines)


def _pace_windows_text(snap: dict) -> str:
    pace = _goal_pace(snap) or {}
    windows = pace.get("windows") or {}
    if not windows:
        return "No goal pace window data available."

    order = ["mtd", "30d", "60d", "90d", "qtd", "ytd", "since_inception"]
    labels = {
        "mtd": "MTD",
        "30d": "30D",
        "60d": "60D",
        "90d": "90D",
        "qtd": "QTD",
        "ytd": "YTD",
        "since_inception": "All Time",
    }
    lines = ["<b>⏱ Pace Windows</b>\n"]
    for key in order:
        window = windows.get(key)
        if not isinstance(window, dict):
            continue
        delta = window.get("delta") or {}
        pace_block = window.get("pace") or {}
        lines.append(f"<b>{labels.get(key, key.upper())}</b>")
        lines.append(
            f"  Portfolio vs Expected: {_delta_str(delta.get('portfolio_value'))} "
            f"({_fmt_pct(delta.get('portfolio_value_pct'), 1)})"
        )
        lines.append(
            f"  Income vs Expected: {_delta_str(delta.get('monthly_income'))}/mo "
            f"({_fmt_pct(delta.get('monthly_income_pct'), 1)})"
        )
        lines.append(
            f"  Pace: {_fmt_pct(pace_block.get('pct_of_tier_pace'), 1)} | "
            f"Months Ahead/Behind: {pace_block.get('months_ahead_behind') or '—'}"
        )
        lines.append(
            f"  Window: {window.get('start_date') or '—'} → {window.get('end_date') or '—'}"
        )
        lines.append("")
    return "\n".join(lines).rstrip()

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
    months_delta_raw = current_pace.get("months_ahead_behind")
    months_delta = float(months_delta_raw) if isinstance(months_delta_raw, (int, float)) else 0.0
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

    months_delta_raw = current.get("months_ahead_behind")
    months_delta = float(months_delta_raw) if isinstance(months_delta_raw, (int, float)) else 0.0
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
            mv_delta_raw = delta.get("portfolio_value")
            mv_delta = float(mv_delta_raw) if isinstance(mv_delta_raw, (int, float)) else 0.0

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


def _position_hub_text(conn, snap: dict | None, symbol: str) -> str:
    symbol_upper = (symbol or "").upper()
    lines = [f"<b>🧭 {symbol_upper} Position Hub</b>\n"]

    if snap:
        holdings = _holdings_flat(snap)
        holding = next((h for h in holdings if (h.get("symbol") or "").upper() == symbol_upper), None)
        if holding:
            ultimate = _holding_ultimate(holding)
            lines.append(f"Value: {_fmt_money(ultimate.get('market_value'))}")
            lines.append(f"Weight: {_fmt_pct(ultimate.get('weight_pct'))}")
            lines.append(f"Monthly Income: {_fmt_money(ultimate.get('projected_monthly_dividend'))}")
            lines.append(f"Yield: {_fmt_pct(ultimate.get('current_yield_pct'))}")
            lines.append(f"Unrealized: {_fmt_money(ultimate.get('unrealized_pnl'))} ({_fmt_pct(ultimate.get('unrealized_pct'))})")

            reliability = ultimate.get("reliability") or {}
            next_ex = ultimate.get("next_ex_date_est")
            if next_ex or reliability:
                lines.append("")
                lines.append("<b>Dividend Context:</b>")
                lines.append(f"  Next Ex-Date: {next_ex or '—'}")
                lines.append(f"  Reliability: {_fmt_ratio(reliability.get('consistency_score'), 2)}")
                lines.append(f"  6M Trend: {reliability.get('trend_6m') or '—'}")

    first_seen = conn.execute(
        "SELECT MIN(as_of_date_local) FROM daily_holdings WHERE symbol = ?",
        (symbol_upper,),
    ).fetchone()
    if first_seen and first_seen[0]:
        lines.append("")
        lines.append(f"First Seen In Snapshots: {first_seen[0]}")

    lines.append("")
    lines.append("<b>Path:</b>")
    lines.append("  Snapshot: current cost, P&L, yield, and risk")
    lines.append("  History: shares, value, weight, income, and activity across time")
    lines.append("  Transactions: symbol-only buys, sells, dividends, and cash flow")
    lines.append("  Dividends: next ex-date, projected events, and recent receipts")
    return "\n".join(lines)


def _position_hub_markup(symbol: str) -> dict:
    symbol_upper = symbol.upper()
    return build_inline_keyboard([
        [
            {"text": "Snapshot", "callback_data": f"pcur:{symbol_upper}"},
            {"text": "Dividends", "callback_data": f"dvs:{symbol_upper}"},
        ],
        [
            {"text": "History 30d", "callback_data": f"ph:{symbol_upper}:30"},
            {"text": "History 90d", "callback_data": f"ph:{symbol_upper}:90"},
        ],
        [
            {"text": "Tx 30d", "callback_data": f"ptx:{symbol_upper}:30"},
            {"text": "Tx 90d", "callback_data": f"ptx:{symbol_upper}:90"},
        ],
        [_nav_row("nav:psw")[0], _nav_row("nav:psw")[1]],
    ])


def _position_current_markup(symbol: str) -> dict:
    symbol_upper = symbol.upper()
    return build_inline_keyboard([
        [
            {"text": "History 30d", "callback_data": f"ph:{symbol_upper}:30"},
            {"text": "History 90d", "callback_data": f"ph:{symbol_upper}:90"},
        ],
        [
            {"text": "Tx 30d", "callback_data": f"ptx:{symbol_upper}:30"},
            {"text": "Dividends", "callback_data": f"dvs:{symbol_upper}"},
        ],
        [
            {"text": "◀ Hub", "callback_data": f"pos:{symbol_upper}"},
            {"text": "⌂ Home", "callback_data": "nav:root"},
        ],
    ])


_HISTORY_METRICS = {
    "nlv": {
        "column": "net_liquidation_value",
        "label": "Net Liquidation Value",
        "emoji": "📊",
        "style": "money",
        "chart": "performance",
        "inverse": False,
    },
    "market": {
        "column": "market_value",
        "label": "Market Value",
        "emoji": "🏦",
        "style": "money",
        "chart": "performance",
        "inverse": False,
    },
    "unrealized": {
        "column": "unrealized_pnl",
        "label": "Unrealized P&L",
        "emoji": "📈",
        "style": "money",
        "chart": "performance",
        "inverse": False,
    },
    "income": {
        "column": "projected_monthly_income",
        "label": "Projected Monthly Income",
        "emoji": "💰",
        "style": "money",
        "chart": "income",
        "inverse": False,
    },
    "yield": {
        "column": "portfolio_yield_pct",
        "label": "Portfolio Yield",
        "emoji": "📈",
        "style": "pct",
        "chart": "yield",
        "inverse": False,
    },
    "ltv": {
        "column": "ltv_pct",
        "label": "Loan To Value",
        "emoji": "🏦",
        "style": "pct",
        "chart": "margin",
        "inverse": True,
    },
    "buffer": {
        "column": "buffer_to_margin_call_pct",
        "label": "Margin Call Buffer",
        "emoji": "🛟",
        "style": "pct",
        "chart": "margin",
        "inverse": True,
    },
}


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _latest_table_date(conn, table: str, column: str) -> str | None:
    row = conn.execute(f"SELECT MAX({column}) FROM {table}").fetchone()
    return row[0] if row and row[0] else None


def _window_bounds(anchor: str | None, days_back: int) -> tuple[str | None, str | None]:
    anchor_dt = _parse_iso_date(anchor)
    if not anchor_dt:
        return None, None
    span = max(1, int(days_back or 30))
    start_dt = anchor_dt - timedelta(days=span - 1)
    return start_dt.isoformat(), anchor_dt.isoformat()


def _fmt_style_value(value, style: str) -> str:
    if style == "money":
        return _fmt_money(value)
    if style == "pct":
        return _fmt_pct(value, 2)
    return str(value) if value is not None else "—"


def _fmt_style_delta(value, style: str) -> str:
    if value is None:
        return "—"
    try:
        delta = float(value)
    except (TypeError, ValueError):
        return "—"
    if style == "money":
        return _delta_str(delta, "money")
    if style == "pct":
        return f"{delta:+.2f}pp"
    return f"{delta:+.2f}"


def _float_or_none(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _effective_tx_bucket(tx_type: str | None, name: str | None) -> str:
    raw = (tx_type or "").lower()
    text = (name or "").lower()
    if raw == "margin_repay" and "interest" in text:
        return "margin_interest"
    if raw in {"buy", "sell"}:
        return "trade"
    return raw or "other"


def _matches_tx_filter(bucket: str, tx_type: str | None, tx_filter: str) -> bool:
    if tx_filter == "all":
        return True
    if tx_filter == "trade":
        return bucket == "trade"
    if tx_filter == "dividend":
        return (tx_type or "").lower() == "dividend"
    if tx_filter == "cash":
        return (tx_type or "").lower() in {"contribution", "withdrawal"}
    if tx_filter == "margin":
        return bucket in {"margin_interest", "margin_repay", "margin_borrow"}
    return bucket == tx_filter or (tx_type or "").lower() == tx_filter


def _tx_label(tx_type: str | None, bucket: str) -> str:
    raw = (tx_type or "").lower()
    if bucket == "margin_interest":
        return "Margin Interest"
    labels = {
        "buy": "Buy",
        "sell": "Sell",
        "dividend": "Dividend",
        "interest": "Interest",
        "contribution": "Deposit",
        "withdrawal": "Withdrawal",
        "margin_borrow": "Margin Borrow",
        "margin_repay": "Margin Repay",
        "fee": "Fee",
    }
    return labels.get(raw, raw.replace("_", " ").title() or "Transaction")


_WINDOW_LABELS = {
    "mtd": "MTD",
    "30d": "30D",
    "qtd": "QTD",
    "ytd": "YTD",
    "ltd": "LTD",
}


def _normalize_window_key(value: str | None, default: str = "ytd") -> str:
    raw = (value or "").strip().lower()
    aliases = {
        "mtd": "mtd",
        "month": "mtd",
        "30": "30d",
        "30d": "30d",
        "month30": "30d",
        "qtd": "qtd",
        "quarter": "qtd",
        "ytd": "ytd",
        "year": "ytd",
        "ltd": "ltd",
        "all": "ltd",
        "alltime": "ltd",
    }
    return aliases.get(raw, default)


def _capital_gains_text(snap: dict, window_key: str = "ytd") -> str:
    normalized = _normalize_window_key(window_key, "ytd")
    gains = ((snap.get("capital_gains") or {}).get("windows") or {}).get(normalized) or {}
    recent_sales = (snap.get("capital_gains") or {}).get("recent_sales") or []
    label = _WINDOW_LABELS.get(normalized, normalized.upper())

    lines = [f"<b>📈 Realized P&amp;L - {label}</b>\n"]
    lines.append(f"As of: {_as_of(snap) or '—'}")
    lines.append(f"Gross Proceeds: {_fmt_money(gains.get('gross_proceeds'))}")
    lines.append(f"Net Proceeds: {_fmt_money(gains.get('net_proceeds'))}")
    lines.append(f"Cost Basis Sold: {_fmt_money(gains.get('realized_cost_basis'))}")
    lines.append(f"Realized P&amp;L: {_fmt_money(gains.get('realized_pnl'))}")
    lines.append(f"Realized Return: {_fmt_pct(gains.get('realized_pnl_pct'), 2)}")
    lines.append(
        f"Sales: {int(gains.get('sale_count') or 0)} "
        f"({int(gains.get('winning_sales') or 0)} wins / {int(gains.get('losing_sales') or 0)} losses)"
    )

    if recent_sales:
        lines.append("")
        lines.append("<b>Recent Sales:</b>")
        for row in recent_sales[:6]:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"  {row.get('date') or '—'} | {row.get('symbol') or '?'} | "
                f"{_fmt_money(row.get('realized_pnl'))} | {_fmt_pct(row.get('realized_pnl_pct'), 2)}"
            )

    return "\n".join(lines)


def _cashflow_text(snap: dict, window_key: str = "ytd") -> str:
    normalized = _normalize_window_key(window_key, "ytd")
    cash = ((snap.get("cashflow") or {}).get("windows") or {}).get(normalized) or {}
    label = _WINDOW_LABELS.get(normalized, normalized.upper())

    lines = [f"<b>💸 Cashflow - {label}</b>\n"]
    lines.append(f"As of: {_as_of(snap) or '—'}")
    lines.append(f"External Net: {_fmt_money(cash.get('external_net'))}")
    lines.append(f"Contributions: {_fmt_money(cash.get('contributions_total'))}")
    lines.append(f"Withdrawals: {_fmt_money(cash.get('withdrawals_total'))}")
    lines.append("")
    lines.append("<b>Trading:</b>")
    lines.append(f"  Buy Spend: {_fmt_money(cash.get('buys_total'))}")
    lines.append(f"  Sell Proceeds: {_fmt_money(cash.get('sells_total'))}")
    lines.append(f"  Trading Net Cash: {_fmt_money(cash.get('trading_net_cash'))}")
    lines.append("")
    lines.append("<b>Income / Financing:</b>")
    lines.append(f"  Dividends: {_fmt_money(cash.get('dividends_total'))}")
    lines.append(f"  Interest Income: {_fmt_money(cash.get('interest_income_total'))}")
    lines.append(f"  Margin Interest: {_fmt_money(cash.get('margin_interest_total'))}")
    lines.append(f"  Margin Borrowed: {_fmt_money(cash.get('margin_borrowed_total'))}")
    lines.append(f"  Margin Repaid: {_fmt_money(cash.get('margin_repaid_total'))}")
    lines.append(f"  Fees: {_fmt_money(cash.get('fees_total'))}")
    lines.append("")
    lines.append(f"<b>Portfolio Cash Net:</b> {_fmt_money(cash.get('portfolio_cash_net'))}")
    lines.append(f"<b>Realized Total Return:</b> {_fmt_money(cash.get('realized_total_return'))}")
    return "\n".join(lines)


def _recent_sales_text(conn, limit: int = 10) -> str:
    rows = conn.execute(
        """
        SELECT
          date,
          symbol,
          shares_sold,
          gross_proceeds,
          realized_cost_basis,
          realized_pnl,
          realized_pnl_pct,
          weighted_holding_period_days,
          matched_complete
        FROM realized_trade_ledger
        ORDER BY date DESC, COALESCE(transaction_datetime, date) DESC, lm_transaction_id DESC
        LIMIT ?
        """,
        (max(1, min(int(limit), 25)),),
    ).fetchall()
    if not rows:
        return "No realized sales found."

    lines = ["<b>🧾 Recent Sales</b>\n"]
    for row in rows:
        tx_date, symbol, shares_sold, gross_proceeds, cost_basis, realized_pnl, realized_pnl_pct, holding_days, matched_complete = row
        status = "" if matched_complete else " (partial lot match)"
        lines.append(
            f"  {tx_date} | {symbol or '?'} | {float(shares_sold or 0.0):,.4f} sh | "
            f"{_fmt_money(realized_pnl)} | {_fmt_pct(realized_pnl_pct, 2)}{status}"
        )
        lines.append(
            f"    Proceeds {_fmt_money(gross_proceeds)} | Cost {_fmt_money(cost_basis)} | "
            f"Held {holding_days if holding_days is not None else '—'}d"
        )
    return "\n".join(lines)


def _top_symbols(conn, *, sort_column: str, limit: int = 8) -> list[dict]:
    latest = _latest_table_date(conn, "daily_holdings", "as_of_date_local")
    if not latest:
        return []
    rows = conn.execute(
        f"""
        SELECT symbol, shares, market_value, weight_pct, projected_monthly_dividend, current_yield_pct
        FROM daily_holdings
        WHERE as_of_date_local = ?
        ORDER BY {sort_column} DESC, market_value DESC
        LIMIT ?
        """,
        (latest, limit),
    ).fetchall()
    return [
        {
            "symbol": row[0],
            "shares": row[1],
            "market_value": row[2],
            "weight_pct": row[3],
            "projected_monthly_dividend": row[4],
            "current_yield_pct": row[5],
        }
        for row in rows
        if row[0]
    ]


def _history_text(conn, metric: str, days_back: int) -> str:
    meta = _HISTORY_METRICS.get(metric)
    if not meta:
        return "Unknown history metric."

    anchor = _latest_table_date(conn, "daily_portfolio", "as_of_date_local")
    start, end = _window_bounds(anchor, days_back)
    if not start or not end:
        return "No daily history available."

    column = meta["column"]
    rows = conn.execute(
        f"""
        SELECT as_of_date_local, {column}
        FROM daily_portfolio
        WHERE as_of_date_local BETWEEN ? AND ?
          AND {column} IS NOT NULL
        ORDER BY as_of_date_local
        """,
        (start, end),
    ).fetchall()
    pairs = []
    for row in rows:
        numeric_value = _float_or_none(row[1])
        if numeric_value is None:
            continue
        pairs.append((row[0], numeric_value))
    if not pairs:
        return f"No {meta['label'].lower()} history available."

    first_date, first_val = pairs[0]
    last_date, last_val = pairs[-1]
    values = [value for _, value in pairs]
    min_date, min_val = min(pairs, key=lambda item: item[1])
    max_date, max_val = max(pairs, key=lambda item: item[1])
    avg_val = sum(values) / len(values) if values else None
    delta = last_val - first_val if len(pairs) >= 2 else 0.0
    pct_change = None
    if meta["style"] == "money" and abs(first_val) > 1e-12:
        pct_change = (delta / first_val) * 100.0

    icon = _delta_icon(delta, inverse=bool(meta.get("inverse")))
    lines = [f"<b>{meta['emoji']} {meta['label']} History</b>\n"]
    lines.append(f"Window: {first_date} → {last_date} ({len(pairs)} points)")
    lines.append(f"Current: {_fmt_style_value(last_val, meta['style'])}")
    change_line = f"{icon} Change: {_fmt_style_delta(delta, meta['style'])}"
    if pct_change is not None:
        change_line += f" ({pct_change:+.2f}%)"
    lines.append(change_line)
    lines.append(
        f"Avg: {_fmt_style_value(avg_val, meta['style'])} | "
        f"Low: {_fmt_style_value(min_val, meta['style'])} ({min_date}) | "
        f"High: {_fmt_style_value(max_val, meta['style'])} ({max_date})"
    )

    lines.append("")
    lines.append("<b>Recent:</b>")
    for point_date, point_value in pairs[-5:]:
        lines.append(f"  {point_date}: {_fmt_style_value(point_value, meta['style'])}")
    return "\n".join(lines)


def _position_history_text(conn, symbol: str, days_back: int) -> str:
    symbol_upper = (symbol or "").upper()
    anchor = _latest_table_date(conn, "daily_holdings", "as_of_date_local")
    start, end = _window_bounds(anchor, days_back)
    if not start or not end:
        return "No holdings history available."

    rows = conn.execute(
        """
        SELECT
            as_of_date_local,
            shares,
            market_value,
            weight_pct,
            projected_monthly_dividend,
            current_yield_pct,
            next_ex_date_est,
            last_ex_date,
            reliability_consistency_score,
            reliability_trend_6m
        FROM daily_holdings
        WHERE symbol = ?
          AND as_of_date_local BETWEEN ? AND ?
        ORDER BY as_of_date_local
        """,
        (symbol_upper, start, end),
    ).fetchall()
    if not rows:
        return f"No holdings history found for <code>{symbol_upper}</code>."

    first = rows[0]
    last = rows[-1]
    first_seen = conn.execute(
        "SELECT MIN(as_of_date_local) FROM daily_holdings WHERE symbol = ?",
        (symbol_upper,),
    ).fetchone()[0]

    start_shares = float(first[1] or 0.0)
    end_shares = float(last[1] or 0.0)
    start_mv = float(first[2] or 0.0)
    end_mv = float(last[2] or 0.0)
    start_weight = float(first[3] or 0.0)
    end_weight = float(last[3] or 0.0)
    start_income = float(first[4] or 0.0)
    end_income = float(last[4] or 0.0)
    start_yield = float(first[5] or 0.0)
    end_yield = float(last[5] or 0.0)

    peak_row = max(rows, key=lambda row: float(row[2] or 0.0))
    low_row = min(rows, key=lambda row: float(row[2] or 0.0))
    tx_rows = conn.execute(
        """
        SELECT date, transaction_type, quantity, amount, name
        FROM investment_transactions
        WHERE symbol = ?
          AND date BETWEEN ? AND ?
        ORDER BY date DESC
        """,
        (symbol_upper, start, end),
    ).fetchall()

    buy_count = 0
    sell_count = 0
    buy_shares = 0.0
    sell_shares = 0.0
    dividends_received = 0.0
    dividend_events = 0
    for tx_date, tx_type, quantity, amount, name in tx_rows:
        raw_type = (tx_type or "").lower()
        if raw_type == "buy":
            buy_count += 1
            buy_shares += float(quantity or 0.0)
        elif raw_type == "sell":
            sell_count += 1
            sell_shares += abs(float(quantity or 0.0))
        elif raw_type == "dividend":
            dividend_events += 1
            dividends_received += abs(float(amount or 0.0))

    lines = [f"<b>📈 {symbol_upper} Position History</b>\n"]
    lines.append(f"Window: {first[0]} → {last[0]} ({len(rows)} snapshots)")
    if first_seen:
        lines.append(f"First Seen: {first_seen}")
    lines.append("")
    lines.append("<b>Current:</b>")
    lines.append(f"  Shares: {end_shares:,.4f}")
    lines.append(f"  Value: {_fmt_money(end_mv)}")
    lines.append(f"  Weight: {_fmt_pct(end_weight)}")
    lines.append(f"  Monthly Income: {_fmt_money(end_income)}")
    lines.append(f"  Yield: {_fmt_pct(end_yield)}")
    lines.append("")
    lines.append("<b>Window Change:</b>")
    lines.append(f"  Shares: {end_shares - start_shares:+,.4f}")
    lines.append(f"  Value: {_delta_str(end_mv - start_mv)}")
    lines.append(f"  Weight: {end_weight - start_weight:+.2f}pp")
    lines.append(f"  Monthly Income: {_delta_str(end_income - start_income)}")
    lines.append(f"  Yield: {end_yield - start_yield:+.2f}pp")
    lines.append("")
    lines.append("<b>Range:</b>")
    lines.append(f"  High: {_fmt_money(peak_row[2])} ({peak_row[0]})")
    lines.append(f"  Low: {_fmt_money(low_row[2])} ({low_row[0]})")

    if buy_count or sell_count or dividend_events:
        lines.append("")
        lines.append("<b>Window Activity:</b>")
        if buy_count:
            lines.append(f"  Buys: {buy_count} ({buy_shares:,.4f} shares)")
        if sell_count:
            lines.append(f"  Sells: {sell_count} ({sell_shares:,.4f} shares)")
        if dividend_events:
            lines.append(f"  Dividends: {_fmt_money(dividends_received)} ({dividend_events} events)")

    next_ex = last[6]
    last_ex = last[7]
    consistency = last[8]
    trend_6m = last[9]
    if next_ex or last_ex or consistency is not None or trend_6m:
        lines.append("")
        lines.append("<b>Dividend Profile:</b>")
        if next_ex:
            lines.append(f"  Next Ex-Date: {next_ex}")
        if last_ex:
            lines.append(f"  Last Ex-Date: {last_ex}")
        if consistency is not None:
            lines.append(f"  Reliability: {float(consistency):.2f}")
        if trend_6m:
            lines.append(f"  6M Trend: {trend_6m}")
    return "\n".join(lines)


def _load_period_snapshot_uncached(conn, kind: str, rolling: bool = False) -> dict | None:
    normalized = _normalize_period_kind(kind)
    if not normalized:
        return None
    period_type = _PERIOD_KIND_TO_DB[normalized]
    row = conn.execute(
        """
        SELECT period_start_date, period_end_date
        FROM period_summary
        WHERE period_type = ? AND is_rolling = ?
        ORDER BY period_end_date DESC
        LIMIT 1
        """,
        (period_type, 1 if rolling else 0),
    ).fetchone()
    if not row:
        return None
    return assemble_period_snapshot_target(
        conn,
        period_type,
        row[1],
        period_start_date=row[0],
        rolling=rolling,
    )


def _load_period_snapshot(conn, kind: str, rolling: bool = False) -> dict | None:
    normalized = _normalize_period_kind(kind)
    if not normalized:
        return None
    return _cached_runtime_artifact(
        _cache_key("period_snapshot", _period_signature(conn, normalized, rolling)),
        lambda: _load_period_snapshot_uncached(conn, normalized, rolling),
        ttl_seconds=_TELEGRAM_VIEW_TTL_SECONDS,
    )


def _period_change_text(conn, kind: str, rolling: bool = False) -> str:
    snap = _load_period_snapshot(conn, kind, rolling=rolling)
    if not snap:
        mode_label = "rolling" if rolling else "final"
        return f"No {mode_label} {kind} period data available."

    meta = snap.get("meta") or {}
    period = meta.get("period") or {}
    portfolio = snap.get("portfolio") or {}
    values = portfolio.get("values") or {}
    income = portfolio.get("income") or {}
    activity = snap.get("activity") or {}
    holdings_summary = snap.get("holdings_summary") or []

    mode_label = "Rolling" if rolling else "Final"
    label = f"{mode_label} {str(period.get('type') or kind).title()} Change Attribution"
    lines = [f"<b>🔎 {label}</b>\n"]
    lines.append(
        f"Range: {period.get('start_date_local', '—')} → {period.get('end_date_local', '—')}"
    )
    lines.append("")

    delta_values = values.get("delta") or {}
    lines.append("<b>Portfolio Change:</b>")
    lines.append(f"  Market Value: {_delta_str(delta_values.get('market_value'))}")
    lines.append(f"  Net Liq: {_delta_str(delta_values.get('net_liquidation_value'))}")
    lines.append(f"  Cost Basis: {_delta_str(delta_values.get('cost_basis'))}")
    lines.append(f"  Monthly Income: {_delta_str(income.get('delta_projected_monthly'))}")
    lines.append(f"  Yield: {income.get('delta_yield_pct', 0.0):+.2f}pp")
    lines.append("")

    contrib = abs(float(((activity.get("contributions") or {}).get("total")) or 0.0))
    withdraw = abs(float(((activity.get("withdrawals") or {}).get("total")) or 0.0))
    dividends = abs(float(((activity.get("dividends") or {}).get("total_received")) or 0.0))
    margin_interest = abs(float(((activity.get("interest") or {}).get("total_paid")) or 0.0))
    trades = int(((activity.get("trades") or {}).get("total_count")) or 0)
    positions = activity.get("positions") or {}
    added = positions.get("added") or []
    removed = positions.get("removed") or []

    lines.append("<b>Flow + Activity:</b>")
    lines.append(f"  Contributions: {_fmt_money(contrib)}")
    lines.append(f"  Withdrawals: {_fmt_money(withdraw)}")
    lines.append(f"  Dividends Received: {_fmt_money(dividends)}")
    lines.append(f"  Margin Interest: {_fmt_money(margin_interest)}")
    lines.append(f"  Trades: {trades}")
    lines.append(f"  Positions Added/Removed: {len(added)}/{len(removed)}")

    if added or removed:
        if added:
            lines.append(f"  Added: {', '.join(added[:4])}")
        if removed:
            lines.append(f"  Removed: {', '.join(removed[:4])}")

    movers = [
        row for row in holdings_summary
        if isinstance(row, dict) and isinstance((row.get("values") or {}).get("market_value_delta"), (int, float))
    ]
    if movers:
        lines.append("")
        lines.append("<b>Top Value Movers:</b>")
        for row in sorted(
            movers,
            key=lambda item: abs(float((item.get("values") or {}).get("market_value_delta") or 0.0)),
            reverse=True,
        )[:4]:
            values_block = row.get("values") or {}
            lines.append(
                f"  {row.get('symbol', '?')}: {_delta_str(values_block.get('market_value_delta'))} "
                f"({values_block.get('market_value_delta_pct', 0.0):+.1f}%)"
            )

    income_movers = []
    for row in holdings_summary:
        if not isinstance(row, dict):
            continue
        income_block = row.get("income") or {}
        start_val = income_block.get("start_projected_monthly")
        end_val = income_block.get("end_projected_monthly")
        if not isinstance(start_val, (int, float)) or not isinstance(end_val, (int, float)):
            continue
        income_movers.append((row.get("symbol", "?"), float(end_val - start_val)))

    if income_movers:
        lines.append("")
        lines.append("<b>Top Income Movers:</b>")
        for sym, delta in sorted(income_movers, key=lambda item: abs(item[1]), reverse=True)[:4]:
            lines.append(f"  {sym}: {_delta_str(delta)} /mo")

    return "\n".join(lines)


def _dividend_calendar_text(conn, mode: str = "up30", symbol: str | None = None) -> str:
    latest_upcoming = _latest_table_date(conn, "daily_dividends_upcoming", "as_of_date_local")
    latest_tx = _latest_table_date(conn, "investment_transactions", "date")

    if symbol:
        symbol_upper = symbol.upper()
        lines = [f"<b>📅 {symbol_upper} Dividend View</b>\n"]
        latest_holding = conn.execute(
            """
            SELECT projected_monthly_dividend, current_yield_pct, next_ex_date_est, last_ex_date
            FROM daily_holdings
            WHERE symbol = ?
            ORDER BY as_of_date_local DESC
            LIMIT 1
            """,
            (symbol_upper,),
        ).fetchone()
        if latest_holding:
            lines.append(f"Projected Monthly: {_fmt_money(latest_holding[0])}")
            lines.append(f"Yield: {_fmt_pct(latest_holding[1])}")
            if latest_holding[2]:
                lines.append(f"Next Ex-Date: {latest_holding[2]}")
            if latest_holding[3]:
                lines.append(f"Last Ex-Date: {latest_holding[3]}")

        if latest_upcoming:
            upcoming = conn.execute(
                """
                SELECT ex_date_est, pay_date_est, amount_est
                FROM daily_dividends_upcoming
                WHERE as_of_date_local = ? AND symbol = ?
                ORDER BY ex_date_est
                LIMIT 3
                """,
                (latest_upcoming, symbol_upper),
            ).fetchall()
            if upcoming:
                lines.append("")
                lines.append("<b>Upcoming:</b>")
                for ex_date, pay_date, amount_est in upcoming:
                    lines.append(f"  Ex {ex_date} | Pay {pay_date or '—'} | {_fmt_money(amount_est)}")

        if latest_tx:
            start, end = _window_bounds(latest_tx, 180)
            recent = conn.execute(
                """
                SELECT date, amount
                FROM investment_transactions
                WHERE symbol = ? AND transaction_type = 'dividend' AND date BETWEEN ? AND ?
                ORDER BY date DESC
                LIMIT 6
                """,
                (symbol_upper, start, end),
            ).fetchall()
            if recent:
                lines.append("")
                lines.append("<b>Recent Receipts:</b>")
                for tx_date, amount in recent:
                    lines.append(f"  {tx_date}: {_fmt_money(abs(float(amount or 0.0)))}")
        return "\n".join(lines)

    if mode == "recent30":
        if not latest_tx:
            return "No dividend transaction history available."
        start, end = _window_bounds(latest_tx, 30)
        rows = conn.execute(
            """
            SELECT date, symbol, amount
            FROM investment_transactions
            WHERE transaction_type = 'dividend'
              AND date BETWEEN ? AND ?
            ORDER BY date DESC
            LIMIT 12
            """,
            (start, end),
        ).fetchall()
        total = sum(abs(float(row[2] or 0.0)) for row in rows)
        lines = ["<b>💸 Recent Dividend Receipts</b>\n"]
        lines.append(f"Window: {start} → {end}")
        lines.append(f"Total Received: {_fmt_money(total)}")
        lines.append("")
        for tx_date, sym, amount in rows:
            lines.append(f"  {tx_date} | {sym or '?'} | {_fmt_money(abs(float(amount or 0.0)))}")
        return "\n".join(lines) if rows else "No recent dividend receipts found."

    if not latest_upcoming:
        return "No upcoming dividend calendar available."

    days_back = 7 if mode == "up7" else 30
    anchor_dt = _parse_iso_date(latest_upcoming)
    if not anchor_dt:
        return "No upcoming dividend calendar available."
    end_dt = anchor_dt + timedelta(days=days_back)
    rows = conn.execute(
        """
        SELECT symbol, ex_date_est, pay_date_est, amount_est
        FROM daily_dividends_upcoming
        WHERE as_of_date_local = ?
          AND ex_date_est BETWEEN ? AND ?
        ORDER BY ex_date_est, symbol
        """,
        (latest_upcoming, latest_upcoming, end_dt.isoformat()),
    ).fetchall()
    total = sum(float(row[3] or 0.0) for row in rows)
    title = "Next 7 Days" if mode == "up7" else "Next 30 Days"
    lines = [f"<b>📅 Dividend Calendar - {title}</b>\n"]
    lines.append(f"As of: {latest_upcoming}")
    lines.append(f"Projected Total: {_fmt_money(total)}")
    lines.append("")
    for sym, ex_date, pay_date, amount_est in rows[:12]:
        lines.append(f"  {sym}: Ex {ex_date} | Pay {pay_date or '—'} | {_fmt_money(amount_est)}")
    return "\n".join(lines) if rows else f"No dividend events in the next {days_back} days."


def _transaction_drilldown_text(conn, days_back: int, tx_filter: str = "all", symbol: str | None = None) -> str:
    latest_tx = _latest_table_date(conn, "investment_transactions", "date")
    start, end = _window_bounds(latest_tx, days_back)
    if not start or not end:
        return "No transaction history available."

    params: list[object] = [start, end]
    symbol_clause = ""
    if symbol:
        symbol_clause = " AND symbol = ?"
        params.append(symbol.upper())

    rows = conn.execute(
        f"""
        SELECT
          date,
          transaction_type,
          symbol,
          quantity,
          amount,
          name,
          economic_bucket,
          cash_flow_amount
        FROM investment_transactions
        WHERE date BETWEEN ? AND ?{symbol_clause}
        ORDER BY date DESC, lm_transaction_id DESC
        """,
        params,
    ).fetchall()
    filtered = []
    summary: dict[str, dict[str, float]] = {}
    for tx_date, tx_type, tx_symbol, quantity, amount, name, economic_bucket, cash_flow_amount in rows:
        bucket = (economic_bucket or _effective_tx_bucket(tx_type, name) or "other").lower()
        if not _matches_tx_filter(bucket, tx_type, tx_filter):
            continue
        cash_effect = _float_or_none(cash_flow_amount)
        if cash_effect is None:
            raw_amount = abs(float(amount or 0.0))
            if bucket in {"trade", "margin_interest", "withdrawal", "margin_repay", "fee"}:
                cash_effect = -raw_amount if bucket != "trade" or (tx_type or "").lower() == "buy" else raw_amount
            else:
                cash_effect = raw_amount
        filtered.append((tx_date, tx_type, tx_symbol, quantity, amount, name, bucket, cash_effect))
        stats = summary.setdefault(bucket, {"count": 0, "gross": 0.0, "net": 0.0})
        stats["count"] += 1
        stats["gross"] += abs(float(cash_effect or 0.0))
        stats["net"] += float(cash_effect or 0.0)

    if not filtered:
        target = f" for {symbol.upper()}" if symbol else ""
        return f"No {tx_filter} transactions found{target} in the selected window."

    title_target = f" - {symbol.upper()}" if symbol else ""
    lines = [f"<b>🧾 Transaction Drilldown{title_target}</b>\n"]
    lines.append(f"Window: {start} → {end}")
    lines.append(f"Filter: {tx_filter.replace('_', ' ').title()}")
    lines.append("")
    lines.append("<b>Summary:</b>")
    for bucket, stats in sorted(summary.items(), key=lambda item: item[1]["count"], reverse=True):
        lines.append(
            f"  {_tx_label(bucket, bucket)}: {int(stats['count'])} | "
            f"gross {_fmt_money(stats['gross'])} | net {_fmt_money(stats['net'])}"
        )

    lines.append("")
    lines.append("<b>Recent Transactions:</b>")
    for tx_date, tx_type, tx_symbol, quantity, amount, name, bucket, cash_effect in filtered[:12]:
        label = _tx_label(tx_type, bucket)
        amount_val = abs(float(amount or 0.0))
        qty = abs(float(quantity or 0.0))
        symbol_part = f" {tx_symbol}" if tx_symbol else ""
        if (tx_type or "").lower() in {"buy", "sell"}:
            lines.append(
                f"  {tx_date} | {label}{symbol_part} | {qty:,.4f} sh | "
                f"gross {_fmt_money(amount_val)} | cash {_fmt_money(cash_effect)}"
            )
        else:
            lines.append(f"  {tx_date} | {label}{symbol_part} | {_fmt_money(cash_effect)}")
    return "\n".join(lines)


def _nav_row(back_target: str) -> list[dict]:
    return [
        {"text": "◀ Back", "callback_data": back_target},
        {"text": "⌂ Home", "callback_data": "nav:root"},
    ]


def _root_menu_text(snap: dict | None) -> str:
    totals = _totals(snap or {})
    income = _income(snap or {})
    as_of = _as_of(snap)
    lines = ["<b>📱 Telegram Portfolio Menu</b>\n"]
    if as_of:
        lines.append(f"As of: {as_of}")
    if totals:
        lines.append(f"Net Liq: {_fmt_money(totals.get('net_liquidation_value'))}")
        lines.append(f"Projected Monthly: {_fmt_money(income.get('projected_monthly_income'))}")
    lines.append("")
    lines.append("Browse by section:")
    lines.append("• Portfolio: status, balance, snapshot, holdings, benchmark, allocation")
    lines.append("• Income: upcoming payments, income profile, MTD progress, receipts, dividend menu")
    lines.append("• Risk/Margin: drawdown, volatility, tail risk, rate shock, call buffer, coverage")
    lines.append("• History: NLV, market, unrealized, income, yield, LTV, margin buffer trends")
    lines.append("• Positions: top symbols -> position hub -> snapshot/history/transactions/dividends")
    lines.append("• Changes: daily compare plus final/rolling period attribution")
    lines.append("• Periods: weekly/monthly/quarterly/yearly reports with holdings/activity/risk/goals/margin drilldowns")
    lines.append("• Transactions: normalized cashflow, realized P&L windows, recent sales, raw ledger views")
    lines.append("• Planning: goal, net goal, pace, pace windows, projection, simulation, what-if, rebalance")
    lines.append("• System: health, alerts, settings, macro, data coverage")
    lines.append("• Charts / AI: fast chart shortcuts and daily/period insights")
    return "\n".join(lines)


def _positions_menu_payload(conn, sort_mode: str = "weight") -> tuple[str, dict]:
    if sort_mode == "income":
        top = _top_symbols(conn, sort_column="projected_monthly_dividend")
        title = "<b>📊 Positions - Top Income Contributors</b>\n"
        switch_label = "By Weight"
        switch_target = "nav:psw"
    else:
        top = _top_symbols(conn, sort_column="weight_pct")
        title = "<b>📊 Positions - Top By Weight</b>\n"
        switch_label = "By Income"
        switch_target = "nav:psi"

    lines = [title]
    lines.append("Tap a symbol to open its position hub.")
    lines.append("Path: Positions → Symbol → Snapshot / History / Transactions / Dividends")
    lines.append("")
    for row in top:
        lines.append(
            f"• {row['symbol']}: {_fmt_money(row.get('market_value'))} | "
            f"{_fmt_pct(row.get('weight_pct'))} | {_fmt_money(row.get('projected_monthly_dividend'))}/mo"
        )

    button_rows: list[list[dict]] = [[{"text": switch_label, "callback_data": switch_target}]]
    current_row: list[dict] = []
    for row in top:
        current_row.append({"text": row["symbol"], "callback_data": f"pos:{row['symbol']}"})
        if len(current_row) == 2:
            button_rows.append(current_row)
            current_row = []
    if current_row:
        button_rows.append(current_row)
    button_rows.append(_nav_row("nav:root"))
    return "\n".join(lines), build_inline_keyboard(button_rows)


def _dividends_menu_payload(conn) -> tuple[str, dict]:
    upcoming = _top_symbols(conn, sort_column="projected_monthly_dividend", limit=6)
    lines = ["<b>📅 Dividends</b>\n"]
    lines.append("Browse upcoming payments, recent receipts, or open a symbol hub for its full dividend path.")
    lines.append("Path: Income → Dividends → Symbol → Snapshot / Dividends / History / Transactions")
    if upcoming:
        lines.append("")
        lines.append("<b>Top Income Symbols:</b>")
        for row in upcoming:
            lines.append(f"• {row['symbol']}: {_fmt_money(row.get('projected_monthly_dividend'))}/mo")

    buttons: list[list[dict]] = [
        [
            {"text": "Next 7d", "callback_data": "div:up7"},
            {"text": "Next 30d", "callback_data": "div:up30"},
        ],
        [
            {"text": "Recent 30d", "callback_data": "div:recent30"},
        ],
    ]
    current_row: list[dict] = []
    for row in upcoming:
        current_row.append({"text": row["symbol"], "callback_data": f"pos:{row['symbol']}"})
        if len(current_row) == 2:
            buttons.append(current_row)
            current_row = []
    if current_row:
        buttons.append(current_row)
    buttons.append(_nav_row("nav:root"))
    return "\n".join(lines), build_inline_keyboard(buttons)


def _menu_payload(conn, section: str, snap: dict | None) -> tuple[str, dict]:
    normalized = (section or "root").lower()
    if normalized == "root":
        return _root_menu_text(snap), build_inline_keyboard([
            [
                {"text": "Portfolio", "callback_data": "nav:pf"},
                {"text": "Income", "callback_data": "nav:inc"},
            ],
            [
                {"text": "Risk/Margin", "callback_data": "nav:rm"},
                {"text": "History", "callback_data": "nav:hi"},
            ],
            [
                {"text": "Positions", "callback_data": "nav:psw"},
                {"text": "Changes", "callback_data": "nav:chg"},
            ],
            [
                {"text": "Periods", "callback_data": "nav:per"},
                {"text": "Dividends", "callback_data": "nav:div"},
            ],
            [
                {"text": "Transactions", "callback_data": "nav:tx"},
                {"text": "Planning", "callback_data": "nav:plan"},
            ],
            [
                {"text": "System", "callback_data": "nav:sys"},
                {"text": "Charts", "callback_data": "nav:ct"},
            ],
            [
                {"text": "AI", "callback_data": "nav:ai"},
            ],
        ])

    if normalized in {"ov", "pf"}:
        text = (
            "<b>📊 Portfolio</b>\n\n"
            "Current-state portfolio views. Inside this section: quick status, balance-sheet detail, snapshot metadata, holdings, benchmark-relative results, concentration, and the full daily digest."
        )
        markup = build_inline_keyboard([
            [
                {"text": "Status", "callback_data": "cmd:status"},
                {"text": "Balance", "callback_data": "cmd:balance"},
            ],
            [
                {"text": "Snapshot", "callback_data": "cmd:snapshot"},
                {"text": "Holdings", "callback_data": "cmd:holdings"},
            ],
            [
                {"text": "Allocation", "callback_data": "cmd:allocation"},
                {"text": "Benchmark", "callback_data": "cmd:benchmark"},
            ],
            [
                {"text": "Compare", "callback_data": "cmd:compare"},
                {"text": "Full Digest", "callback_data": "cmd:full"},
            ],
            [
                {"text": "Health", "callback_data": "cmd:health"},
            ],
            [_nav_row("nav:root")[0], _nav_row("nav:root")[1]],
        ])
        return text, markup

    if normalized == "inc":
        text = (
            "<b>💰 Income</b>\n\n"
            "Income and dividend views. Inside this section: upcoming payments, profile and growth, month-to-date progress, recent receipts, income history, and the full dividend calendar branch."
        )
        markup = build_inline_keyboard([
            [
                {"text": "Upcoming", "callback_data": "cmd:income"},
                {"text": "Profile", "callback_data": "cmd:income_profile"},
            ],
            [
                {"text": "MTD", "callback_data": "cmd:mtd"},
                {"text": "Received 30d", "callback_data": "cmd:received"},
            ],
            [
                {"text": "Dividend Menu", "callback_data": "nav:div"},
                {"text": "Yield 90d", "callback_data": "hist:yield:90"},
            ],
            [
                {"text": "Income 30d", "callback_data": "hist:income:30"},
                {"text": "Income 90d", "callback_data": "hist:income:90"},
            ],
            [
                {"text": "Calendar Chart", "callback_data": "chart:dividends"},
            ],
            [_nav_row("nav:root")[0], _nav_row("nav:root")[1]],
        ])
        return text, markup

    if normalized == "rm":
        text = (
            "<b>⚠️ Risk & Margin</b>\n\n"
            "Risk, leverage, and data-quality checks. Inside this section: volatility/drawdown, tail-risk detail, margin cost and safety buffer, rate-shock stress, macro context, and coverage health."
        )
        markup = build_inline_keyboard([
            [
                {"text": "Risk", "callback_data": "cmd:risk"},
                {"text": "Risk Detail", "callback_data": "cmd:risk_detail"},
            ],
            [
                {"text": "Margin", "callback_data": "cmd:margin"},
                {"text": "Rate Shock", "callback_data": "cmd:rate_shock"},
            ],
            [
                {"text": "Macro", "callback_data": "cmd:macro"},
                {"text": "Coverage", "callback_data": "cmd:coverage"},
            ],
            [
                {"text": "Health", "callback_data": "cmd:health"},
                {"text": "Benchmark", "callback_data": "cmd:benchmark"},
            ],
            [
                {"text": "Risk Chart", "callback_data": "chart:risk:90"},
                {"text": "Drawdown Chart", "callback_data": "chart:drawdown:180"},
            ],
            [_nav_row("nav:root")[0], _nav_row("nav:root")[1]],
        ])
        return text, markup

    if normalized == "hi":
        text = (
            "<b>📈 History</b>\n\n"
            "Choose a metric and window. History views are anchored to the latest stored daily snapshot and show current value, range, average, recent points, and chart shortcuts."
        )
        markup = build_inline_keyboard([
            [
                {"text": "NLV 30d", "callback_data": "hist:nlv:30"},
                {"text": "NLV 90d", "callback_data": "hist:nlv:90"},
            ],
            [
                {"text": "Income 30d", "callback_data": "hist:income:30"},
                {"text": "Income 90d", "callback_data": "hist:income:90"},
            ],
            [
                {"text": "Yield 90d", "callback_data": "hist:yield:90"},
                {"text": "LTV 90d", "callback_data": "hist:ltv:90"},
            ],
            [
                {"text": "Market 90d", "callback_data": "hist:market:90"},
                {"text": "Unrealized 90d", "callback_data": "hist:unrealized:90"},
            ],
            [
                {"text": "Buffer 90d", "callback_data": "hist:buffer:90"},
            ],
            [_nav_row("nav:root")[0], _nav_row("nav:root")[1]],
        ])
        return text, markup

    if normalized in {"ps", "psw", "psi"}:
        sort_mode = "income" if normalized == "psi" else "weight"
        return _positions_menu_payload(conn, sort_mode=sort_mode)

    if normalized == "chg":
        text = (
            "<b>🔎 Changes</b>\n\n"
            "Attribution views focus on what actually moved value and income: daily compare, final weekly/monthly/quarterly/yearly periods, and rolling month-to-date."
        )
        markup = build_inline_keyboard([
            [
                {"text": "Daily Compare", "callback_data": "cmd:compare"},
                {"text": "Weekly", "callback_data": "chg:weekly:f"},
            ],
            [
                {"text": "Monthly", "callback_data": "chg:monthly:f"},
                {"text": "Quarterly", "callback_data": "chg:quarterly:f"},
            ],
            [
                {"text": "Yearly", "callback_data": "chg:yearly:f"},
                {"text": "Rolling MTD", "callback_data": "chg:monthly:r"},
            ],
            [_nav_row("nav:root")[0], _nav_row("nav:root")[1]],
        ])
        return text, markup

    if normalized == "div":
        return _dividends_menu_payload(conn)

    if normalized == "tx":
        text = (
            "<b>🧾 Transactions</b>\n\n"
            "Drill into normalized cash movement, realized gains/losses, recent sales, and the underlying ledger."
        )
        markup = build_inline_keyboard([
            [
                {"text": "All 7d", "callback_data": "tx:7:all"},
                {"text": "All 30d", "callback_data": "tx:30:all"},
            ],
            [
                {"text": "Trades 30d", "callback_data": "tx:30:trade"},
                {"text": "Dividends 60d", "callback_data": "tx:60:dividend"},
            ],
            [
                {"text": "Cash 60d", "callback_data": "tx:60:cash"},
                {"text": "Margin 90d", "callback_data": "tx:90:margin"},
            ],
            [
                {"text": "P&L YTD", "callback_data": "cg:ytd"},
                {"text": "Cashflow YTD", "callback_data": "cf:ytd"},
            ],
            [
                {"text": "P&L 30D", "callback_data": "cg:30d"},
                {"text": "Recent Sales", "callback_data": "sales:10"},
            ],
            [_nav_row("nav:root")[0], _nav_row("nav:root")[1]],
        ])
        return text, markup

    if normalized == "plan":
        text = (
            "<b>🎯 Planning</b>\n\n"
            "Goal and decision support views. Inside this section: gross/net progress, pace, pace windows, projections, scenario simulation, what-if math, and rebalance guidance."
        )
        markup = build_inline_keyboard([
            [
                {"text": "Goal", "callback_data": "cmd:goal"},
                {"text": "Net Goal", "callback_data": "cmd:goal_net"},
            ],
            [
                {"text": "Pace", "callback_data": "cmd:pace"},
                {"text": "Pace Windows", "callback_data": "cmd:pace_windows"},
            ],
            [
                {"text": "Projection", "callback_data": "cmd:projection"},
                {"text": "Simulate", "callback_data": "cmd:simulate"},
            ],
            [
                {"text": "What-If", "callback_data": "cmd:whatif"},
                {"text": "Rebalance", "callback_data": "cmd:rebalance"},
            ],
            [_nav_row("nav:root")[0], _nav_row("nav:root")[1]],
        ])
        return text, markup

    if normalized == "per":
        text = (
            "<b>📅 Period Reports</b>\n\n"
            "Latest built period summaries. Each report opens with drilldowns for holdings, trades, activity, risk, and AI insight buttons."
        )
        markup = build_inline_keyboard([
            [
                {"text": "Weekly", "callback_data": "period:weekly"},
                {"text": "Monthly", "callback_data": "period:monthly"},
            ],
            [
                {"text": "Quarterly", "callback_data": "period:quarterly"},
                {"text": "Yearly", "callback_data": "period:yearly"},
            ],
            [_nav_row("nav:root")[0], _nav_row("nav:root")[1]],
        ])
        return text, markup

    if normalized == "sys":
        text = (
            "<b>🛠️ System</b>\n\n"
            "Bot and data-health views. Inside this section: alert state, settings, macro snapshot, trend history, coverage quality, and health checks."
        )
        markup = build_inline_keyboard([
            [
                {"text": "Alerts", "callback_data": "cmd:alerts"},
                {"text": "Settings", "callback_data": "cmd:settings"},
            ],
            [
                {"text": "Health", "callback_data": "cmd:health"},
                {"text": "Coverage", "callback_data": "cmd:coverage"},
            ],
            [
                {"text": "Macro", "callback_data": "cmd:macro"},
                {"text": "Trend", "callback_data": "cmd:trend"},
            ],
            [
                {"text": "Snapshot", "callback_data": "cmd:snapshot"},
                {"text": "Daily Summary", "callback_data": "cmd:status"},
            ],
            [_nav_row("nav:root")[0], _nav_row("nav:root")[1]],
        ])
        return text, markup

    if normalized == "ct":
        text = (
            "<b>📉 Charts</b>\n\n"
            "Chart shortcuts for the data already in your snapshots: NLV, income, yield, pace, risk, drawdown, allocation, margin, attribution, and dividend calendar."
        )
        markup = build_inline_keyboard([
            [
                {"text": "Pace 90d", "callback_data": "chart:pace:90"},
                {"text": "NLV 90d", "callback_data": "chart:performance:90"},
            ],
            [
                {"text": "Income 90d", "callback_data": "chart:income:90"},
                {"text": "Yield 90d", "callback_data": "chart:yield:90"},
            ],
            [
                {"text": "Risk 90d", "callback_data": "chart:risk:90"},
                {"text": "Drawdown 180d", "callback_data": "chart:drawdown:180"},
            ],
            [
                {"text": "Allocation", "callback_data": "chart:allocation:90"},
                {"text": "Margin 90d", "callback_data": "chart:margin:90"},
            ],
            [
                {"text": "Attribution", "callback_data": "chart:attribution:90"},
                {"text": "Div Calendar", "callback_data": "chart:dividends"},
            ],
            [_nav_row("nav:root")[0], _nav_row("nav:root")[1]],
        ])
        return text, markup

    if normalized == "ai":
        text = (
            "<b>🤖 AI</b>\n\n"
            "AI views for the current daily snapshot or the latest built period summaries."
        )
        markup = build_inline_keyboard([
            [
                {"text": "Daily Quick", "callback_data": "insights:quick"},
                {"text": "Daily Deep", "callback_data": "insights:deep"},
            ],
            [
                {"text": "Weekly Quick", "callback_data": "period_insights:weekly:quick"},
                {"text": "Monthly Quick", "callback_data": "period_insights:monthly:quick"},
            ],
            [
                {"text": "Weekly Deep", "callback_data": "period_insights:weekly:deep"},
                {"text": "Monthly Deep", "callback_data": "period_insights:monthly:deep"},
            ],
            [_nav_row("nav:root")[0], _nav_row("nav:root")[1]],
        ])
        return text, markup

    return _menu_payload(conn, "root", snap)


def _menu_markup(conn, snap: dict | None, section: str = "root") -> dict:
    return _menu_payload(conn, section, snap)[1]


def _history_detail_markup(metric: str, days_back: int) -> dict:
    meta = _HISTORY_METRICS.get(metric, _HISTORY_METRICS["nlv"])
    return build_inline_keyboard([
        [
            {"text": "30d", "callback_data": f"hist:{metric}:30"},
            {"text": "90d", "callback_data": f"hist:{metric}:90"},
            {"text": "180d", "callback_data": f"hist:{metric}:180"},
        ],
        [
            {"text": "365d", "callback_data": f"hist:{metric}:365"},
            {"text": "Chart", "callback_data": f"chart:{meta['chart']}:{days_back}"},
        ],
        [_nav_row("nav:hi")[0], _nav_row("nav:hi")[1]],
    ])


def _position_detail_markup(symbol: str, days_back: int) -> dict:
    symbol_upper = symbol.upper()
    return build_inline_keyboard([
        [
            {"text": "Current", "callback_data": f"pcur:{symbol_upper}"},
            {"text": "Dividends", "callback_data": f"dvs:{symbol_upper}"},
        ],
        [
            {"text": "30d", "callback_data": f"ph:{symbol_upper}:30"},
            {"text": "90d", "callback_data": f"ph:{symbol_upper}:90"},
            {"text": "180d", "callback_data": f"ph:{symbol_upper}:180"},
        ],
        [
            {"text": "Tx 30d", "callback_data": f"ptx:{symbol_upper}:30"},
            {"text": "Tx 90d", "callback_data": f"ptx:{symbol_upper}:90"},
        ],
        [
            {"text": "◀ Hub", "callback_data": f"pos:{symbol_upper}"},
            {"text": "⌂ Home", "callback_data": "nav:root"},
        ],
    ])


def _change_detail_markup(kind: str, rolling: bool) -> dict:
    mode = "r" if rolling else "f"
    return build_inline_keyboard([
        [
            {"text": "Week", "callback_data": "chg:weekly:f"},
            {"text": "Month", "callback_data": "chg:monthly:f"},
            {"text": "Quarter", "callback_data": "chg:quarterly:f"},
        ],
        [
            {"text": "Year", "callback_data": "chg:yearly:f"},
            {"text": "Rolling MTD", "callback_data": "chg:monthly:r"},
        ],
        [
            {"text": "Refresh", "callback_data": f"chg:{kind}:{mode}"},
        ],
        [_nav_row("nav:chg")[0], _nav_row("nav:chg")[1]],
    ])


def _dividend_detail_markup(mode: str, symbol: str | None = None) -> dict:
    buttons: list[list[dict]] = [
        [
            {"text": "Next 7d", "callback_data": "div:up7"},
            {"text": "Next 30d", "callback_data": "div:up30"},
        ],
        [
            {"text": "Recent 30d", "callback_data": "div:recent30"},
        ],
    ]
    if symbol:
        buttons.insert(0, [{"text": f"{symbol.upper()} Refresh", "callback_data": f"dvs:{symbol.upper()}"}])
        buttons.append(
            [
                {"text": "Position Hub", "callback_data": f"pos:{symbol.upper()}"},
                {"text": "Current", "callback_data": f"pcur:{symbol.upper()}"},
            ]
        )
        buttons.append(
            [
                {"text": "◀ Hub", "callback_data": f"pos:{symbol.upper()}"},
                {"text": "⌂ Home", "callback_data": "nav:root"},
            ]
        )
        return build_inline_keyboard(buttons)
    buttons.append(_nav_row("nav:div"))
    return build_inline_keyboard(buttons)


def _tx_detail_markup(days_back: int, tx_filter: str, symbol: str | None = None) -> dict:
    base = symbol.upper() if symbol else ""
    buttons: list[list[dict]]
    if symbol:
        buttons = [
            [
                {"text": "Current", "callback_data": f"pcur:{base}"},
                {"text": "Dividends", "callback_data": f"dvs:{base}"},
            ],
            [
                {"text": "30d", "callback_data": f"ptx:{base}:30"},
                {"text": "90d", "callback_data": f"ptx:{base}:90"},
            ],
            [
                {"text": "◀ Hub", "callback_data": f"pos:{base}"},
                {"text": "⌂ Home", "callback_data": "nav:root"},
            ],
        ]
    else:
        buttons = [
            [
                {"text": "7d", "callback_data": f"tx:7:{tx_filter}"},
                {"text": "30d", "callback_data": f"tx:30:{tx_filter}"},
                {"text": "90d", "callback_data": f"tx:90:{tx_filter}"},
            ],
            [
                {"text": "All", "callback_data": f"tx:{days_back}:all"},
                {"text": "Trades", "callback_data": f"tx:{days_back}:trade"},
                {"text": "Dividends", "callback_data": f"tx:{days_back}:dividend"},
            ],
            [
                {"text": "Cash", "callback_data": f"tx:{days_back}:cash"},
                {"text": "Margin", "callback_data": f"tx:{days_back}:margin"},
            ],
            [
                {"text": "P&L YTD", "callback_data": "cg:ytd"},
                {"text": "Cashflow YTD", "callback_data": "cf:ytd"},
            ],
            [
                {"text": "Recent Sales", "callback_data": "sales:10"},
            ],
            [_nav_row("nav:tx")[0], _nav_row("nav:tx")[1]],
        ]
    return build_inline_keyboard(buttons)


def _period_report_markup(period_snap: dict, period_kind: str) -> dict:
    detail_keyboard = build_period_insight_keyboard(period_snap).get("inline_keyboard") or []
    rows = [
        [
            {"text": "🤖 AI Insight", "callback_data": f"period_insights:{period_kind}:quick"},
            {"text": "🧠 Deep Analysis", "callback_data": f"period_insights:{period_kind}:deep"},
        ]
    ]
    rows.extend(detail_keyboard)
    return build_inline_keyboard(rows)


def _cached_menu_payload(conn, section: str, snap: dict | None) -> tuple[str, dict]:
    normalized = (section or "root").lower()
    return _cached_surface_artifact(
        conn,
        "menu_payload",
        lambda: _menu_payload(conn, normalized, snap),
        normalized,
    )


def _cached_command_text(conn, command: str, producer) -> str:
    return _cached_surface_artifact(
        conn,
        "command_text",
        producer,
        command,
    )


def _cached_position_hub_view(conn, snap: dict | None, symbol: str) -> str:
    return _cached_runtime_artifact(
        _cache_key(
            "position_hub_text",
            _daily_signature(conn),
            _holdings_signature(conn),
            _transactions_signature(conn),
            symbol.upper(),
        ),
        lambda: _position_hub_text(conn, snap, symbol),
        ttl_seconds=_TELEGRAM_VIEW_TTL_SECONDS,
    )


def _cached_position_current_view(conn, snap: dict | None, symbol: str) -> str:
    return _cached_runtime_artifact(
        _cache_key(
            "position_current_text",
            _daily_signature(conn),
            _holdings_signature(conn),
            symbol.upper(),
        ),
        lambda: _position_text(snap or {}, symbol),
        ttl_seconds=_TELEGRAM_VIEW_TTL_SECONDS,
    )


def _cached_daily_report(conn) -> tuple[str | None, str | None]:
    return _cached_surface_artifact(
        conn,
        "daily_report_html",
        lambda: build_daily_report_html(conn),
    )


def _cached_quick_summary(conn, snap: dict) -> str:
    return _cached_surface_artifact(
        conn,
        "daily_quick_summary",
        lambda: _quick_summary_text(snap, conn),
    )


def _cached_period_report(conn, period_kind: str) -> tuple[str | None, str | None]:
    normalized = _normalize_period_kind(period_kind) or period_kind
    return _cached_runtime_artifact(
        _cache_key(
            "period_report_html",
            _period_signature(conn, normalized, False),
            _period_signature(conn, normalized, True),
            normalized,
        ),
        lambda: build_period_report_html(conn, normalized),
        ttl_seconds=_TELEGRAM_VIEW_TTL_SECONDS,
    )


def _cached_period_detail_text(conn, action: str, period_snap: dict) -> str:
    period_meta = ((period_snap.get("meta") or {}).get("period") or {}) if isinstance(period_snap, dict) else {}
    period_kind = _normalize_period_kind(period_meta.get("type")) or "weekly"
    period_start = period_meta.get("start_date_local") or ""
    period_end = period_meta.get("end_date_local") or ""
    formatter_map = {
        "period_holdings": format_period_holdings_html,
        "period_trades": format_period_trades_html,
        "period_activity": format_period_activity_html,
        "period_pnl": format_period_pnl_html,
        "period_risk": format_period_risk_html,
        "period_goals": format_period_goals_html,
        "period_margin": format_period_margin_html,
    }
    formatter = formatter_map.get(action)
    return _cached_runtime_artifact(
        _cache_key(
            "period_detail_text",
            _period_signature(conn, period_kind, False),
            _period_signature(conn, period_kind, True),
            action,
            period_kind,
            period_start,
            period_end,
        ),
        lambda: formatter(period_snap) if formatter else "No period action available.",
        ttl_seconds=_TELEGRAM_VIEW_TTL_SECONDS,
    )


def _cached_morning_brief(conn) -> tuple[str | None, str | None]:
    return _cached_surface_artifact(
        conn,
        "morning_brief_html",
        lambda: build_morning_brief_html(conn),
    )


def _cached_evening_recap(conn) -> tuple[str | None, str | None]:
    return _cached_surface_artifact(
        conn,
        "evening_recap_html",
        lambda: build_evening_recap_html(conn),
    )


def _cached_history_view(conn, metric: str, days_back: int) -> str:
    return _cached_runtime_artifact(
        _cache_key("history_text", _daily_signature(conn), metric, int(days_back)),
        lambda: _history_text(conn, metric, days_back),
        ttl_seconds=_TELEGRAM_VIEW_TTL_SECONDS,
    )


def _cached_position_history_view(conn, symbol: str, days_back: int) -> str:
    return _cached_runtime_artifact(
        _cache_key(
            "position_history_text",
            _holdings_signature(conn),
            _transactions_signature(conn),
            symbol.upper(),
            int(days_back),
        ),
        lambda: _position_history_text(conn, symbol, days_back),
        ttl_seconds=_TELEGRAM_VIEW_TTL_SECONDS,
    )


def _cached_period_change_view(conn, kind: str, rolling: bool = False) -> str:
    normalized = _normalize_period_kind(kind) or kind
    return _cached_runtime_artifact(
        _cache_key("period_change_text", _period_signature(conn, normalized, rolling)),
        lambda: _period_change_text(conn, normalized, rolling=rolling),
        ttl_seconds=_TELEGRAM_VIEW_TTL_SECONDS,
    )


def _cached_dividend_calendar_view(conn, mode: str = "up30", symbol: str | None = None) -> str:
    return _cached_runtime_artifact(
        _cache_key(
            "dividend_calendar_text",
            _dividends_signature(conn),
            _transactions_signature(conn),
            _holdings_signature(conn),
            mode,
            (symbol or "").upper(),
        ),
        lambda: _dividend_calendar_text(conn, mode=mode, symbol=symbol),
        ttl_seconds=_TELEGRAM_VIEW_TTL_SECONDS,
    )


def _cached_transaction_view(conn, days_back: int, tx_filter: str = "all", symbol: str | None = None) -> str:
    return _cached_runtime_artifact(
        _cache_key(
            "transaction_drilldown_text",
            _transactions_signature(conn),
            int(days_back),
            tx_filter,
            (symbol or "").upper(),
        ),
        lambda: _transaction_drilldown_text(conn, days_back, tx_filter, symbol=symbol),
        ttl_seconds=_TELEGRAM_VIEW_TTL_SECONDS,
    )


def _cached_capital_gains_view(conn, window_key: str = "ytd") -> str:
    normalized = _normalize_window_key(window_key, "ytd")

    def _build():
        _, snap = _latest_snapshot(conn)
        return _capital_gains_text(snap or {}, normalized) if snap else "No daily snapshot available."

    return _cached_runtime_artifact(
        _cache_key(
            "capital_gains_text",
            _daily_signature(conn),
            _transactions_signature(conn),
            normalized,
        ),
        _build,
        ttl_seconds=_TELEGRAM_VIEW_TTL_SECONDS,
    )


def _cached_cashflow_view(conn, window_key: str = "ytd") -> str:
    normalized = _normalize_window_key(window_key, "ytd")

    def _build():
        _, snap = _latest_snapshot(conn)
        return _cashflow_text(snap or {}, normalized) if snap else "No daily snapshot available."

    return _cached_runtime_artifact(
        _cache_key(
            "cashflow_text",
            _daily_signature(conn),
            _transactions_signature(conn),
            normalized,
        ),
        _build,
        ttl_seconds=_TELEGRAM_VIEW_TTL_SECONDS,
    )


def _cached_sales_view(conn, limit: int = 10) -> str:
    normalized_limit = max(1, min(int(limit), 25))
    return _cached_runtime_artifact(
        _cache_key("recent_sales_text", _transactions_signature(conn), normalized_limit),
        lambda: _recent_sales_text(conn, normalized_limit),
        ttl_seconds=_TELEGRAM_VIEW_TTL_SECONDS,
    )


def _cached_chart_image(conn, chart_type: str, chart_days: int = 90, snap: dict | None = None) -> bytes | None:
    chart_key = _cache_key(
        "telegram_chart",
        _daily_signature(conn),
        _dividends_signature(conn),
        chart_type,
        int(chart_days),
    )

    def _build_chart():
        from ..services.charts import (
            generate_allocation_chart,
            generate_attribution_chart,
            generate_dividend_calendar_chart,
            generate_drawdown_chart,
            generate_income_chart,
            generate_margin_chart,
            generate_pace_chart,
            generate_performance_chart,
            generate_risk_chart,
            generate_yield_chart,
        )

        _, latest_snap = _latest_snapshot(conn)
        active_snap = snap or latest_snap
        chart_gen = {
            "pace": lambda: generate_pace_chart(conn, days=chart_days),
            "income": lambda: generate_income_chart(conn, days=chart_days),
            "performance": lambda: generate_performance_chart(conn, days=chart_days),
            "perf": lambda: generate_performance_chart(conn, days=chart_days),
            "attribution": lambda: generate_attribution_chart(active_snap) if active_snap else None,
            "yield": lambda: generate_yield_chart(conn, days=chart_days),
            "risk": lambda: generate_risk_chart(conn, days=chart_days),
            "drawdown": lambda: generate_drawdown_chart(conn, days=chart_days),
            "dd": lambda: generate_drawdown_chart(conn, days=chart_days),
            "allocation": lambda: generate_allocation_chart(active_snap) if active_snap else None,
            "alloc": lambda: generate_allocation_chart(active_snap) if active_snap else None,
            "margin": lambda: generate_margin_chart(conn, days=chart_days),
            "dividends": lambda: generate_dividend_calendar_chart(active_snap) if active_snap else None,
            "divs": lambda: generate_dividend_calendar_chart(active_snap) if active_snap else None,
        }
        gen = chart_gen.get(chart_type)
        return gen() if gen else None

    return _cached_runtime_artifact(chart_key, _build_chart, ttl_seconds=_TELEGRAM_CHART_TTL_SECONDS)


def _telegram_warm_signature(conn) -> tuple:
    return _telegram_surface_signature(conn)


def _warm_telegram_cache(db_path: str):
    conn = get_conn(db_path)
    try:
        _, snap = _latest_snapshot(conn)
        if not snap:
            return

        menu_sections = ("root", "pf", "inc", "rm", "hi", "psw", "psi", "chg", "div", "tx", "plan", "per", "sys", "ct", "ai")
        for section in menu_sections:
            _cached_menu_payload(conn, section, snap)

        cached_commands = (
            "status",
            "balance",
            "income",
            "income_profile",
            "pnl",
            "cashflow",
            "sales",
            "goal",
            "goal_net",
            "perf",
            "risk",
            "risk_detail",
            "margin",
            "rate_shock",
            "benchmark",
            "allocation",
            "coverage",
            "pace",
            "pace_windows",
            "alerts",
            "health",
            "macro",
            "mtd",
            "holdings",
            "goals",
            "received",
            "simulate",
            "projection",
            "snapshot",
            "settings",
            "compare",
            "full",
            "whatif",
            "trend",
            "rebalance",
        )
        _, as_of_snap = _latest_snapshot(conn)
        as_of = _as_of(as_of_snap)
        no_snap = "No snapshot."

        def _command_handler(command: str):
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
                _, html = _cached_daily_report(conn)
                return html or no_snap

            cmd_map = {
                "status": lambda: _status_text(snap) if snap else no_snap,
                "balance": lambda: _balance_text(snap) if snap else no_snap,
                "income": lambda: _income_text(snap) if snap else no_snap,
                "income_profile": lambda: _income_profile_text(snap) if snap else no_snap,
                "pnl": lambda: _capital_gains_text(snap) if snap else no_snap,
                "cashflow": lambda: _cashflow_text(snap) if snap else no_snap,
                "sales": lambda: _recent_sales_text(conn, 10),
                "goal": lambda: _goal_text(snap) if snap else no_snap,
                "goal_net": lambda: _goal_net_text(snap) if snap else no_snap,
                "perf": lambda: _perf_text(snap) if snap else no_snap,
                "risk": lambda: _risk_text(snap) if snap else no_snap,
                "risk_detail": lambda: _risk_detail_text(snap) if snap else no_snap,
                "margin": lambda: _margin_text(snap) if snap else no_snap,
                "rate_shock": lambda: _rate_shock_text(snap) if snap else no_snap,
                "benchmark": lambda: _benchmark_text(snap) if snap else no_snap,
                "allocation": lambda: _allocation_text(snap) if snap else no_snap,
                "coverage": lambda: _coverage_text(snap) if snap else no_snap,
                "pace": lambda: _pace_text(snap) if snap else no_snap,
                "pace_windows": lambda: _pace_windows_text(snap) if snap else no_snap,
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
            return cmd_map[command]

        for command in cached_commands:
            _cached_command_text(conn, command, _command_handler(command))

        _cached_quick_summary(conn, snap)
        _cached_daily_report(conn)
        _cached_morning_brief(conn)
        _cached_evening_recap(conn)

        history_days = (30, 90, 180, 365)
        for metric in _HISTORY_METRICS:
            for days_back in history_days:
                _cached_history_view(conn, metric, days_back)

        for kind in ("weekly", "monthly", "quarterly", "yearly"):
            _cached_period_change_view(conn, kind, rolling=False)
            _cached_period_report(conn, kind)
            period_snap = _latest_period_target_snapshot(conn, kind, prefer_rolling=True)
            if period_snap:
                for action in ("period_holdings", "period_trades", "period_activity", "period_pnl", "period_risk", "period_goals", "period_margin"):
                    _cached_period_detail_text(conn, action, period_snap)
        _cached_period_change_view(conn, "monthly", rolling=True)

        for mode in ("up7", "up30", "recent30"):
            _cached_dividend_calendar_view(conn, mode=mode)

        for window_key in ("mtd", "30d", "qtd", "ytd", "ltd"):
            _cached_capital_gains_view(conn, window_key)
            _cached_cashflow_view(conn, window_key)
        _cached_sales_view(conn, 10)

        for days_back, tx_filter in (
            (7, "all"),
            (30, "all"),
            (30, "trade"),
            (60, "dividend"),
            (60, "cash"),
            (90, "margin"),
        ):
            _cached_transaction_view(conn, days_back, tx_filter)

        latest_holdings_date = _latest_table_date(conn, "daily_holdings", "as_of_date_local")
        symbols = []
        if latest_holdings_date:
            rows = conn.execute(
                """
                SELECT symbol
                FROM daily_holdings
                WHERE as_of_date_local = ?
                ORDER BY weight_pct DESC, market_value DESC, symbol
                """,
                (latest_holdings_date,),
            ).fetchall()
            symbols = [row[0].upper() for row in rows if row and row[0]]

        for symbol in symbols:
            _cached_position_hub_view(conn, snap, symbol)
            _cached_position_current_view(conn, snap, symbol)
            _cached_dividend_calendar_view(conn, symbol=symbol)
            for days_back in (30, 90, 180):
                _cached_position_history_view(conn, symbol, days_back)
            for days_back in (30, 90):
                _cached_transaction_view(conn, days_back, "all", symbol=symbol)

        for chart_type, chart_days in (
            ("pace", 90),
            ("performance", 30),
            ("performance", 90),
            ("performance", 180),
            ("performance", 365),
            ("income", 30),
            ("income", 90),
            ("income", 180),
            ("income", 365),
            ("yield", 30),
            ("yield", 90),
            ("yield", 180),
            ("yield", 365),
            ("margin", 30),
            ("margin", 90),
            ("margin", 180),
            ("margin", 365),
            ("risk", 90),
            ("drawdown", 180),
            ("allocation", 90),
            ("attribution", 90),
            ("dividends", 90),
        ):
            _cached_chart_image(conn, chart_type, chart_days, snap=snap)
    finally:
        conn.close()


def warm_telegram_cache_async(db_path: str | None = None):
    target_db = db_path or settings.db_path
    conn = get_conn(target_db)
    try:
        signature = _telegram_surface_signature(conn)
    finally:
        conn.close()

    global _LAST_WARMED_SIGNATURE, _WARM_IN_FLIGHT
    with _WARM_STATE_LOCK:
        if _WARM_IN_FLIGHT or signature == _LAST_WARMED_SIGNATURE:
            return
        _WARM_IN_FLIGHT = True

    def _runner():
        global _LAST_WARMED_SIGNATURE, _WARM_IN_FLIGHT
        try:
            _warm_telegram_cache(target_db)
            with _WARM_STATE_LOCK:
                _LAST_WARMED_SIGNATURE = signature
        finally:
            with _WARM_STATE_LOCK:
                _WARM_IN_FLIGHT = False

    threading.Thread(target=_runner, name="telegram-cache-warm", daemon=True).start()


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


async def _send_or_edit_html(
    tg: TelegramClient,
    chat_id: int | str,
    message_id: int | None,
    text_html: str,
    *,
    reply_markup: dict | None = None,
    prefer_edit: bool = False,
):
    if prefer_edit and message_id:
        ok = await tg.edit_message_text(chat_id, message_id, text_html, reply_markup=reply_markup)
        if ok:
            return True
    return await tg.send_message_html(text_html, reply_markup=reply_markup)


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

    async def _show(text_html: str, *, reply_markup: dict | None = None, prefer_edit: bool = False):
        await _send_or_edit_html(
            tg,
            chat_id,
            message_id,
            text_html,
            reply_markup=reply_markup,
            prefer_edit=prefer_edit,
        )

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

    elif action == "nav":
        _, snap = _latest_snapshot(conn)
        menu_text, markup = _cached_menu_payload(conn, param or "root", snap)
        await _show(menu_text, reply_markup=markup, prefer_edit=True)

    elif action == "hist":
        hist_parts = (param or "").split(":")
        metric = (hist_parts[0] if hist_parts and hist_parts[0] else "nlv").lower()
        days_back = int(hist_parts[1]) if len(hist_parts) > 1 and hist_parts[1].isdigit() else 90
        await _show(
            _cached_history_view(conn, metric, days_back),
            reply_markup=_history_detail_markup(metric, days_back),
            prefer_edit=True,
        )

    elif action == "ph":
        hist_parts = (param or "").split(":")
        symbol = (hist_parts[0] if hist_parts and hist_parts[0] else "").upper()
        days_back = int(hist_parts[1]) if len(hist_parts) > 1 and hist_parts[1].isdigit() else 90
        if not symbol:
            await _show("Position view requires a symbol.", prefer_edit=True)
        else:
            await _show(
                _cached_position_history_view(conn, symbol, days_back),
                reply_markup=_position_detail_markup(symbol, days_back),
                prefer_edit=True,
            )

    elif action == "pos":
        symbol = (param or "").upper()
        if not symbol:
            await _show("Position hub requires a symbol.", prefer_edit=True)
        else:
            _, snap = _latest_snapshot(conn)
            await _show(
                _cached_position_hub_view(conn, snap, symbol),
                reply_markup=_position_hub_markup(symbol),
                prefer_edit=True,
            )

    elif action == "pcur":
        symbol = (param or "").upper()
        if not symbol:
            await _show("Current position view requires a symbol.", prefer_edit=True)
        else:
            _, snap = _latest_snapshot(conn)
            if not snap:
                await _show("No snapshot available.", prefer_edit=True)
            else:
                await _show(
                    _cached_position_current_view(conn, snap, symbol),
                    reply_markup=_position_current_markup(symbol),
                    prefer_edit=True,
                )

    elif action == "ptx":
        tx_parts = (param or "").split(":")
        symbol = (tx_parts[0] if tx_parts and tx_parts[0] else "").upper()
        days_back = int(tx_parts[1]) if len(tx_parts) > 1 and tx_parts[1].isdigit() else 30
        if not symbol:
            await _show("Transaction view requires a symbol.", prefer_edit=True)
        else:
            await _show(
                _cached_transaction_view(conn, days_back, "all", symbol=symbol),
                reply_markup=_tx_detail_markup(days_back, "all", symbol=symbol),
                prefer_edit=True,
            )

    elif action == "chg":
        change_parts = (param or "").split(":")
        kind = _normalize_period_kind(change_parts[0] if change_parts else "monthly") or "monthly"
        rolling = len(change_parts) > 1 and change_parts[1].lower() == "r"
        await _show(
            _cached_period_change_view(conn, kind, rolling=rolling),
            reply_markup=_change_detail_markup(kind, rolling),
            prefer_edit=True,
        )

    elif action == "div":
        mode = (param or "up30").lower()
        await _show(
            _cached_dividend_calendar_view(conn, mode=mode),
            reply_markup=_dividend_detail_markup(mode),
            prefer_edit=True,
        )

    elif action == "dvs":
        symbol = (param or "").upper()
        if not symbol:
            await _show("Dividend view requires a symbol.", prefer_edit=True)
        else:
            await _show(
                _cached_dividend_calendar_view(conn, symbol=symbol),
                reply_markup=_dividend_detail_markup("symbol", symbol=symbol),
                prefer_edit=True,
            )

    elif action == "tx":
        tx_parts = (param or "").split(":")
        days_back = int(tx_parts[0]) if tx_parts and tx_parts[0].isdigit() else 30
        tx_filter = (tx_parts[1] if len(tx_parts) > 1 and tx_parts[1] else "all").lower()
        await _show(
            _cached_transaction_view(conn, days_back, tx_filter),
            reply_markup=_tx_detail_markup(days_back, tx_filter),
            prefer_edit=True,
        )

    elif action == "cg":
        window_key = _normalize_window_key(param or "ytd", "ytd")
        await _show(
            _cached_capital_gains_view(conn, window_key),
            reply_markup=build_inline_keyboard([
                [
                    {"text": "MTD", "callback_data": "cg:mtd"},
                    {"text": "30D", "callback_data": "cg:30d"},
                    {"text": "YTD", "callback_data": "cg:ytd"},
                ],
                [
                    {"text": "QTD", "callback_data": "cg:qtd"},
                    {"text": "LTD", "callback_data": "cg:ltd"},
                    {"text": "Recent Sales", "callback_data": "sales:10"},
                ],
                _nav_row("nav:tx"),
            ]),
            prefer_edit=True,
        )

    elif action == "cf":
        window_key = _normalize_window_key(param or "ytd", "ytd")
        await _show(
            _cached_cashflow_view(conn, window_key),
            reply_markup=build_inline_keyboard([
                [
                    {"text": "MTD", "callback_data": "cf:mtd"},
                    {"text": "30D", "callback_data": "cf:30d"},
                    {"text": "YTD", "callback_data": "cf:ytd"},
                ],
                [
                    {"text": "QTD", "callback_data": "cf:qtd"},
                    {"text": "LTD", "callback_data": "cf:ltd"},
                    {"text": "Recent Sales", "callback_data": "sales:10"},
                ],
                _nav_row("nav:tx"),
            ]),
            prefer_edit=True,
        )

    elif action == "sales":
        limit = max(1, min(int(param), 25)) if (param or "").isdigit() else 10
        await _show(
            _cached_sales_view(conn, limit),
            reply_markup=build_inline_keyboard([
                [
                    {"text": "10", "callback_data": "sales:10"},
                    {"text": "15", "callback_data": "sales:15"},
                    {"text": "25", "callback_data": "sales:25"},
                ],
                [
                    {"text": "P&L YTD", "callback_data": "cg:ytd"},
                    {"text": "Cashflow YTD", "callback_data": "cf:ytd"},
                ],
                _nav_row("nav:tx"),
            ]),
            prefer_edit=True,
        )

    elif action == "cmd":
        # Menu button - execute a command and either refresh the menu panel or send a new message
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
            _, html = _cached_daily_report(conn)
            return html or no_snap

        cmd_map = {
            "status": lambda: _status_text(snap) if snap else no_snap,
            "balance": lambda: _balance_text(snap) if snap else no_snap,
            "income": lambda: _income_text(snap) if snap else no_snap,
            "income_profile": lambda: _income_profile_text(snap) if snap else no_snap,
            "pnl": lambda: _capital_gains_text(snap) if snap else no_snap,
            "cashflow": lambda: _cashflow_text(snap) if snap else no_snap,
            "sales": lambda: _recent_sales_text(conn, 10),
            "goal": lambda: _goal_text(snap) if snap else no_snap,
            "goal_net": lambda: _goal_net_text(snap) if snap else no_snap,
            "perf": lambda: _perf_text(snap) if snap else no_snap,
            "risk": lambda: _risk_text(snap) if snap else no_snap,
            "risk_detail": lambda: _risk_detail_text(snap) if snap else no_snap,
            "margin": lambda: _margin_text(snap) if snap else no_snap,
            "rate_shock": lambda: _rate_shock_text(snap) if snap else no_snap,
            "benchmark": lambda: _benchmark_text(snap) if snap else no_snap,
            "allocation": lambda: _allocation_text(snap) if snap else no_snap,
            "coverage": lambda: _coverage_text(snap) if snap else no_snap,
            "pace": lambda: _pace_text(snap) if snap else no_snap,
            "pace_windows": lambda: _pace_windows_text(snap) if snap else no_snap,
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
            detail_back_target = {
                "status": "nav:pf",
                "balance": "nav:pf",
                "snapshot": "nav:pf",
                "holdings": "nav:pf",
                "allocation": "nav:pf",
                "benchmark": "nav:pf",
                "compare": "nav:pf",
                "income": "nav:inc",
                "income_profile": "nav:inc",
                "mtd": "nav:inc",
                "received": "nav:inc",
                "perf": "nav:inc",
                "pnl": "nav:tx",
                "cashflow": "nav:tx",
                "sales": "nav:tx",
                "risk": "nav:rm",
                "risk_detail": "nav:rm",
                "margin": "nav:rm",
                "rate_shock": "nav:rm",
                "coverage": "nav:rm",
                "pace": "nav:plan",
                "pace_windows": "nav:plan",
                "goal": "nav:plan",
                "goal_net": "nav:plan",
                "projection": "nav:plan",
                "simulate": "nav:plan",
                "whatif": "nav:plan",
                "rebalance": "nav:plan",
                "alerts": "nav:sys",
                "settings": "nav:sys",
                "health": "nav:sys",
                "macro": "nav:sys",
                "trend": "nav:sys",
            }.get(param)
            detail_markup = build_inline_keyboard([_nav_row(detail_back_target)]) if detail_back_target else None
            await _show(
                _cached_command_text(conn, param, handler),
                reply_markup=detail_markup,
                prefer_edit=bool(detail_back_target),
            )

    elif action == "chart":
        try:
            chart_parts = param.split(":", 1)
            chart_type = chart_parts[0]
            chart_days = int(chart_parts[1]) if len(chart_parts) > 1 and chart_parts[1].isdigit() else 90

            _, snap = _latest_snapshot(conn)
            img = _cached_chart_image(conn, chart_type, chart_days, snap=snap)
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
            elif chart_type in {
                "pace", "income", "performance", "attribution", "yield",
                "risk", "drawdown", "allocation", "margin", "dividends",
            }:
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
                _as_of, html = _cached_period_report(conn, period_kind)
                if html:
                    period_snap = _latest_period_target_snapshot(conn, period_kind, prefer_rolling=True)
                    period_menu = _period_report_markup(period_snap, period_kind) if period_snap else None
                    await tg.send_message_html(html, reply_markup=period_menu)
                else:
                    await tg.send_message_html(f"No {period_kind} report available")
            except Exception:
                await tg.send_message_html(f"Error generating {period_kind} report")

    elif action in {"period_holdings", "period_trades", "period_activity", "period_pnl", "period_risk", "period_goals", "period_margin"}:
        period_snap = _target_period_snapshot_from_callback(conn, param)
        if not period_snap:
            await tg.send_message_html("No period snapshot found for this action.")
        else:
            html = _cached_period_detail_text(conn, action, period_snap)
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
            _, html = _cached_daily_report(conn)
            if html:
                await tg.send_message_html(html)
            else:
                await tg.send_message_html("No report available")
        elif param in {"weekly", "monthly", "quarterly", "yearly"}:
            try:
                _, html = _cached_period_report(conn, param)
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
    tg = TelegramClient(settings.telegram_bot_token, chat_id)
    latest_cache: tuple[str | None, dict | None] | None = None

    def _latest():
        nonlocal latest_cache
        if latest_cache is None:
            latest_cache = _latest_snapshot(conn)
        return latest_cache

    if cmd in {"start", "help"}:
        _, snap = _latest()
        await tg.send_message_html(_help_text(), reply_markup=_menu_markup(conn, snap, "root"))
        return {"ok": True}
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
    elif cmd == "menu":
        section_aliases = {
            "portfolio": "pf",
            "overview": "pf",
            "income": "inc",
            "risk": "rm",
            "margin": "rm",
            "history": "hi",
            "positions": "psw",
            "position": "psw",
            "changes": "chg",
            "dividends": "div",
            "calendar": "div",
            "transactions": "tx",
            "planning": "plan",
            "periods": "per",
            "system": "sys",
            "charts": "ct",
            "ai": "ai",
        }
        requested = section_aliases.get((args[0] if args else "root").lower(), "root")
        _, snap = _latest()
        menu_text, markup = _cached_menu_payload(conn, requested, snap)
        await tg.send_message_html(menu_text, reply_markup=markup)
        return {"ok": True}
    elif cmd == "history":
        if not args:
            _, snap = _latest()
            menu_text, markup = _cached_menu_payload(conn, "hi", snap)
            await tg.send_message_html(menu_text, reply_markup=markup)
            return {"ok": True}
        metric = args[0].lower()
        days_back = int(args[1]) if len(args) > 1 and args[1].isdigit() else 90
        await tg.send_message_html(
            _cached_history_view(conn, metric, days_back),
            reply_markup=_history_detail_markup(metric, days_back),
        )
        return {"ok": True}
    elif cmd == "changes":
        if not args:
            _, snap = _latest()
            menu_text, markup = _cached_menu_payload(conn, "chg", snap)
            await tg.send_message_html(menu_text, reply_markup=markup)
            return {"ok": True}
        period_kind = _normalize_period_kind(args[0])
        if not period_kind:
            reply = "Usage: /changes <weekly|monthly|quarterly|yearly> [rolling]"
        else:
            rolling = any(arg.lower() in {"rolling", "r", "mtd"} for arg in args[1:])
            await tg.send_message_html(
                _cached_period_change_view(conn, period_kind, rolling=rolling),
                reply_markup=_change_detail_markup(period_kind, rolling),
            )
            return {"ok": True}
    elif cmd in {"calendar", "divcalendar"}:
        if not args:
            _, snap = _latest()
            menu_text, markup = _cached_menu_payload(conn, "div", snap)
            await tg.send_message_html(menu_text, reply_markup=markup)
            return {"ok": True}
        first = args[0].lower()
        if first in {"7", "7d", "next7", "up7"}:
            mode = "up7"
            await tg.send_message_html(
                _cached_dividend_calendar_view(conn, mode=mode),
                reply_markup=_dividend_detail_markup(mode),
            )
            return {"ok": True}
        if first in {"30", "30d", "next30", "up30"}:
            mode = "up30"
            await tg.send_message_html(
                _cached_dividend_calendar_view(conn, mode=mode),
                reply_markup=_dividend_detail_markup(mode),
            )
            return {"ok": True}
        if first in {"recent", "recent30"}:
            mode = "recent30"
            await tg.send_message_html(
                _cached_dividend_calendar_view(conn, mode=mode),
                reply_markup=_dividend_detail_markup(mode),
            )
            return {"ok": True}
        symbol = args[0].upper()
        await tg.send_message_html(
            _cached_dividend_calendar_view(conn, symbol=symbol),
            reply_markup=_dividend_detail_markup("symbol", symbol=symbol),
        )
        return {"ok": True}
    elif cmd in {"transactions", "txns", "tx"}:
        if not args:
            _, snap = _latest()
            menu_text, markup = _cached_menu_payload(conn, "tx", snap)
            await tg.send_message_html(menu_text, reply_markup=markup)
            return {"ok": True}
        tx_filter_aliases = {
            "trades": "trade",
            "trade": "trade",
            "div": "dividend",
            "dividends": "dividend",
            "dividend": "dividend",
            "cash": "cash",
            "margin": "margin",
            "all": "all",
        }
        days_back = 30
        tx_filter = "all"
        for arg in args:
            if arg.isdigit():
                days_back = max(1, min(int(arg), 365))
            else:
                tx_filter = tx_filter_aliases.get(arg.lower(), tx_filter)
        await tg.send_message_html(
            _cached_transaction_view(conn, days_back, tx_filter),
            reply_markup=_tx_detail_markup(days_back, tx_filter),
        )
        return {"ok": True}
    elif cmd == "pnl":
        window_key = _normalize_window_key(args[0] if args else "ytd", "ytd")
        await tg.send_message_html(
            _cached_capital_gains_view(conn, window_key),
            reply_markup=build_inline_keyboard([
                [
                    {"text": "MTD", "callback_data": "cg:mtd"},
                    {"text": "30D", "callback_data": "cg:30d"},
                    {"text": "YTD", "callback_data": "cg:ytd"},
                ],
                [
                    {"text": "QTD", "callback_data": "cg:qtd"},
                    {"text": "LTD", "callback_data": "cg:ltd"},
                    {"text": "Recent Sales", "callback_data": "sales:10"},
                ],
                _nav_row("nav:tx"),
            ]),
        )
        return {"ok": True}
    elif cmd == "cashflow":
        window_key = _normalize_window_key(args[0] if args else "ytd", "ytd")
        await tg.send_message_html(
            _cached_cashflow_view(conn, window_key),
            reply_markup=build_inline_keyboard([
                [
                    {"text": "MTD", "callback_data": "cf:mtd"},
                    {"text": "30D", "callback_data": "cf:30d"},
                    {"text": "YTD", "callback_data": "cf:ytd"},
                ],
                [
                    {"text": "QTD", "callback_data": "cf:qtd"},
                    {"text": "LTD", "callback_data": "cf:ltd"},
                    {"text": "Recent Sales", "callback_data": "sales:10"},
                ],
                _nav_row("nav:tx"),
            ]),
        )
        return {"ok": True}
    elif cmd == "sales":
        limit = max(1, min(int(args[0]), 25)) if args and args[0].isdigit() else 10
        await tg.send_message_html(
            _cached_sales_view(conn, limit),
            reply_markup=build_inline_keyboard([
                [
                    {"text": "10", "callback_data": "sales:10"},
                    {"text": "15", "callback_data": "sales:15"},
                    {"text": "25", "callback_data": "sales:25"},
                ],
                [
                    {"text": "P&L YTD", "callback_data": "cg:ytd"},
                    {"text": "Cashflow YTD", "callback_data": "cf:ytd"},
                ],
                _nav_row("nav:tx"),
            ]),
        )
        return {"ok": True}
    elif cmd == "status":
        _, snap = _latest()
        reply = _status_text(snap) if snap else "No daily snapshot available."
    elif cmd == "balance":
        _, snap = _latest()
        reply = _balance_text(snap) if snap else "No daily snapshot available."
    elif cmd == "full":
        _, snap = _latest()
        if not snap:
            reply = "No daily snapshot available."
        else:
            summary = _cached_quick_summary(conn, snap)
            markup = build_inline_keyboard([
                [{"text": "📖 Show Full Report", "callback_data": "expand:full"}],
            ])
            await tg.send_message_html(summary, reply_markup=markup)
            return {"ok": True}
    elif cmd == "income":
        _, snap = _latest()
        reply = _income_text(snap) if snap else "No daily snapshot available."
    elif cmd == "incomeprofile":
        _, snap = _latest()
        reply = _income_profile_text(snap) if snap else "No daily snapshot available."
    elif cmd == "mtd":
        _, snap = _latest()
        reply = _mtd_text(snap) if snap else "No daily snapshot available."
    elif cmd == "received":
        _, snap = _latest()
        reply = _received_text(snap) if snap else "No daily snapshot available."
    elif cmd == "perf":
        _, snap = _latest()
        reply = _perf_text(snap) if snap else "No daily snapshot available."
    elif cmd == "holdings":
        _, snap = _latest()
        reply = _holdings_text(snap) if snap else "No daily snapshot available."
    elif cmd == "allocation":
        _, snap = _latest()
        reply = _allocation_text(snap) if snap else "No daily snapshot available."
    elif cmd == "benchmark":
        _, snap = _latest()
        reply = _benchmark_text(snap) if snap else "No daily snapshot available."
    elif cmd == "risk":
        _, snap = _latest()
        reply = _risk_text(snap) if snap else "No daily snapshot available."
    elif cmd == "riskdetail":
        _, snap = _latest()
        reply = _risk_detail_text(snap) if snap else "No daily snapshot available."
    elif cmd == "margin":
        _, snap = _latest()
        reply = _margin_text(snap) if snap else "No daily snapshot available."
    elif cmd == "rateshock":
        _, snap = _latest()
        reply = _rate_shock_text(snap) if snap else "No daily snapshot available."
    elif cmd == "goal":
        _, snap = _latest()
        reply = _goal_text(snap) if snap else "No daily snapshot available."
    elif cmd == "goalnet":
        _, snap = _latest()
        reply = _goal_net_text(snap) if snap else "No daily snapshot available."
    elif cmd == "goals":
        _, snap = _latest()
        if not snap:
            reply = "No daily snapshot available."
        else:
            goal_tiers = _goal_tiers(snap)
            if goal_tiers:
                reply = format_goal_tiers_html(goal_tiers)
            else:
                reply = "Goal tiers not available. Run a sync to generate tier data."
    elif cmd == "projection":
        _, snap = _latest()
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
        _, snap = _latest()
        reply = _snapshot_text(snap) if snap else "No daily snapshot available."
    elif cmd == "compare":
        as_of, snap = _latest()
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
        _, snap = _latest()
        reply = _macro_text(snap) if snap else "No daily snapshot available."
    elif cmd == "pace":
        _, snap = _latest()
        reply = _pace_text(snap) if snap else "No daily snapshot available."
    elif cmd == "pacewindows":
        _, snap = _latest()
        reply = _pace_windows_text(snap) if snap else "No daily snapshot available."
    elif cmd == "health":
        _, snap = _latest()
        reply = _health_text(conn, snap)
    elif cmd == "coverage":
        _, snap = _latest()
        reply = _coverage_text(snap) if snap else "No daily snapshot available."
    elif cmd == "position":
        if not args:
            reply = "Usage: /position &lt;symbol&gt; [days]\nExample: /position SCHD\nExample: /position SCHD 180"
        else:
            symbol = args[0].upper()
            if len(args) > 1 and args[1].isdigit():
                days_back = int(args[1])
                await tg.send_message_html(
                    _cached_position_history_view(conn, symbol, days_back),
                    reply_markup=_position_detail_markup(symbol, days_back),
                )
            else:
                _, snap = _latest()
                await tg.send_message_html(
                    _cached_position_hub_view(conn, snap, symbol),
                    reply_markup=_position_hub_markup(symbol),
                )
            return {"ok": True}
    elif cmd == "simulate":
        _, snap = _latest()
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
            _, snap = _latest()
            if chart_type not in {
                "pace", "income", "performance", "perf", "attribution", "yield",
                "risk", "drawdown", "dd", "allocation", "alloc", "margin",
                "dividends", "divs",
            }:
                await tg.send_message_html(
                    "Usage: /chart &lt;type&gt; [days]\n"
                    "Types: pace, income, performance, attribution,\n"
                    "yield, risk, drawdown, allocation, margin, dividends\n"
                    "Example: /chart performance 365"
                )
                return {"ok": True}
            img = _cached_chart_image(conn, chart_type, chart_days, snap=snap)
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
            as_of, html = _cached_period_report(conn, cmd)
            if html:
                period_snap = _latest_period_target_snapshot(conn, cmd, prefer_rolling=True)
                reply_markup = _period_report_markup(period_snap, cmd) if period_snap else None
                await tg.send_message_html(html, reply_markup=reply_markup)
                return {"ok": True}
            reply = f"No {cmd} report available."
        except Exception:
            reply = f"Error generating {cmd} report."
    elif cmd == "insights":
        if not settings.anthropic_api_key:
            reply = "ANTHROPIC_API_KEY not configured. Set it in your .env file."
        else:
            try:
                from ..services.ai_insights import generate_insight
                _, snap = _latest()
                if not snap:
                    reply = "No snapshot available."
                else:
                    use_deep = args and args[0].lower() == "deep"
                    model = "claude-opus-4-20250514" if use_deep else "claude-sonnet-4-20250514"
                    label = "Deep Analysis" if use_deep else "Quick Insight"
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
            await _send_period_ai_insight(tg, conn, period_kind, deep=use_deep)
            return {"ok": True}
    elif cmd == "morning":
        as_of, html = _cached_morning_brief(conn)
        reply = html if html else "No snapshot available."
    elif cmd == "evening":
        as_of, html = _cached_evening_recap(conn)
        reply = html if html else "No snapshot available."
    elif cmd == "digest":
        text, markup = _digest_keyboard_and_text(conn)
        await tg.send_message_html(text, reply_markup=markup)
        return {"ok": True}
    elif cmd == "whatif":
        _, snap = _latest()
        if not snap:
            reply = "No daily snapshot available."
        else:
            reply = _whatif_text(snap, args)
    elif cmd == "trend":
        category = args[0] if args else None
        reply = _trend_text(conn, category)
    elif cmd == "rebalance":
        _, snap = _latest()
        reply = _rebalance_text(snap) if snap else "No daily snapshot available."
    elif cmd == "about":
        _, snap = _latest()
        meta = snap.get("meta") if snap else {}
        reply = f"<b>Alert Bot</b>\nSchema: {meta.get('schema_version', '—')}\nAs of: {snap.get('as_of_date_local', '—') if snap else '—'}"
    else:
        # NLP: try matching plain text to a known command
        if not text.startswith("/"):
            matched_cmd = _nlp_match(text)
            if matched_cmd:
                # Dispatch to matched command by re-routing
                _, snap = _latest()
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

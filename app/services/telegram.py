from __future__ import annotations
import json
import re
import httpx
import structlog

log = structlog.get_logger()


def build_inline_keyboard(buttons: list[list[dict]]) -> dict:
    """
    Build Telegram inline keyboard markup.

    Args:
        buttons: 2D list of button dicts with keys:
            - text: Button display text
            - callback_data: Data sent when button pressed (max 64 bytes)

    Example:
        buttons = [
            [{"text": "✓ Ack", "callback_data": "ack:abc123"}],
            [{"text": "🔇 Silence 24h", "callback_data": "silence:24"}]
        ]
    """
    return {"inline_keyboard": buttons}


class TelegramClient:
    def __init__(self, bot_token: str, chat_id: str | int, timeout: float = 15.0):
        self.bot_token = bot_token
        self.chat_id = str(chat_id)
        self.base = f"https://api.telegram.org/bot{bot_token}"
        self.timeout = timeout

    def _sanitize_html(self, text_html: str) -> str:
        return re.sub(r"<br\s*/?>", "\n", text_html, flags=re.IGNORECASE)

    async def send_message_html(
        self,
        text_html: str,
        chat_id: str | int | None = None,
        disable_preview: bool = True,
        reply_markup: dict | None = None,
    ):
        url = f"{self.base}/sendMessage"
        target_chat_id = str(chat_id) if chat_id is not None else self.chat_id
        text_html = self._sanitize_html(text_html)
        data = {
            "chat_id": target_chat_id,
            "text": text_html,
            "parse_mode": "HTML",
            "disable_web_page_preview": disable_preview,
            "disable_notification": False,
        }
        if reply_markup:
            data["reply_markup"] = json.dumps(reply_markup)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(url, data=data)
            if r.status_code != 200:
                log.warning("telegram_send_failed", status=r.status_code, body=r.text[:500])
                return False
            return True

    def send_message_html_sync(
        self,
        text_html: str,
        chat_id: str | int | None = None,
        disable_preview: bool = True,
        reply_markup: dict | None = None,
    ):
        url = f"{self.base}/sendMessage"
        target_chat_id = str(chat_id) if chat_id is not None else self.chat_id
        text_html = self._sanitize_html(text_html)
        data = {
            "chat_id": target_chat_id,
            "text": text_html,
            "parse_mode": "HTML",
            "disable_web_page_preview": disable_preview,
            "disable_notification": False,
        }
        if reply_markup:
            data["reply_markup"] = json.dumps(reply_markup)
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(url, data=data)
            if r.status_code != 200:
                log.warning("telegram_send_failed", status=r.status_code, body=r.text[:500])
                return False
            return True

    async def answer_callback_query(
        self,
        callback_query_id: str,
        text: str | None = None,
        show_alert: bool = False,
    ):
        """Answer a callback query from inline keyboard button press."""
        url = f"{self.base}/answerCallbackQuery"
        data = {"callback_query_id": callback_query_id}
        if text:
            data["text"] = text
        data["show_alert"] = show_alert
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(url, data=data)
            if r.status_code != 200:
                log.warning("telegram_answer_callback_failed", status=r.status_code, body=r.text[:500])
                return False
            return True

    async def edit_message_reply_markup(
        self,
        chat_id: str | int,
        message_id: int,
        reply_markup: dict | None = None,
    ):
        """Edit the inline keyboard of an existing message."""
        url = f"{self.base}/editMessageReplyMarkup"
        data = {
            "chat_id": str(chat_id),
            "message_id": message_id,
        }
        if reply_markup:
            data["reply_markup"] = json.dumps(reply_markup)
        else:
            data["reply_markup"] = json.dumps({"inline_keyboard": []})
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(url, data=data)
            if r.status_code != 200:
                log.warning("telegram_edit_markup_failed", status=r.status_code, body=r.text[:500])
                return False
            return True

    async def edit_message_text(
        self,
        chat_id: str | int,
        message_id: int,
        text_html: str,
        reply_markup: dict | None = None,
        disable_preview: bool = True,
    ):
        """Edit the text of an existing message."""
        url = f"{self.base}/editMessageText"
        text_html = self._sanitize_html(text_html)
        data = {
            "chat_id": str(chat_id),
            "message_id": message_id,
            "text": text_html,
            "parse_mode": "HTML",
            "disable_web_page_preview": disable_preview,
        }
        if reply_markup:
            data["reply_markup"] = json.dumps(reply_markup)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(url, data=data)
            if r.status_code != 200:
                log.warning("telegram_edit_text_failed", status=r.status_code, body=r.text[:500])
                return False
            return True

    async def send_photo(
        self,
        photo_bytes: bytes,
        caption: str | None = None,
        chat_id: str | int | None = None,
    ):
        """Send a photo (PNG bytes) to Telegram."""
        url = f"{self.base}/sendPhoto"
        target_chat_id = str(chat_id) if chat_id is not None else self.chat_id
        data = {"chat_id": target_chat_id}
        if caption:
            data["caption"] = caption
            data["parse_mode"] = "HTML"
        files = {"photo": ("chart.png", photo_bytes, "image/png")}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(url, data=data, files=files)
            if r.status_code != 200:
                log.warning("telegram_send_photo_failed", status=r.status_code, body=r.text[:500])
                return False
            return True


def format_goal_tiers_html(goal_tiers: dict) -> str:
    """Format goal tiers data as HTML for telegram."""
    if not goal_tiers or not goal_tiers.get("tiers"):
        return "<b>No goal tiers data available</b>"

    current = goal_tiers.get("current_state", {})
    tiers = goal_tiers.get("tiers", [])

    lines = ["<b>📊 Dividend Goal Tier Analysis</b>\n"]

    # Current state
    lines.append("<b>Current State:</b>")
    lines.append(f"💰 Portfolio Value: ${current.get('portfolio_value', 0):,.2f}")
    lines.append(f"💵 Monthly Income: ${current.get('projected_monthly_income', 0):,.2f}")
    lines.append(f"📈 Portfolio Yield: {current.get('portfolio_yield_pct', 0):.2f}%")
    lines.append(f"🎯 Target Monthly: ${current.get('target_monthly', 0):,.2f}")
    if current.get('margin_loan_balance'):
        lines.append(f"🏦 Margin Loan: ${current.get('margin_loan_balance', 0):,.2f} ({current.get('current_ltv_pct', 0):.1f}% LTV)")
    lines.append("")

    # Tiers
    lines.append("<b>Goal Achievement Timelines:</b>\n")

    for tier in tiers:
        tier_num = tier.get("tier", 0)
        name = tier.get("name", "Unknown")
        months = tier.get("months_to_goal")
        goal_date = tier.get("estimated_goal_date", "N/A")
        assumptions = tier.get("assumptions", {})

        # Emoji for each tier
        emoji_map = {1: "🐌", 2: "🚶", 3: "🏃", 4: "🚀", 5: "🌟", 6: "⚡"}
        emoji = emoji_map.get(tier_num, "📌")

        lines.append(f"{emoji} <b>Tier {tier_num}: {name}</b>")

        if months is not None:
            years = months // 12
            remaining_months = months % 12
            if years > 0:
                time_str = f"{years}y {remaining_months}m" if remaining_months > 0 else f"{years}y"
            else:
                time_str = f"{remaining_months}m"
            lines.append(f"   ⏱ {time_str} ({goal_date})")
        else:
            lines.append("   ⏱ Goal not achievable with current assumptions")

        # Show key assumptions
        assumption_parts = []
        if assumptions.get("monthly_contribution", 0) > 0:
            assumption_parts.append(f"${assumptions['monthly_contribution']:.0f}/mo")
        if assumptions.get("drip_enabled"):
            assumption_parts.append("DRIP")
        if assumptions.get("annual_appreciation_pct", 0) > 0:
            assumption_parts.append(f"{assumptions['annual_appreciation_pct']:.0f}% growth")
        if assumptions.get("ltv_maintained"):
            assumption_parts.append(f"{assumptions.get('target_ltv_pct', 0):.0f}% LTV")

        if assumption_parts:
            lines.append(f"   📋 {' + '.join(assumption_parts)}")
        else:
            desc = (tier.get("description") or "").strip()
            lines.append(f"   📋 {desc or 'Hold only; no contributions or leverage'}")

        lines.append("")

    lines.append("💡 <i>Use Tier 3-4 for realistic planning, Tier 5-6 for maximum potential</i>")

    return "\n".join(lines)


def send_goal_tiers_to_telegram(goal_tiers: dict, telegram_client: 'TelegramClient') -> bool:
    """Format and send goal tiers to telegram."""
    html_message = format_goal_tiers_html(goal_tiers)
    return telegram_client.send_message_html_sync(html_message)


def _to_float(value):
    if isinstance(value, bool):
        return None
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coalesce(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _normalize_period_kind(kind: str | None) -> str:
    if not kind:
        return "period"
    raw = str(kind).strip().lower()
    upper = str(kind).strip().upper()
    if raw in {"weekly", "monthly", "quarterly", "yearly"}:
        return raw
    if upper in {"WEEK", "MONTH", "QUARTER", "YEAR"}:
        return {
            "WEEK": "weekly",
            "MONTH": "monthly",
            "QUARTER": "quarterly",
            "YEAR": "yearly",
        }[upper]
    if raw in {"week", "weeks"}:
        return "weekly"
    if raw in {"month", "months"}:
        return "monthly"
    if raw in {"quarter", "quarters"}:
        return "quarterly"
    if raw in {"year", "years"}:
        return "yearly"
    if "week" in raw:
        return "weekly"
    if "month" in raw:
        return "monthly"
    if "quarter" in raw:
        return "quarterly"
    if "year" in raw:
        return "yearly"
    return raw or "period"


def _period_identity(period_snap: dict) -> tuple[str, str, str, str]:
    period = period_snap.get("period") if isinstance(period_snap.get("period"), dict) else {}
    meta = period_snap.get("meta") if isinstance(period_snap.get("meta"), dict) else {}
    meta_period = meta.get("period") if isinstance(meta.get("period"), dict) else {}
    ts = period_snap.get("timestamps") if isinstance(period_snap.get("timestamps"), dict) else {}

    period_kind = _normalize_period_kind(
        _coalesce(
            meta_period.get("type"),
            period.get("type"),
            period.get("label"),
            "period",
        )
    )
    start_date = _coalesce(
        meta_period.get("start_date_local"),
        period.get("start_date"),
        period.get("start_date_local"),
        ts.get("period_start_local"),
        "",
    )
    end_date = _coalesce(
        meta_period.get("end_date_local"),
        period.get("end_date"),
        period.get("end_date_local"),
        ts.get("period_end_local"),
        "",
    )

    label = period.get("label")
    if not label:
        pretty = period_kind.replace("_", " ").title()
        if start_date and end_date:
            label = f"{pretty} ({start_date} to {end_date})"
        else:
            label = pretty

    return period_kind, str(start_date or ""), str(end_date or ""), str(label or "Period")


def _activity_block(period_snap: dict) -> dict:
    activity = period_snap.get("activity")
    if isinstance(activity, dict):
        return activity
    ps = period_snap.get("period_summary")
    if isinstance(ps, dict):
        maybe_activity = ps.get("activity")
        if isinstance(maybe_activity, dict):
            return maybe_activity
    return {}


def build_period_insight_keyboard(period_snap: dict) -> dict:
    """Build period insight action buttons for callback workflow."""
    period_kind, period_start, period_end, _ = _period_identity(period_snap)
    period_id = period_kind
    if period_start and period_end:
        period_id = f"{period_kind}:{period_start}:{period_end}"

    buttons = [
        [
            {"text": "📊 Holdings Changes", "callback_data": f"period_holdings:{period_id}"},
            {"text": "💱 Trades", "callback_data": f"period_trades:{period_id}"},
        ],
        [
            {"text": "💰 Activity Details", "callback_data": f"period_activity:{period_id}"},
            {"text": "⚠️ Risk Breakdown", "callback_data": f"period_risk:{period_id}"},
        ],
        [
            {"text": "🎯 Goals & Pace", "callback_data": f"period_goals:{period_id}"},
            {"text": "🏦 Margin", "callback_data": f"period_margin:{period_id}"},
        ],
        [
            {"text": "✓ Done", "callback_data": f"period_dismiss:{period_id}"},
        ],
    ]
    return build_inline_keyboard(buttons)


def format_period_holdings_html(period_snap: dict) -> str:
    """Format period holdings changes as HTML for telegram (target schema first)."""
    _, _, _, period_label = _period_identity(period_snap)
    lines = [f"<b>📊 Holdings Changes - {period_label}</b>\n"]

    # Target V5 shape: top-level holdings_summary[]
    holdings_summary = period_snap.get("holdings_summary")
    if isinstance(holdings_summary, list) and holdings_summary:
        normalized = []
        for row in holdings_summary:
            if not isinstance(row, dict):
                continue
            values = row.get("values") if isinstance(row.get("values"), dict) else {}
            performance = row.get("performance") if isinstance(row.get("performance"), dict) else {}
            start_weight = _to_float(values.get("start_weight_pct"))
            end_weight = _to_float(values.get("end_weight_pct"))
            weight_delta = (
                round(end_weight - start_weight, 3)
                if start_weight is not None and end_weight is not None
                else None
            )
            normalized.append(
                {
                    "symbol": str(row.get("symbol") or "?"),
                    "pnl_pct": _coalesce(
                        _to_float(performance.get("period_return_pct")),
                        _to_float(values.get("market_value_delta_pct")),
                    ),
                    "pnl_dollar": _to_float(values.get("market_value_delta")),
                    "weight_delta": weight_delta,
                }
            )

        with_pct = [r for r in normalized if r.get("pnl_pct") is not None]
        if with_pct:
            gainers = sorted(with_pct, key=lambda r: r["pnl_pct"], reverse=True)[:5]
            losers = sorted(with_pct, key=lambda r: r["pnl_pct"])[:5]

            lines.append("<b>Top Gainers:</b>")
            for row in gainers:
                pnl_dollar = row.get("pnl_dollar")
                pnl_dollar_txt = f" (${pnl_dollar:+,.0f})" if pnl_dollar is not None else ""
                lines.append(f"  {row['symbol']}: {row['pnl_pct']:+.1f}%{pnl_dollar_txt}")
            lines.append("")

            lines.append("<b>Top Losers:</b>")
            for row in losers:
                pnl_dollar = row.get("pnl_dollar")
                pnl_dollar_txt = f" (${pnl_dollar:+,.0f})" if pnl_dollar is not None else ""
                lines.append(f"  {row['symbol']}: {row['pnl_pct']:+.1f}%{pnl_dollar_txt}")
            lines.append("")

        increased = [r for r in normalized if isinstance(r.get("weight_delta"), (int, float)) and r["weight_delta"] > 0]
        if increased:
            lines.append("<b>Largest Weight Increases:</b>")
            for row in sorted(increased, key=lambda r: r["weight_delta"], reverse=True)[:3]:
                lines.append(f"  {row['symbol']}: {row['weight_delta']:+.2f}pp")
            lines.append("")

        decreased = [r for r in normalized if isinstance(r.get("weight_delta"), (int, float)) and r["weight_delta"] < 0]
        if decreased:
            lines.append("<b>Largest Weight Decreases:</b>")
            for row in sorted(decreased, key=lambda r: r["weight_delta"])[:3]:
                lines.append(f"  {row['symbol']}: {row['weight_delta']:+.2f}pp")

        return "\n".join(lines) if len(lines) > 1 else "<b>No holdings changes data available</b>"

    # Legacy fallback shape
    ps = period_snap.get("period_summary") if isinstance(period_snap.get("period_summary"), dict) else {}
    holdings_data = ps.get("holdings") if isinstance(ps.get("holdings"), dict) else {}

    gainers = holdings_data.get("top_gainers", [])[:5]
    if gainers:
        lines.append("<b>Top Gainers:</b>")
        for row in gainers:
            sym = row.get("symbol", "?")
            pnl_pct = _to_float(row.get("pnl_pct_period")) or 0.0
            pnl_dollar = _to_float(row.get("pnl_dollar_period")) or 0.0
            lines.append(f"  {sym}: {pnl_pct:+.1f}% (${pnl_dollar:+,.0f})")
        lines.append("")

    losers = holdings_data.get("top_losers", [])[:5]
    if losers:
        lines.append("<b>Top Losers:</b>")
        for row in losers:
            sym = row.get("symbol", "?")
            pnl_pct = _to_float(row.get("pnl_pct_period")) or 0.0
            pnl_dollar = _to_float(row.get("pnl_dollar_period")) or 0.0
            lines.append(f"  {sym}: {pnl_pct:+.1f}% (${pnl_dollar:+,.0f})")
        lines.append("")

    increased = holdings_data.get("weight_increased", [])[:3]
    if increased:
        lines.append("<b>Largest Weight Increases:</b>")
        for row in increased:
            sym = row.get("symbol", "?")
            w_delta = _to_float(row.get("weight_delta_pct")) or 0.0
            lines.append(f"  {sym}: +{w_delta:.2f}pp")
        lines.append("")

    decreased = holdings_data.get("weight_decreased", [])[:3]
    if decreased:
        lines.append("<b>Largest Weight Decreases:</b>")
        for row in decreased:
            sym = row.get("symbol", "?")
            w_delta = _to_float(row.get("weight_delta_pct")) or 0.0
            lines.append(f"  {sym}: {w_delta:.2f}pp")

    return "\n".join(lines) if len(lines) > 1 else "<b>No holdings changes data available</b>"


def format_period_trades_html(period_snap: dict) -> str:
    """Format period trades as HTML for telegram."""
    _, _, _, period_label = _period_identity(period_snap)
    activity = _activity_block(period_snap)
    trades = activity.get("trades") if isinstance(activity.get("trades"), dict) else {}

    lines = [f"<b>💱 Trades - {period_label}</b>\n"]

    total = int(trades.get("total_count") or 0)
    buys = int(trades.get("buy_count") or 0)
    sells = int(trades.get("sell_count") or 0)

    lines.append(f"Total Trades: {total} ({buys} buys, {sells} sells)")
    lines.append("")

    by_symbol_raw = trades.get("by_symbol")
    by_symbol = []
    if isinstance(by_symbol_raw, dict):
        for symbol, counts in by_symbol_raw.items():
            payload = counts if isinstance(counts, dict) else {}
            buy_ct = int(payload.get("buy_count") or 0)
            sell_ct = int(payload.get("sell_count") or 0)
            by_symbol.append(
                {
                    "symbol": symbol or "?",
                    "count": buy_ct + sell_ct,
                    "buy_count": buy_ct,
                    "sell_count": sell_ct,
                }
            )
    elif isinstance(by_symbol_raw, list):
        for row in by_symbol_raw:
            if not isinstance(row, dict):
                continue
            buy_ct = int(row.get("buy_count") or 0)
            sell_ct = int(row.get("sell_count") or 0)
            count = int(row.get("count") or (buy_ct + sell_ct))
            by_symbol.append(
                {
                    "symbol": row.get("symbol", "?"),
                    "count": count,
                    "buy_count": buy_ct,
                    "sell_count": sell_ct,
                }
            )

    if by_symbol:
        lines.append("<b>Most Active:</b>")
        for row in sorted(by_symbol, key=lambda x: x.get("count", 0), reverse=True)[:10]:
            lines.append(
                f"  {row.get('symbol', '?')}: {row.get('count', 0)} trades "
                f"({row.get('buy_count', 0)}B/{row.get('sell_count', 0)}S)"
            )

    return "\n".join(lines) if len(lines) > 1 else "<b>No trades data available</b>"


def format_period_activity_html(period_snap: dict) -> str:
    """Format period activity (contributions, withdrawals, dividends) as HTML for telegram."""
    _, _, _, period_label = _period_identity(period_snap)
    activity = _activity_block(period_snap)

    lines = [f"<b>💰 Activity Details - {period_label}</b>\n"]

    contrib = activity.get("contributions") if isinstance(activity.get("contributions"), dict) else {}
    contrib_total = abs(_to_float(contrib.get("total")) or 0.0)
    contrib_count = int(contrib.get("count") or 0)
    if contrib_total > 0:
        lines.append(f"<b>Contributions:</b> ${contrib_total:,.2f} ({contrib_count} deposits)")

    withdraw = activity.get("withdrawals") if isinstance(activity.get("withdrawals"), dict) else {}
    withdraw_total = abs(_to_float(withdraw.get("total")) or 0.0)
    withdraw_count = int(withdraw.get("count") or 0)
    if withdraw_total > 0:
        lines.append(f"<b>Withdrawals:</b> ${withdraw_total:,.2f} ({withdraw_count} withdrawals)")

    divs = activity.get("dividends") if isinstance(activity.get("dividends"), dict) else {}
    div_total = abs(_to_float(divs.get("total_received")) or 0.0)
    div_count = int(divs.get("count") or 0)
    if div_total > 0:
        lines.append(f"<b>Dividends:</b> ${div_total:,.2f} ({div_count} events)")
        lines.append("")

        events = divs.get("events") if isinstance(divs.get("events"), list) else []
        if events:
            lines.append("<b>Top Dividend Payers:</b>")
            sorted_divs = sorted(
                [event for event in events if isinstance(event, dict)],
                key=lambda x: abs(_to_float(x.get("amount")) or 0.0),
                reverse=True,
            )[:10]
            for event in sorted_divs:
                sym = event.get("symbol", "?")
                amt = abs(_to_float(event.get("amount")) or 0.0)
                lines.append(f"  {sym}: ${amt:.2f}")

    interest = activity.get("interest") if isinstance(activity.get("interest"), dict) else {}
    interest_total = abs(_to_float(interest.get("total_paid")) or 0.0)
    if interest_total > 0:
        lines.append("")
        lines.append(f"<b>Margin Interest:</b> ${interest_total:,.2f}")
        avg_rate = _to_float(interest.get("avg_rate_pct"))
        annualized = _to_float(interest.get("annualized"))
        if avg_rate is not None:
            lines.append(f"  Avg APR: {avg_rate:.2f}%")
        if annualized is not None:
            lines.append(f"  Annualized: ${annualized:,.2f}")

    margin = activity.get("margin") if isinstance(activity.get("margin"), dict) else {}
    borrowed = _coalesce(_to_float(margin.get("borrowed")), _to_float(margin.get("net_borrow")), 0.0)
    repaid = _coalesce(_to_float(margin.get("repaid")), _to_float(margin.get("net_repay")), 0.0)
    net_change = _to_float(margin.get("net_change"))
    if net_change is None:
        net_change = float(borrowed or 0.0) - float(repaid or 0.0)

    if abs(float(borrowed or 0.0)) > 1e-9 or abs(float(repaid or 0.0)) > 1e-9 or abs(float(net_change or 0.0)) > 1e-9:
        lines.append("")
        lines.append("<b>Margin Activity:</b>")
        if float(borrowed or 0.0) > 0:
            lines.append(f"  Borrowed: ${float(borrowed):,.2f}")
        if float(repaid or 0.0) > 0:
            lines.append(f"  Repaid: ${float(repaid):,.2f}")
        lines.append(f"  Net Change: ${float(net_change):+,.2f}")

    return "\n".join(lines) if len(lines) > 1 else "<b>No activity data available</b>"


def format_period_risk_html(period_snap: dict) -> str:
    """Format period risk metrics as HTML for telegram."""
    _, _, _, period_label = _period_identity(period_snap)
    lines = [f"<b>⚠️ Risk Breakdown - {period_label}</b>\n"]

    portfolio = period_snap.get("portfolio") if isinstance(period_snap.get("portfolio"), dict) else {}
    risk_target = portfolio.get("risk") if isinstance(portfolio.get("risk"), dict) else {}
    ps = period_snap.get("period_summary") if isinstance(period_snap.get("period_summary"), dict) else {}

    drawdown = (
        risk_target.get("drawdown")
        if isinstance(risk_target.get("drawdown"), dict)
        else ps.get("period_drawdown")
        if isinstance(ps.get("period_drawdown"), dict)
        else {}
    )
    max_dd = _to_float(drawdown.get("period_max_drawdown_pct"))
    max_dd_date = drawdown.get("period_max_drawdown_date") or "N/A"
    if max_dd is not None:
        lines.append("<b>Drawdown:</b>")
        lines.append(f"  Max: {max_dd:.2f}% on {max_dd_date}")
        lines.append("")

    var_metrics = (
        risk_target.get("var")
        if isinstance(risk_target.get("var"), dict)
        else ps.get("var_breach")
        if isinstance(ps.get("var_breach"), dict)
        else ps.get("period_var")
        if isinstance(ps.get("period_var"), dict)
        else {}
    )
    days_exceeding = _to_float(var_metrics.get("days_exceeding_var_95"))
    worst_day = _to_float(var_metrics.get("worst_day_return_pct"))
    best_day = _to_float(var_metrics.get("best_day_return_pct"))
    if days_exceeding is not None or worst_day is not None or best_day is not None:
        lines.append("<b>Volatility Events:</b>")
        lines.append(f"  Days exceeding VaR 95: {int(days_exceeding or 0)}")
        if worst_day is not None:
            lines.append(f"  Worst day: {worst_day:.2f}% on {var_metrics.get('worst_day_date', 'N/A')}")
        if best_day is not None:
            lines.append(f"  Best day: {best_day:+.2f}% on {var_metrics.get('best_day_date', 'N/A')}")
        lines.append("")

    volatility = risk_target.get("volatility") if isinstance(risk_target.get("volatility"), dict) else {}
    ratios = risk_target.get("ratios") if isinstance(risk_target.get("ratios"), dict) else {}
    legacy_risk = ps.get("risk") if isinstance(ps.get("risk"), dict) else {}
    legacy_start = legacy_risk.get("start") if isinstance(legacy_risk.get("start"), dict) else {}
    legacy_end = legacy_risk.get("end") if isinstance(legacy_risk.get("end"), dict) else {}

    vol_start = _coalesce(_to_float(volatility.get("start_vol_30d_pct")), _to_float(legacy_start.get("vol_30d_pct")))
    vol_end = _coalesce(_to_float(volatility.get("end_vol_30d_pct")), _to_float(legacy_end.get("vol_30d_pct")))
    sharpe_start = _coalesce(_to_float(ratios.get("start_sharpe_1y")), _to_float(legacy_start.get("sharpe_1y")))
    sharpe_end = _coalesce(_to_float(ratios.get("end_sharpe_1y")), _to_float(legacy_end.get("sharpe_1y")))
    sortino_start = _coalesce(_to_float(ratios.get("start_sortino_1y")), _to_float(legacy_start.get("sortino_1y")))
    sortino_end = _coalesce(_to_float(ratios.get("end_sortino_1y")), _to_float(legacy_end.get("sortino_1y")))

    if vol_start is not None or vol_end is not None or sharpe_start is not None or sharpe_end is not None:
        lines.append("<b>Risk Metrics (Start → End):</b>")
        if vol_start is not None and vol_end is not None:
            lines.append(f"  Volatility: {vol_start:.1f}% → {vol_end:.1f}%")
        if sharpe_start is not None and sharpe_end is not None:
            lines.append(f"  Sharpe: {sharpe_start:.2f} → {sharpe_end:.2f}")
        if sortino_start is not None and sortino_end is not None:
            lines.append(f"  Sortino: {sortino_start:.2f} → {sortino_end:.2f}")

    return "\n".join(lines) if len(lines) > 1 else "<b>No risk data available</b>"


def format_period_goals_html(period_snap: dict) -> str:
    """Format period goal progress and pace as HTML for telegram."""
    _, _, _, period_label = _period_identity(period_snap)
    goals = period_snap.get("goals") if isinstance(period_snap.get("goals"), dict) else {}
    start = goals.get("start") if isinstance(goals.get("start"), dict) else {}
    end = goals.get("end") if isinstance(goals.get("end"), dict) else {}
    delta = goals.get("delta") if isinstance(goals.get("delta"), dict) else {}
    pace = goals.get("pace") if isinstance(goals.get("pace"), dict) else {}

    def _fmt_months(value) -> str:
        num = _to_float(value)
        if num is None:
            return "—"
        if abs(num - round(num)) < 1e-9:
            return str(int(round(num)))
        return f"{num:.1f}"

    lines = [f"<b>🎯 Goals & Pace - {period_label}</b>\n"]

    target_monthly = _coalesce(_to_float(end.get("target_monthly")), _to_float(start.get("target_monthly")))
    if target_monthly is not None:
        lines.append(f"Target Monthly: ${target_monthly:,.2f}")

    start_income = _to_float(start.get("projected_monthly_income"))
    end_income = _to_float(end.get("projected_monthly_income"))
    delta_income = _coalesce(_to_float(delta.get("projected_monthly_income")), None)
    if start_income is not None or end_income is not None:
        lines.append("<b>Income Progress:</b>")
        if start_income is not None and end_income is not None:
            income_delta_txt = f" ({end_income - start_income:+,.2f})"
            if delta_income is not None:
                income_delta_txt = f" ({delta_income:+,.2f})"
            lines.append(f"  Monthly Income: ${start_income:,.2f} → ${end_income:,.2f}{income_delta_txt}")
        elif end_income is not None:
            lines.append(f"  Monthly Income: ${end_income:,.2f}")

    start_progress = _to_float(start.get("progress_pct"))
    end_progress = _to_float(end.get("progress_pct"))
    delta_progress = _coalesce(_to_float(delta.get("progress_pct")), None)
    if start_progress is not None or end_progress is not None:
        if not any(line == "<b>Income Progress:</b>" for line in lines):
            lines.append("<b>Income Progress:</b>")
        if start_progress is not None and end_progress is not None:
            progress_delta_txt = f" ({end_progress - start_progress:+.2f}pp)"
            if delta_progress is not None:
                progress_delta_txt = f" ({delta_progress:+.2f}pp)"
            lines.append(f"  Goal Progress: {start_progress:.2f}% → {end_progress:.2f}%{progress_delta_txt}")
        elif end_progress is not None:
            lines.append(f"  Goal Progress: {end_progress:.2f}%")

    start_months = _to_float(start.get("months_to_goal"))
    end_months = _to_float(end.get("months_to_goal"))
    delta_months = _coalesce(_to_float(delta.get("months_to_goal")), None)
    if start_months is not None or end_months is not None:
        if start_months is not None and end_months is not None:
            months_delta_txt = f" ({end_months - start_months:+.1f})"
            if delta_months is not None:
                months_delta_txt = f" ({delta_months:+.1f})"
            lines.append(
                f"  Months To Goal: {_fmt_months(start_months)} → {_fmt_months(end_months)}{months_delta_txt}"
            )
        elif end_months is not None:
            lines.append(f"  Months To Goal: {_fmt_months(end_months)}")

    if end.get("estimated_goal_date") or start.get("estimated_goal_date"):
        lines.append(
            f"  Goal Date: {start.get('estimated_goal_date') or '—'} → {end.get('estimated_goal_date') or '—'}"
        )

    pace_start = _to_float(pace.get("start_months_ahead_behind"))
    pace_end = _to_float(pace.get("end_months_ahead_behind"))
    pace_delta = _coalesce(_to_float(pace.get("delta_months_ahead_behind")), None)
    tier_start = _to_float(pace.get("start_tier_pace_pct"))
    tier_end = _to_float(pace.get("end_tier_pace_pct"))
    if pace_start is not None or pace_end is not None or tier_start is not None or tier_end is not None:
        lines.append("")
        lines.append("<b>Pace:</b>")
        if pace_start is not None and pace_end is not None:
            pace_delta_txt = f" ({pace_end - pace_start:+.2f})"
            if pace_delta is not None:
                pace_delta_txt = f" ({pace_delta:+.2f})"
            lines.append(
                f"  Months Ahead/Behind: {_fmt_months(pace_start)} → {_fmt_months(pace_end)}{pace_delta_txt}"
            )
        elif pace_end is not None:
            lines.append(f"  Months Ahead/Behind: {_fmt_months(pace_end)}")
        if tier_start is not None and tier_end is not None:
            lines.append(f"  Tier Pace: {tier_start:.2f}% → {tier_end:.2f}%")
        elif tier_end is not None:
            lines.append(f"  Tier Pace: {tier_end:.2f}%")

    return "\n".join(lines) if len(lines) > 1 else "<b>No goal data available</b>"


def format_period_margin_html(period_snap: dict) -> str:
    """Format period margin balance, cost, and safety as HTML for telegram."""
    _, _, _, period_label = _period_identity(period_snap)
    margin = period_snap.get("margin") if isinstance(period_snap.get("margin"), dict) else {}
    balance = margin.get("balance") if isinstance(margin.get("balance"), dict) else {}
    ltv = margin.get("ltv") if isinstance(margin.get("ltv"), dict) else {}
    interest = margin.get("interest") if isinstance(margin.get("interest"), dict) else {}
    safety = margin.get("safety") if isinstance(margin.get("safety"), dict) else {}

    lines = [f"<b>🏦 Margin - {period_label}</b>\n"]

    start_balance = _to_float(balance.get("start"))
    end_balance = _to_float(balance.get("end"))
    delta_balance = _coalesce(_to_float(balance.get("delta")), None)
    avg_balance = _to_float(balance.get("avg"))
    if start_balance is not None or end_balance is not None or avg_balance is not None:
        lines.append("<b>Balance:</b>")
        if start_balance is not None and end_balance is not None:
            delta_txt = f" ({end_balance - start_balance:+,.2f})"
            if delta_balance is not None:
                delta_txt = f" ({delta_balance:+,.2f})"
            lines.append(f"  Start → End: ${start_balance:,.2f} → ${end_balance:,.2f}{delta_txt}")
        elif end_balance is not None:
            lines.append(f"  End: ${end_balance:,.2f}")
        if avg_balance is not None:
            lines.append(f"  Avg Balance: ${avg_balance:,.2f}")
        min_balance = _to_float(balance.get("min"))
        max_balance = _to_float(balance.get("max"))
        if min_balance is not None and max_balance is not None:
            lines.append(f"  Range: ${min_balance:,.2f} → ${max_balance:,.2f}")

    start_ltv = _to_float(ltv.get("start_pct"))
    end_ltv = _to_float(ltv.get("end_pct"))
    avg_ltv = _to_float(ltv.get("avg_pct"))
    if start_ltv is not None or end_ltv is not None or avg_ltv is not None:
        lines.append("")
        lines.append("<b>LTV:</b>")
        if start_ltv is not None and end_ltv is not None:
            lines.append(f"  Start → End: {start_ltv:.2f}% → {end_ltv:.2f}%")
        elif end_ltv is not None:
            lines.append(f"  End: {end_ltv:.2f}%")
        if avg_ltv is not None:
            lines.append(f"  Avg LTV: {avg_ltv:.2f}%")
        min_ltv = _to_float(ltv.get("min_pct"))
        max_ltv = _to_float(ltv.get("max_pct"))
        if min_ltv is not None and max_ltv is not None:
            lines.append(f"  Range: {min_ltv:.2f}% → {max_ltv:.2f}%")

    total_paid = _to_float(interest.get("total_paid"))
    avg_daily_cost = _to_float(interest.get("avg_daily_cost"))
    avg_apr = _to_float(interest.get("avg_apr_pct"))
    if total_paid is not None or avg_daily_cost is not None or avg_apr is not None:
        lines.append("")
        lines.append("<b>Interest:</b>")
        if total_paid is not None:
            lines.append(f"  Total Paid: ${total_paid:,.2f}")
        if avg_daily_cost is not None:
            lines.append(f"  Avg Daily Cost: ${avg_daily_cost:,.2f}")
        if avg_apr is not None:
            lines.append(f"  Avg APR: {avg_apr:.2f}%")

    min_buffer = _to_float(safety.get("min_buffer_to_call_pct"))
    if min_buffer is not None:
        lines.append("")
        lines.append("<b>Safety:</b>")
        lines.append(f"  Minimum Buffer: {min_buffer:.2f}% on {safety.get('min_buffer_date') or '—'}")
        below_50 = _to_float(safety.get("days_below_50pct_buffer"))
        below_40 = _to_float(safety.get("days_below_40pct_buffer"))
        margin_call_events = _to_float(safety.get("margin_call_events"))
        if below_50 is not None:
            lines.append(f"  Days Below 50% Buffer: {int(below_50)}")
        if below_40 is not None:
            lines.append(f"  Days Below 40% Buffer: {int(below_40)}")
        if margin_call_events is not None:
            lines.append(f"  Margin Call Events: {int(margin_call_events)}")

    return "\n".join(lines) if len(lines) > 1 else "<b>No margin data available</b>"


def send_period_insight_to_telegram(
    period_snap: dict,
    telegram_client: 'TelegramClient',
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
    title_prefix: str | None = None,
) -> bool:
    """
    Generate AI insight for a period snapshot and send to Telegram with interactive menu buttons.

    Args:
        period_snap: Full period snapshot dict (from assemble_period_snapshot)
        telegram_client: TelegramClient instance
        api_key: Anthropic API key for insight generation

    Returns:
        True if message sent successfully, False otherwise
    """
    try:
        from app.services.ai_insights import generate_period_insight

        # Generate HTML-formatted insight using Claude API
        insight_html = generate_period_insight(period_snap, api_key, model=model)

        if not insight_html:
            log.warning("period_insight_generation_failed", reason="no_insight_returned")
            return False

        if title_prefix:
            insight_html = f"{title_prefix}\n\n{insight_html}"
        reply_markup = build_period_insight_keyboard(period_snap)

        # Send to Telegram with buttons
        success = telegram_client.send_message_html_sync(insight_html, reply_markup=reply_markup)

        if success:
            log.info("period_insight_sent_to_telegram", period=period_snap.get("period", {}))
        else:
            log.warning("period_insight_send_failed")

        return success

    except ImportError:
        log.error("period_insight_missing_ai_insights_module")
        return False
    except Exception as e:
        log.error("period_insight_error", error=str(e))
        return False

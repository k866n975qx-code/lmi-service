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

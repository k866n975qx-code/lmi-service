from __future__ import annotations
import re
import httpx
import structlog

log = structlog.get_logger()

class TelegramClient:
    def __init__(self, bot_token: str, chat_id: str | int, timeout: float = 15.0):
        self.bot_token = bot_token
        self.chat_id = str(chat_id)
        self.base = f"https://api.telegram.org/bot{bot_token}"
        self.timeout = timeout

    def _sanitize_html(self, text_html: str) -> str:
        return re.sub(r"<br\s*/?>", "\n", text_html, flags=re.IGNORECASE)

    async def send_message_html(self, text_html: str, chat_id: str | int | None = None, disable_preview: bool = True):
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
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(url, data=data)
            if r.status_code != 200:
                log.warning("telegram_send_failed", status=r.status_code, body=r.text[:500])
                return False
            return True

    def send_message_html_sync(self, text_html: str, chat_id: str | int | None = None, disable_preview: bool = True):
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
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(url, data=data)
            if r.status_code != 200:
                log.warning("telegram_send_failed", status=r.status_code, body=r.text[:500])
                return False
            return True

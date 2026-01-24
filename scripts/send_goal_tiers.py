#!/usr/bin/env python3
"""
Send dividend goal tiers to Telegram.

This script retrieves the latest goal tier analysis from the snapshot
and sends it to the configured Telegram chat.
"""
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.db import get_conn
from app.services.telegram import TelegramClient, send_goal_tiers_to_telegram


def main():
    # Check telegram configuration
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        print("ERROR: Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.")
        return 1

    # Get latest snapshot
    conn = get_conn(settings.db_path)
    cur = conn.cursor()
    row = cur.execute(
        "SELECT payload_json FROM snapshot_daily_current ORDER BY as_of_date_local DESC LIMIT 1"
    ).fetchone()

    if not row:
        print("ERROR: No daily snapshot found. Run sync_all.py first.")
        return 1

    payload = json.loads(row[0])
    goal_tiers = payload.get("goal_tiers")

    if not goal_tiers:
        print("ERROR: Goal tiers not available in snapshot. The snapshot may have been created before this feature was added.")
        print("Run sync_all.py to regenerate the snapshot with goal tiers.")
        return 1

    # Send to telegram
    print("Sending goal tiers to Telegram...")
    telegram_client = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)
    success = send_goal_tiers_to_telegram(goal_tiers, telegram_client)

    if success:
        print("✓ Goal tiers sent successfully to Telegram!")
        return 0
    else:
        print("ERROR: Failed to send message to Telegram.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

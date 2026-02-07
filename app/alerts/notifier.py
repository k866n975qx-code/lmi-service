from __future__ import annotations

from datetime import datetime, timedelta, timezone
import structlog

from collections import defaultdict

from .constants import (
    ALERT_GROUPING_ENABLED,
    ALERT_GROUPING_MIN_SIZE,
    ESCALATION_REMINDER_HOURS,
    MAX_NOTIFICATIONS_PER_DAY,
    MIN_HOURS_BETWEEN_SAME_ALERT,
)
from .storage import (
    count_notifications_since,
    create_alert,
    get_alert_by_id,
    get_open_alert_by_fingerprint,
    mark_alert_notified,
    now_utc_iso,
    set_setting,
    get_setting,
    update_alert_on_trigger,
)

log = structlog.get_logger()

def _hours_since(value: str | None) -> float | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0

def _silenced_until(conn) -> datetime | None:
    raw = get_setting(conn, "silenced_until_utc", None)
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None

def _min_severity(conn) -> int:
    raw = get_setting(conn, "min_severity", None)
    try:
        return int(raw) if raw is not None else 1
    except ValueError:
        return 1

def set_silence(conn, hours: int):
    until = datetime.now(timezone.utc) + timedelta(hours=hours)
    set_setting(conn, "silenced_until_utc", until.isoformat())

def clear_silence(conn):
    set_setting(conn, "silenced_until_utc", "")

def set_min_severity(conn, severity: int):
    set_setting(conn, "min_severity", str(int(severity)))

def _daily_limit_reached(conn, min_severity: int, severity: int) -> bool:
    if severity >= 8:
        return False
    if severity < min_severity:
        return True
    since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    sent = count_notifications_since(conn, since)
    return sent >= MAX_NOTIFICATIONS_PER_DAY

def _compute_next_reminder(base_utc: str, reminder_index: int) -> str | None:
    if reminder_index >= len(ESCALATION_REMINDER_HOURS):
        return None
    hours = ESCALATION_REMINDER_HOURS[reminder_index]
    try:
        dt = datetime.fromisoformat(base_utc)
    except ValueError:
        dt = datetime.now(timezone.utc)
    return (dt + timedelta(hours=hours)).isoformat()

def upsert_alerts(conn, alerts: list[dict]) -> list[dict]:
    """
    Deduplicate alerts by fingerprint and decide whether to notify.
    Returns list of dicts with alert and notify decision.
    """
    results = []
    now = now_utc_iso()
    for alert in alerts:
        existing = get_open_alert_by_fingerprint(conn, alert["fingerprint"])
        if existing:
            severity_increased = int(alert["severity"]) > int(existing["severity"])
            last_notified = existing.get("last_notified_at_utc")
            hours_since = _hours_since(last_notified)
            should_notify = False
            if severity_increased:
                should_notify = True
            elif hours_since is None or hours_since >= MIN_HOURS_BETWEEN_SAME_ALERT:
                should_notify = True
            update_alert_on_trigger(
                conn,
                existing["id"],
                alert["severity"],
                alert["title"],
                alert["body_html"],
                alert["as_of_date_local"],
                now,
                details=alert.get("details"),
            )
            results.append(
                {
                    "alert_id": existing["id"],
                    "alert": alert,
                    "should_notify": should_notify,
                    "severity_increased": severity_increased,
                }
            )
        else:
            existing_any = get_alert_by_id(conn, alert["id"])
            if existing_any:
                update_alert_on_trigger(
                    conn,
                    existing_any["id"],
                    alert["severity"],
                    alert["title"],
                    alert["body_html"],
                    alert["as_of_date_local"],
                    now,
                    details=alert.get("details"),
                )
                results.append(
                    {
                        "alert_id": existing_any["id"],
                        "alert": alert,
                        "should_notify": False,
                        "severity_increased": False,
                    }
                )
            else:
                create_alert(conn, alert, now)
                results.append(
                    {
                        "alert_id": alert["id"],
                        "alert": alert,
                        "should_notify": True,
                        "severity_increased": True,
                    }
                )
    return results

def _iter_immediate_to_send(conn, results: list[dict]):
    min_sev = _min_severity(conn)
    silenced_until = _silenced_until(conn)
    now_dt = datetime.now(timezone.utc)
    for item in results:
        alert = item["alert"]
        severity = int(alert["severity"])
        if not item["should_notify"]:
            continue
        if severity < min_sev:
            continue
        if silenced_until and now_dt < silenced_until and severity < 8:
            continue
        if _daily_limit_reached(conn, min_sev, severity):
            continue
        yield item


def _severity_emoji(severity: int) -> str:
    if severity >= 8:
        return "\U0001f534"
    if severity >= 5:
        return "\U0001f7e1"
    return "\U0001f7e2"


def _group_alerts_by_category(items: list[dict]) -> list[dict]:
    """Group same-category alerts into combined entries.

    Returns a mixed list of:
    - Original item dicts (unchanged) for categories with < MIN_SIZE alerts
    - Group dicts with is_group=True for categories with >= MIN_SIZE alerts
    """
    if not ALERT_GROUPING_ENABLED or not items:
        return items

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        cat = item["alert"].get("category", "unknown")
        by_cat[cat].append(item)

    result: list[dict] = []
    for cat, cat_items in by_cat.items():
        if len(cat_items) < ALERT_GROUPING_MIN_SIZE:
            result.extend(cat_items)
        else:
            max_sev = max(int(it["alert"]["severity"]) for it in cat_items)
            count = len(cat_items)
            lines = [f"\U0001f514 <b>{cat.upper()}</b> \u2014 {count} alerts", ""]
            for it in cat_items[:10]:
                a = it["alert"]
                emoji = _severity_emoji(int(a["severity"]))
                lines.append(f"{emoji} [{a['severity']}] {a['title']}")
                # Extract key detail from body_html (after title line)
                body_lines = a.get("body_html", "").replace("<br/>", "\n").split("\n")
                # Skip the first line (title/emoji part), include remaining lines as details
                if len(body_lines) > 1:
                    for detail in body_lines[1:4]:  # Include up to 3 detail lines
                        detail = detail.strip()
                        if detail and not detail.startswith(emoji):
                            lines.append(f"  • {detail}")
            if count > 10:
                lines.append(f"...+{count - 10} more")
            lines.append("")
            lines.append(f"Max severity: {max_sev} | /alerts for details")
            result.append({
                "is_group": True,
                "category": cat,
                "items": cat_items,
                "max_severity": max_sev,
                "body_html": "\n".join(lines),
            })
    return result


async def send_alerts(conn, alerts: list[dict], telegram_client, channel: str = "telegram", with_buttons: bool = True):
    from ..services.telegram import build_inline_keyboard
    results = upsert_alerts(conn, alerts)
    immediate = list(_iter_immediate_to_send(conn, results))
    grouped = _group_alerts_by_category(immediate)

    for entry in grouped:
        if entry.get("is_group"):
            first_alert_id = entry["items"][0]["alert_id"]
            max_sev = entry["max_severity"]
            reply_markup = None
            if with_buttons and max_sev >= 6:
                reply_markup = build_inline_keyboard([
                    [
                        {"text": f"✓ Ack All ({len(entry['items'])})", "callback_data": f"ackcat:{entry['category'][:28]}"},
                        {"text": "📊 Details", "callback_data": f"details:{first_alert_id[:32]}"},
                    ],
                    [
                        {"text": "🔇 Silence 1h", "callback_data": "silence:1"},
                        {"text": "🔇 Silence 24h", "callback_data": "silence:24"},
                    ],
                ])
            ok = await telegram_client.send_message_html(entry["body_html"], reply_markup=reply_markup)
            for item in entry["items"]:
                sev = int(item["alert"]["severity"])
                next_reminder = _compute_next_reminder(now_utc_iso(), 0) if sev >= 7 else None
                mark_alert_notified(
                    conn, item["alert_id"], channel, ok,
                    None if ok else "send_failed", now_utc_iso(),
                    next_reminder_at_utc=next_reminder, increment_reminder=False,
                )
            log.info("alert_group_sent", category=entry["category"],
                     count=len(entry["items"]), max_severity=max_sev, ok=ok)
        else:
            alert = entry["alert"]
            alert_id = entry["alert_id"]
            severity = int(alert["severity"])
            reply_markup = None
            if with_buttons and severity >= 6:
                reply_markup = build_inline_keyboard([
                    [
                        {"text": "✓ Acknowledge", "callback_data": f"ack:{alert_id[:32]}"},
                        {"text": "📊 Details", "callback_data": f"details:{alert_id[:32]}"},
                    ],
                    [
                        {"text": "🔇 Silence 1h", "callback_data": "silence:1"},
                        {"text": "🔇 Silence 24h", "callback_data": "silence:24"},
                    ],
                ])
            ok = await telegram_client.send_message_html(alert["body_html"], reply_markup=reply_markup)
            next_reminder = _compute_next_reminder(now_utc_iso(), 0) if severity >= 7 else None
            mark_alert_notified(
                conn, alert_id, channel, ok,
                None if ok else "send_failed", now_utc_iso(),
                next_reminder_at_utc=next_reminder, increment_reminder=False,
            )
            log.info("alert_sent", alert_id=alert_id, severity=severity, ok=ok,
                     with_buttons=with_buttons and severity >= 6)

def send_alerts_sync(conn, alerts: list[dict], telegram_client, channel: str = "telegram"):
    results = upsert_alerts(conn, alerts)
    immediate = list(_iter_immediate_to_send(conn, results))
    grouped = _group_alerts_by_category(immediate)

    for entry in grouped:
        if entry.get("is_group"):
            ok = telegram_client.send_message_html_sync(entry["body_html"])
            for item in entry["items"]:
                sev = int(item["alert"]["severity"])
                next_reminder = _compute_next_reminder(now_utc_iso(), 0) if sev >= 7 else None
                mark_alert_notified(
                    conn, item["alert_id"], channel, ok,
                    None if ok else "send_failed", now_utc_iso(),
                    next_reminder_at_utc=next_reminder, increment_reminder=False,
                )
            log.info("alert_group_sent", category=entry["category"],
                     count=len(entry["items"]), max_severity=entry["max_severity"], ok=ok)
        else:
            alert = entry["alert"]
            severity = int(alert["severity"])
            ok = telegram_client.send_message_html_sync(alert["body_html"])
            next_reminder = _compute_next_reminder(now_utc_iso(), 0) if severity >= 7 else None
            mark_alert_notified(
                conn, entry["alert_id"], channel, ok,
                None if ok else "send_failed", now_utc_iso(),
                next_reminder_at_utc=next_reminder, increment_reminder=False,
            )
            log.info("alert_sent", alert_id=entry["alert_id"], severity=severity, ok=ok)

async def send_digest(conn, alerts: list[dict], html: str, telegram_client, severity_hint: int, channel: str = "telegram"):
    min_sev = _min_severity(conn)
    silenced_until = _silenced_until(conn)
    now_dt = datetime.now(timezone.utc)
    if severity_hint < min_sev:
        return False
    if silenced_until and now_dt < silenced_until and severity_hint < 8:
        return False
    if _daily_limit_reached(conn, min_sev, severity_hint):
        return False
    ok = await telegram_client.send_message_html(html)
    mark_alert_notified(conn, None, channel, ok, None if ok else "send_failed", now_utc_iso())
    for alert in alerts:
        next_reminder = _compute_next_reminder(now_utc_iso(), 0) if alert["severity"] >= 7 else None
        mark_alert_notified(
            conn,
            alert["id"],
            channel,
            ok,
            None if ok else "send_failed",
            now_utc_iso(),
            next_reminder_at_utc=next_reminder,
            increment_reminder=False,
        )
    return ok

async def send_due_reminders(conn, telegram_client, channel: str = "telegram"):
    from .storage import list_due_reminders
    from .constants import ESCALATION_REMINDER_HOURS

    due = list_due_reminders(conn, now_utc_iso(), min_severity=7)
    if not due:
        return 0
    min_sev = _min_severity(conn)
    silenced_until = _silenced_until(conn)
    now_dt = datetime.now(timezone.utc)
    sent = 0
    for alert in due:
        reminder_count = int(alert.get("reminder_count") or 0)
        if reminder_count >= len(ESCALATION_REMINDER_HOURS):
            continue
        severity = int(alert.get("severity") or 0)
        if severity < min_sev:
            continue
        if silenced_until and now_dt < silenced_until and severity < 8:
            continue
        if _daily_limit_reached(conn, min_sev, severity):
            continue
        prefix = "🔔 Reminder" if reminder_count == 0 else "⚠️ Final Reminder"
        body = f"{prefix}: {alert['title']}<br/>{alert['body_html']}"
        ok = await telegram_client.send_message_html(body)
        first_triggered = alert.get("first_triggered_at_utc") or now_utc_iso()
        next_reminder = _compute_next_reminder(first_triggered, reminder_count + 1)
        mark_alert_notified(
            conn,
            alert["id"],
            channel,
            ok,
            None if ok else "send_failed",
            now_utc_iso(),
            next_reminder_at_utc=next_reminder,
            increment_reminder=True,
        )
        sent += 1
    return sent

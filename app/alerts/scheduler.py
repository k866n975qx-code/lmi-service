from __future__ import annotations
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from zoneinfo import ZoneInfo
import structlog

from ..config import settings
from ..db import get_conn
from ..alerts.storage import migrate_alerts, list_open_alerts, close_stale_alerts
from ..alerts.evaluator import evaluate_alerts, build_daily_report_html, build_period_report_html
from ..alerts.notifier import send_alerts, send_digest, send_due_reminders
from ..services.telegram import TelegramClient

_log = structlog.get_logger()
_scheduler: AsyncIOScheduler | None = None

def get_scheduler() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler(timezone=ZoneInfo(getattr(settings, "local_tz", "America/Los_Angeles")))
    return _scheduler

def schedule_jobs(sched: AsyncIOScheduler | None = None):
    sched = sched or get_scheduler()
    tz = ZoneInfo(getattr(settings, "local_tz", "America/Los_Angeles"))
    # Daily 07:30
    sched.add_job(run_daily, CronTrigger(hour=getattr(settings, "alerts_daily_hour", 7), minute=getattr(settings, "alerts_daily_minute", 30), timezone=tz), id="alerts_daily", replace_existing=True)
    # Weekly Monday 07:30
    if getattr(settings, "alerts_weekly_enabled", 1):
        sched.add_job(run_weekly, CronTrigger(day_of_week="mon", hour=getattr(settings, "alerts_daily_hour", 7), minute=getattr(settings, "alerts_daily_minute", 30), timezone=tz), id="alerts_weekly", replace_existing=True)
    # Monthly 1st 07:40
    if getattr(settings, "alerts_monthly_enabled", 1):
        sched.add_job(run_monthly, CronTrigger(day="1", hour=getattr(settings, "alerts_daily_hour", 7), minute=40, timezone=tz), id="alerts_monthly", replace_existing=True)
    # Quarterly approx: Jan/Apr/Jul/Oct 1st 07:45
    if getattr(settings, "alerts_quarterly_enabled", 1):
        sched.add_job(run_quarterly, CronTrigger(month="1,4,7,10", day="1", hour=getattr(settings, "alerts_daily_hour", 7), minute=45, timezone=tz), id="alerts_quarterly", replace_existing=True)
    # Reminders every 30 minutes
    sched.add_job(run_reminders, CronTrigger(minute="*/30", timezone=tz), id="alerts_reminders", replace_existing=True)
    # Cleanup stale alerts nightly
    sched.add_job(run_cleanup, CronTrigger(hour=0, minute=5, timezone=tz), id="alerts_cleanup", replace_existing=True)
    sched.start()
    _log.info("alerts_scheduler_started")

async def run_daily():
    conn = get_conn(settings.db_path)
    migrate_alerts(conn)
    alerts = evaluate_alerts(conn)
    if not (getattr(settings, "telegram_bot_token", None) and getattr(settings, "telegram_chat_id", None)):
        return
    tg = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)
    if alerts:
        await send_alerts(conn, alerts, tg)
    as_of, html = build_daily_report_html(conn)
    open_warn = list_open_alerts(conn, min_severity=5, max_severity=7)
    if html:
        await send_digest(conn, open_warn, html, tg, severity_hint=5)

async def run_weekly():
    conn = get_conn(settings.db_path)
    migrate_alerts(conn)
    if not (getattr(settings, "telegram_bot_token", None) and getattr(settings, "telegram_chat_id", None)):
        return
    tg = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)
    as_of, html = build_period_report_html(conn, "weekly")
    open_info = list_open_alerts(conn, min_severity=1, max_severity=4)
    if html:
        await send_digest(conn, open_info, html, tg, severity_hint=3)

async def run_monthly():
    conn = get_conn(settings.db_path)
    migrate_alerts(conn)
    if not (getattr(settings, "telegram_bot_token", None) and getattr(settings, "telegram_chat_id", None)):
        return
    tg = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)
    as_of, html = build_period_report_html(conn, "monthly")
    open_info = list_open_alerts(conn, min_severity=1, max_severity=4)
    if html:
        await send_digest(conn, open_info, html, tg, severity_hint=3)

async def run_quarterly():
    conn = get_conn(settings.db_path)
    migrate_alerts(conn)
    if not (getattr(settings, "telegram_bot_token", None) and getattr(settings, "telegram_chat_id", None)):
        return
    tg = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)
    as_of, html = build_period_report_html(conn, "quarterly")
    open_info = list_open_alerts(conn, min_severity=1, max_severity=4)
    if html:
        await send_digest(conn, open_info, html, tg, severity_hint=3)

async def run_reminders():
    if not (getattr(settings, "telegram_bot_token", None) and getattr(settings, "telegram_chat_id", None)):
        return
    conn = get_conn(settings.db_path)
    migrate_alerts(conn)
    tg = TelegramClient(settings.telegram_bot_token, settings.telegram_chat_id)
    await send_due_reminders(conn, tg)

async def run_cleanup():
    conn = get_conn(settings.db_path)
    migrate_alerts(conn)
    closed = close_stale_alerts(conn, days=7)
    _log.info("alerts_cleanup_done", closed=closed)

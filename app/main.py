from fastapi import FastAPI
from .logging import setup_logging
from .api.routes import router as api_router
from .api.alerts import router as alerts_router
from .alerts.scheduler import get_scheduler, schedule_jobs
from .alerts.storage import migrate_alerts
from .config import settings
from .db import get_conn

setup_logging()
app = FastAPI(title="lmi-service")
app.include_router(api_router)
app.include_router(alerts_router, prefix="/api/alerts", tags=["Alerts"])

@app.on_event("startup")
def _startup():
    conn = get_conn(settings.db_path)
    migrate_alerts(conn)
    schedule_jobs(get_scheduler())

@app.on_event("shutdown")
def _shutdown():
    try:
        get_scheduler().shutdown(wait=False)
    except Exception:
        pass

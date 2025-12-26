from fastapi import FastAPI
from .logging import setup_logging
from .api.routes import router as api_router

setup_logging()
app = FastAPI(title="lmi-service")
app.include_router(api_router)

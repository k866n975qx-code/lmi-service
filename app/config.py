from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Load .env from repo root for local development and scripts.
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=True)
    lm_token: str = Field(alias="LM_TOKEN")
    lm_base_url: str = Field(default="https://dev.lunchmoney.app", alias="LM_BASE_URL")
    lm_plaid_account_ids: str | None = Field(default=None, alias="LM_PLAID_ACCOUNT_IDS")
    lm_start_date: str | None = Field(default=None, alias="LM_START_DATE")
    lm_lookback_days: int = Field(default=7, alias="LM_LOOKBACK_DAYS")
    yf_enable: int = Field(default=1, alias="YF_ENABLE")
    nasdaq_enable: int = Field(default=1, alias="NASDAQ_ENABLE")
    providers_openbb: int = Field(default=1, alias="PROVIDERS_OPENBB")
    yq_enable: int = Field(default=1, alias="YQ_ENABLE")
    stooq_enable: int = Field(default=1, alias="STOOQ_ENABLE")
    fred_api_key: str | None = Field(default=None, alias="FRED_API_KEY")
    local_tz: str = Field(default="America/Los_Angeles", alias="LOCAL_TZ")
    daily_cutover: str = Field(default="00:00", alias="DAILY_CUTOVER")
    sync_time_budget_seconds: int = Field(default=3600, alias="SYNC_TIME_BUDGET_SECONDS")
    http_timeout_seconds: float = Field(default=30.0, alias="HTTP_TIMEOUT_SECONDS")
    http_retry_attempts: int = Field(default=3, alias="HTTP_RETRY_ATTEMPTS")
    http_retry_backoff_seconds: float = Field(default=1.0, alias="HTTP_RETRY_BACKOFF_SECONDS")
    db_path: str = Field(default="./data/app.db", alias="DB_PATH")
    cusip_csv: str = Field(default="./CUSIP.csv", alias="CUSIP_CSV")
    benchmark_primary: str = Field(default="^GSPC", alias="BENCHMARK_PRIMARY")
    benchmark_secondary: str = Field(default="SPY", alias="BENCHMARK_SECONDARY")
    goal_target_monthly: float | None = Field(default=None, alias="GOAL_TARGET_MONTHLY")
    goal_monthly_contribution: float = Field(default=0.0, alias="GOAL_MONTHLY_CONTRIBUTION")
    goal_growth_window_months: int = Field(default=3, alias="GOAL_GROWTH_WINDOW_MONTHS")
    margin_apr_current: float = Field(default=0.0415, alias="MARGIN_APR_CURRENT")
    margin_apr_future: float = Field(default=0.0565, alias="MARGIN_APR_FUTURE")
    margin_apr_future_date: str = Field(default="2026-11-01", alias="MARGIN_APR_FUTURE_DATE")
    cache_enabled: int = Field(default=1, alias="CACHE_ENABLED")
    cache_dir: str = Field(default="./.cache", alias="CACHE_DIR")
    cache_db_path: str = Field(default="./data/cache.sqlite3", alias="CACHE_DB_PATH")
    cache_ttl_hours: int = Field(default=24, alias="CACHE_TTL_HOURS")
    market_batch_size: int = Field(default=25, alias="MARKET_BATCH_SIZE")
    market_rate_limit_seconds: float = Field(default=0.2, alias="MARKET_RATE_LIMIT_SECONDS")
    market_retry_attempts: int = Field(default=2, alias="MARKET_RETRY_ATTEMPTS")
    weekly_calendar_aligned: bool = Field(default=False, alias="WEEKLY_CALENDAR_ALIGNED")

settings = Settings()

# lmi-service

Service to ingest Lunch Money investment & margin-loan data, reconstruct holdings/dividends, enrich via multi-provider market data, and build daily + period snapshots with on-demand diffs.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

Create a `.env` (example):

```bash
# Lunch Money
LM_TOKEN=...
LM_PLAID_ACCOUNT_IDS=317631,317632
LM_START_DATE=2025-10-01

# Providers (disable by setting to 0)
PROVIDERS_OPENBB=1
YQ_ENABLE=1
STOOQ_ENABLE=1
NASDAQ_ENABLE=1
FRED_API_KEY=

# Runtime
LOCAL_TZ=America/Los_Angeles
DAILY_CUTOVER=00:00
DB_PATH=./data/app.db
CUSIP_CSV=./CUSIP.csv

# Benchmarks
BENCHMARK_PRIMARY=^GSPC
BENCHMARK_SECONDARY=SPY

# Sync + rate limits
SYNC_TIME_BUDGET_SECONDS=3600
HTTP_TIMEOUT_SECONDS=30
HTTP_RETRY_ATTEMPTS=3
HTTP_RETRY_BACKOFF_SECONDS=1
MARKET_BATCH_SIZE=25
MARKET_RATE_LIMIT_SECONDS=0.2
MARKET_RETRY_ATTEMPTS=2
```

Initialize and run:

```bash
python scripts/init_db.py
python scripts/sync_all.py
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Endpoints

- `POST /sync-all` → returns `{run_id}` immediately; pipeline continues in-process.
- `GET /status/{run_id}` → run status.
- `GET /health` → DB connectivity + last run.
- `GET /summaries/available` → list stored daily dates + period summaries.
- `GET /period-summary/{kind}/{as_of}?slim=true` → stored period summary (weekly|monthly|quarterly|yearly).
- `GET /period-summary/{kind}/{as_of}/{mode}?slim=true` → to-date or final (`mode=to_date|final`).
- `GET /compare/daily/{left_date}/{right_date}` → daily comparison (on demand).
- `GET /compare/period/{kind}/{left_as_of}/{right_as_of}` → period comparison (on demand).

`slim=true` removes provenance + cache noise; notes only appear when missing data is significant.

## Schema samples

- `samples/daily.json` / `samples/daily_slim.json`
- `samples/period.json` / `samples/period_slim.json`
- `samples/diff_daily.json`
- `samples/diff_period.json`

## Project status

- `TASKS.md` tracks the remaining work (most items are completed).
- `blueprint.md` is the target spec; it is nearly met and used as a reference.

## Design highlights

- SQLite (WAL). Append-only `lm_raw` for Lunch Money pulls.
- Holdings reconstructed from **Lunch Money first**, then market data fetched for needed symbols only.
- Provider chain with cache, rate limits, batching, and fallback to per-symbol fetches.
- Daily snapshot overwrites only when valid and changed; period snapshots persist at boundaries only.
- Diffs are always on demand; not persisted.

See `docs/` for architecture notes and field sourcing rules.

# LMI-Service Architecture

**Last Updated**: 2026-02-15

## Overview

LMI-Service is a dividend portfolio tracking service that ingests investment data from Lunch Money API, enriches it with market data from multiple providers, and provides comprehensive portfolio analytics through a REST API.

**Tech Stack**: Python 3.11, FastAPI, SQLite (WAL mode), Anthropic Claude (AI insights), Telegram Bot

---

## Directory Structure

```
app/
├── main.py                      # FastAPI app entry point
├── config.py                    # Configuration (Pydantic settings)
├── db.py                        # Database connection & DDL
├── logging.py                   # Structured logging setup
├── utils.py                     # Utilities (SHA256, retry, dates)
├── cache_layer.py               # Local cache for provider data
│
├── api/                         # REST API endpoints
│   ├── routes.py                # Main API router (health, sync, snapshots, diff)
│   ├── alerts.py                # Telegram webhook & alerts API
│   └── schemas.py               # Pydantic request/response models
│
├── pipeline/                    # Core data pipeline (21 modules)
│   ├── orchestrator.py          # Master sync loop (trigger_sync, _sync_impl)
│   ├── utils.py                 # Pipeline utilities (LM API, runs, CUSIP)
│   ├── transactions.py          # Investment transaction normalization
│   ├── holdings.py              # FIFO lot reconstruction
│   ├── corporate_actions.py     # Dividends & splits from providers
│   ├── market.py                # Multi-provider market data (yfinance, yq, stooq, openbb)
│   ├── snapshots.py             # Build daily/period snapshots (188K lines)
│   ├── snapshot_views.py        # Assemble snapshots from flat tables (120K lines)
│   ├── flat_persist.py          # Write flat tables (daily_portfolio, period_summary, etc.)
│   ├── periods.py               # Build period snapshots from dailies
│   ├── diff_daily.py            # Daily snapshot comparison
│   ├── diff_periods.py          # Period snapshot comparison
│   ├── metrics.py               # Risk metrics (TWR, Sortino, Sharpe, VaR, CVaR)
│   ├── facts.py                 # Upsert source facts from providers
│   ├── validation.py            # Snapshot validation
│   ├── null_reasons.py          # Replace nulls with explanatory strings
│   ├── locking.py               # Distributed locking for concurrent syncs
│   └── daily_transform_v5.py    # V4→V5 schema transform
│
├── providers/                   # Market data providers (7 adapters)
│   ├── yfinance_adapter.py      # Yahoo Finance (primary)
│   ├── yahooquery_adapter.py     # Yahoo Query (backup)
│   ├── stooq_adapter.py          # Stooq (backup)
│   ├── openbb_adapter.py         # OpenBB (router)
│   ├── fred_adapter.py           # FRED (macro data)
│   ├── nasdaq_calendar_adapter.py # Dividend calendar
│   └── common.py                # Shared provider utilities
│
├── alerts/                      # Alert system (4 modules)
│   ├── evaluator.py             # Alert evaluation (74K lines, 17 alert types)
│   ├── notifier.py              # Alert dispatch & grouping
│   ├── scheduler.py             # APScheduler jobs (daily digest, periodic)
│   ├── storage.py               # Alert CRUD & DDL
│   └── constants.py             # Alert threshold constants
│
└── services/                    # External services (3 modules)
    ├── telegram.py              # Telegram bot client & message formatting
    ├── ai_insights.py           # Anthropic Claude AI analysis
    └── charts.py                # Matplotlib chart generation

migrations/
├── 001_init.sql                 # Legacy initial schema (deprecated)
├── 002_flat_tables.sql          # Flat tables (deprecated)
├── 003_period_intervals.sql    # Period intervals (deprecated)
└── 004_consolidated_flat_schema.sql  # Current schema (v5, fully flattened)

scripts/
├── sync_all.py                  # Manual sync trigger
├── init_db.py                   # Initialize DB & seed CUSIP
├── backfill_period_summaries.py # Regenerate period snapshots
├── backfill_rolling_summaries.py # Regenerate rolling snapshots
├── cleanup_rolling_summaries.py  # Remove orphaned rolling snapshots
├── validate_flat_invariants.py   # Validate flat schema invariants
├── verify_attribution_and_ltv.py # Verify attribution & LTV calculations
├── pull_db_from_server.sh       # Pull DB from server (macOS)
└── push_db_to_server.sh         # Push DB to server (with dry-run)

systemd/
├── lmi@.service                 # API service (per-repo instance)
├── lmi-sync@.service            # One-shot sync job
└── lmi-sync@.timer              # Hourly sync trigger
```

---

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SYNC PIPELINE (_sync_impl)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. LUNCH MONEY PULL (append_lm_raw)                                       │
│     └─> lm_raw (append-only, SHA256 dedup)                               │
│                                                                              │
│  2. NORMALIZE TRANSACTIONS (normalize_investment_transactions)               │
│     └─> investment_transactions (upsert)                                   │
│                                                                              │
│  3. UPSERT DIVIDEND EVENTS (upsert_lm_dividend_events)                       │
│     └─> dividend_events_lm                                                 │
│                                                                              │
│  4. LOAD PROVIDER ACTIONS (load_provider_actions)                           │
│     ├─> Dividends (provider → dividend_events_provider)                     │
│     └─> Splits (provider → split_events)                                   │
│                                                                              │
│  5. ENSURE CUSIP MAP (ensure_cusip_map)                                     │
│     └─> cusip_map (ingest from CSV if empty)                               │
│                                                                              │
│  6. RECONSTRUCT HOLDINGS (reconstruct_holdings)                             │
│     └─> Holdings dict from transactions                                     │
│                                                                              │
│  7. LOAD MARKET DATA (MarketData.load)                                     │
│     ├─> yfinance (primary, 0.002s rate limit)                             │
│     ├─> yahooquery (backup, 0.12s rate limit)                             │
│     ├─> stooq (fallback, 0s rate limit)                                   │
│     └─> openbb (router, 0.05s rate limit)                                 │
│                                                                              │
│  8. ESTIMATE DIVIDEND SCHEDULE (estimate_dividend_schedule)                  │
│     └─> Populate upcoming dividends                                        │
│                                                                              │
│  9. BUILD DAILY SNAPSHOT (build_daily_snapshot)                             │
│     ├─> Assemble portfolio totals, holdings, income, risk, goals, margin   │
│     ├─> Transform V4 → V5 (daily_transform_v5.py)                         │
│     └─> Return V5 dict                                                     │
│                                                                              │
│ 10. VALIDATE SNAPSHOT (validate_daily_snapshot)                             │
│     └─> Check invariants (weight sum, NLV, holdings count)               │
│                                                                              │
│ 11. PERSIST DAILY SNAPSHOT (persist_daily_snapshot)                          │
│     ├─> flat_persist._write_daily_flat (13 tables)                       │
│     │   ├─> daily_portfolio (1 row)                                       │
│     │   ├─> daily_holdings (N rows)                                       │
│     │   ├─> daily_goal_tiers (6 rows)                                     │
│     │   ├─> daily_margin_rate_scenarios (N rows)                          │
│     │   ├─> daily_return_attribution (N rows)                             │
│     │   └─> daily_dividends_upcoming (N rows)                             │
│     └─> Delete old snapshot_daily_current row                             │
│                                                                              │
│ 12. PERSIST ROLLING SUMMARIES (persist_rolling_summaries)                    │
│     └─> period_summary (WEEK/MONTH/QUARTER/YEAR with is_rolling=1)        │
│                                                                              │
│ 13. MAYBE PERSIST PERIODIC (maybe_persist_periodic)                          │
│     └─> period_summary (on period boundary)                               │
│                                                                              │
│ 14. EVALUATE ALERTS (evaluate_alerts)                                       │
│     ├─> 17 alert types (margin, risk, income, goal, dividend, etc.)      │
│     └─> alert_messages table                                              │
│                                                                              │
│ 15. SEND ALERTS (send_alerts_sync)                                          │
│     └─> Telegram bot (grouped, inline buttons)                           │
│                                                                              │
│ 16. FACTS FROM SOURCES (upsert_facts_from_sources)                           │
│     └─> facts_source_daily (provenance tracking)                          │
│                                                                              │
│ 17. FINISH RUN (finish_run_ok / finish_run_fail)                             │
│     └─> Update runs table status                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Input Sources
| Source | Table | Description |
|--------|-------|-------------|
| Lunch Money API | `lm_raw` | Raw JSON payloads (append-only) |
| - | `investment_transactions` | Normalized transactions (buy, sell, dividend, etc.) |
| - | `dividend_events_lm` | Dividend events from LM |
| Market providers | `dividend_events_provider` | Dividend events from yfinance/yahooquery |
| - | `split_events` | Stock split events |
| CUSIP CSV | `cusip_map` | Symbol to CUSIP mappings |

### Derived Tables (Snapshot)
| Category | Tables | Description |
|----------|--------|-------------|
| Daily | `daily_portfolio`, `daily_holdings`, `daily_goal_tiers`, `daily_margin_rate_scenarios`, `daily_return_attribution`, `daily_dividends_upcoming` | Daily snapshot data |
| Period | `period_summary`, `period_risk_stats`, `period_intervals`, `period_interval_holdings`, `period_interval_attribution`, `period_holding_changes`, `period_macro_stats` | Period aggregates |
| Activity | `period_activity`, `period_contributions`, `period_withdrawals`, `period_dividend_events`, `period_trades`, `period_margin_detail`, `period_position_lists` | Activity tracking |
| System | `runs`, `locks`, `metadata`, `alert_messages` | System state |

---

## Provider Chain

Market data providers are tried in order with per-provider rate limits:

| Provider | Rate Limit | Use Case |
|----------|------------|----------|
| yfinance | 0.002s | Primary (fastest) |
| yahooquery | 0.12s | Backup |
| stooq | 0s | Fallback |
| openbb | 0.05s | Router for yfinance/FMP/Polygon |

**Macro Data**: FRED (risk-free rate, benchmark yields)

**Benchmarks**: Primary `^GSPC` (S&P 500), Secondary `SPY`

---

## Facts vs Derived Fields

**Source Facts** ([`facts_source_daily`](app/db.py)):
- Only provider-backed fields are stored
- Each row records: `{provider, endpoint, params, source_ref, commit_sha}`
- Used for provenance and debugging

**Derived Fields**:
- Computed during snapshot build from source facts
- Not stored in `facts_source_daily`
- Examples: TWR, volatility, Sharpe, Sortino, VaR, CVaR, goal pace

---

## API Endpoints

### Health & Sync
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health & last run status |
| `/sync-all` | POST | Trigger background sync |
| `/sync-window` | POST | Trigger sync for date window |
| `/status/{run_id}` | GET | Sync run status |
| `/cache/{action}` | POST | Cache admin (invalidate/backfill) |

### Snapshots
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v5/daily` | GET | Latest daily snapshot (V5) |
| `/compare/daily/{date1}/{date2}` | GET | Compare two daily snapshots |
| `/period-summary/{kind}/latest` | GET | Latest period summary (weekly/monthly/quarterly/yearly) |
| `/period-summary/{kind}/rolling` | GET | Rolling period summary |
| `/summaries/available` | GET | List available period summaries |

### Telegram & Alerts
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/alerts/telegram/webhook` | POST | Telegram webhook (commands, callbacks) |
| `/api/alerts/evaluate` | POST | Manual alert evaluation |
| `/api/alerts/sync-settings` | POST | Sync alert settings to Telegram |

---

## Alert System

**17 Alert Types** ([`alerts/evaluator.py`](app/alerts/evaluator.py)):
- Margin: LTV yellow/red, margin interest spike, coverage ratio low, margin call buffer low
- Risk: Volatility spike, VaR breach, drawdown
- Income: Dividend cut/omit, projected monthly low
- Goal: Off track, tier change, milestone, pace slippage, surplus decline
- Dividend: Ex-dividend today, dividend decrease
- Portfolio: Large withdrawal, concentration high

**Scheduled Jobs** ([`alerts/scheduler.py`](app/alerts/scheduler.py)):
- Daily digest (8 PM ET, configurable)
- Morning brief (9:25 AM ET, pre-market)
- Evening recap (4:15 PM ET, post-close)

**Telegram Features**:
- 40+ commands (`/status`, `/income`, `/goal`, `/pace`, `/chart`, `/position`, etc.)
- Inline buttons (acknowledge, silence, view details)
- Natural language queries ("how am I doing?")
- Collapsible reports
- Chart generation (10 chart types)

---

## Key Design Decisions

1. **Append-only raw data**: `lm_raw` is never updated, only appended. Deduplication via SHA256.

2. **Fully flattened schema**: No JSON blobs in daily/period tables (v5 migration). Every field has its own column.

3. **V5 schema**: Snapshot format uses nested dicts but stored flat. `assemble_daily_snapshot` / `assemble_period_snapshot` reassemble from flat tables.

4. **Daily overwrite**: `snapshot_daily_current` has one row per date, overwritten on each successful sync.

5. **Period snapshots**: Generated on-demand from daily snapshots using `periods.py`.

6. **FIFO lots**: Holdings reconstructed from transactions using FIFO cost basis.

7. **Provider fallback**: Try yfinance → yahooquery → stooq with per-provider rate limits.

8. **Locking**: Distributed lock (`locks` table) prevents concurrent syncs.

9. **Time budget**: Sync aborts if `sync_time_budget_seconds` exceeded (default 300s).

10. **Validation**: Snapshot invariants checked before persist (weight sum, NLV, holdings count).

---

## Configuration

Key environment variables (see [`config.py`](app/config.py)):

| Variable | Default | Description |
|----------|---------|-------------|
| `LM_TOKEN` | - | Lunch Money API token |
| `TELEGRAM_BOT_TOKEN` | - | Telegram bot token |
| `TELEGRAM_CHAT_ID` | - | Telegram chat ID |
| `DB_PATH` | `./data/app.db` | SQLite database path |
| `AS_OF_DATE` | today | Target date for snapshots |
| `LOCAL_TZ` | `America/Los_Angeles` | Local timezone |
| `SYNC_TIME_BUDGET_SECONDS` | 300 | Max sync duration |
| `CACHE_ENABLED` | `true` | Enable local cache |
| `CACHE_TTL_HOURS` | 24 | Cache TTL |

---

## Deployment

**Local Development**: See [`COMMANDS_LOCAL.md`](COMMANDS_LOCAL.md)

**Server (Ubuntu)**: See [`COMMANDS_SERVER.md`](COMMANDS_SERVER.md)

**Systemd Services**:
- `lmi@<repo>.service` - API service (port 8010, auto-restart)
- `lmi-sync@<repo>.timer` - Hourly sync trigger
- `lmi-sync@<repo>.service` - One-shot sync job

---

## Related Documentation

- [`ROADMAP.md`](ROADMAP.md) - Development roadmap
- [`schema_implementation_status.md`](schema_implementation_status.md) - V5 implementation status
- [`flatten_remaining_fields.md`](flatten_remaining_fields.md) - Future schema work

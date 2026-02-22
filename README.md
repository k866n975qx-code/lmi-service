# lmi-service

A dividend portfolio tracking service that ingests investment data from Lunch Money API, enriches it with market data from multiple providers, and provides comprehensive portfolio analytics through a REST API and Telegram bot.

## Features

- **Automated Data Ingestion**: Hourly sync with Lunch Money API for transactions, balances, and dividends
- **Multi-Provider Market Data**: yfinance, yahooquery, stooq, OpenBB with automatic fallback
- **Comprehensive Analytics**: Portfolio values, income tracking, risk metrics (TWR, Sortino, Sharpe, VaR), goal progress
- **Period Snapshots**: Weekly, monthly, quarterly, and yearly summaries with activity tracking
- **Alert System**: 17 alert types with Telegram notifications (margin, risk, income, goals, dividends)
- **Telegram Bot**: 40+ commands for portfolio insights, charts, and natural language queries
- **AI Insights**: Anthropic Claude-powered portfolio analysis

## Tech Stack

- **Python 3.11** with FastAPI
- **SQLite** (WAL mode) with fully flattened v5 schema
- **Anthropic Claude** for AI insights
- **Telegram Bot API** for notifications and interactive commands
- **Matplotlib** for chart generation

## Quick Start

### 1. Setup

```bash
# Clone repository
git clone <repo-url>
cd lmi-service

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -U pip wheel
pip install -r requirements.txt
```

### 2. Configure

Create a `.env` file:

```bash
# Lunch Money
LM_TOKEN=your_lm_token_here
LM_PLAID_ACCOUNT_IDS=317631,317632
LM_START_DATE=2025-10-01

# Telegram (optional, for alerts)
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Providers
PROVIDERS_OPENBB=1
YF_ENABLE=1
YQ_ENABLE=1
STOOQ_ENABLE=1
FRED_API_KEY=your_fred_key_here

# Runtime
LOCAL_TZ=America/Los_Angeles
DB_PATH=./data/app.db
CUSIP_CSV=./CUSIP.csv

# Benchmarks
BENCHMARK_PRIMARY=^GSPC
BENCHMARK_SECONDARY=SPY
```

### 3. Initialize and Run

```bash
# Initialize database
python scripts/init_db.py

# Run first sync
python scripts/sync_all.py

# Start API server
uvicorn app.main:app --host 0.0.0.0 --port 8010
```

### 4. Verify

```bash
# Health check
curl http://127.0.0.1:8010/health

# Get latest daily snapshot
curl http://127.0.0.1:8010/api/v5/daily

# Get latest period summary
curl http://127.0.0.1:8010/period-summary/weekly/latest
```

## API Endpoints

### Health & Sync
| Endpoint | Description |
|----------|-------------|
| `GET /health` | Service health + last run status |
| `POST /sync-all` | Trigger background sync (returns `run_id`) |
| `GET /status/{run_id}` | Sync run status |
| `POST /sync-window` | Trigger sync for date window |

### Snapshots
| Endpoint | Description |
|----------|-------------|
| `GET /api/v5/daily` | Latest daily snapshot (V5 schema) |
| `GET /period-summary/{kind}/latest` | Latest period (weekly/monthly/quarterly/yearly) |
| `GET /period-summary/{kind}/rolling` | Rolling period summary |
| `GET /summaries/available` | List available period summaries |

### Comparisons
| Endpoint | Description |
|----------|-------------|
| `GET /compare/daily/{date1}/{date2}` | Compare two daily snapshots |
| `GET /compare/period/{kind}/{left}/{right}` | Compare two period summaries |

### Telegram & Alerts
| Endpoint | Description |
|----------|-------------|
| `POST /api/alerts/telegram/webhook` | Telegram webhook |
| `POST /api/alerts/evaluate` | Manual alert evaluation |
| `POST /api/alerts/sync-settings` | Sync settings to Telegram |

## Telegram Bot

The Telegram bot provides 40+ commands for portfolio insights:

**Core Commands:**
- `/status` - Portfolio overview
- `/income` - Income summary and attribution
- `/goal` - Goal progress and tier analysis
- `/pace` - Goal pace tracking across time windows
- `/perf` - Performance metrics
- `/risk` - Risk analysis (volatility, drawdown, VaR)
- `/alerts` - Open alerts
- `/chart` - Generate charts (pace, attribution, performance, etc.)

**Features:**
- Natural language queries ("how am I doing?")
- Inline buttons for alert actions
- Collapsible reports
- 10 chart types

Setup: See [`docs/TELEGRAM_BOT_SETUP.md`](docs/TELEGRAM_BOT_SETUP.md)

## Deployment

### Local Development

See [`docs/COMMANDS_LOCAL.md`](docs/COMMANDS_LOCAL.md) for:
- Development setup
- Running tests
- Backfilling data
- Troubleshooting

### Server (Ubuntu/Production)

See [`docs/COMMANDS_SERVER.md`](docs/COMMANDS_SERVER.md) for:
- Systemd service setup
- Hourly sync configuration
- Monitoring and logging
- Cloudflare tunnel setup

### Systemd Services

```bash
# Install services
sudo cp systemd/lmi@.service /etc/systemd/system/
sudo cp systemd/lmi-sync@.service /etc/systemd/system/
sudo cp systemd/lmi-sync@.timer /etc/systemd/system/
sudo systemctl daemon-reload

# Enable (replace <repo> with folder name)
sudo systemctl enable --now lmi@<repo>.service
sudo systemctl enable --now lmi-sync@<repo>.timer
```

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System architecture and design |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Development roadmap |
| [`docs/COMMANDS_LOCAL.md`](docs/COMMANDS_LOCAL.md) | Local development commands |
| [`docs/COMMANDS_SERVER.md`](docs/COMMANDS_SERVER.md) | Server deployment commands |
| [`docs/LOCAL_TEST_COMMANDS.md`](docs/LOCAL_TEST_COMMANDS.md) | Quick command reference |
| [`docs/TELEGRAM_BOT_SETUP.md`](docs/TELEGRAM_BOT_SETUP.md) | Telegram bot setup |
| [`docs/schema_implementation_status.md`](docs/schema_implementation_status.md) | V5 schema implementation status |
| [`docs/flatten_remaining_fields.md`](docs/flatten_remaining_fields.md) | Future schema work |

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/init_db.py` | Initialize database and seed CUSIP mappings |
| `scripts/sync_all.py` | Manual sync trigger |
| `scripts/backfill_period_summaries.py` | Regenerate period snapshots |
| `scripts/backfill_rolling_summaries.py` | Regenerate rolling summaries |
| `scripts/cleanup_rolling_summaries.py` | Remove orphaned rolling snapshots |
| `scripts/validate_flat_invariants.py` | Validate schema invariants |
| `scripts/verify_attribution_and_ltv.py` | Verify calculations |
| `scripts/pull_db_from_server.sh` | Pull DB from server (macOS) |
| `scripts/push_db_to_server.sh` | Push DB to server (with dry-run) |

## Design Highlights

- **Append-only raw data**: `lm_raw` table never updated, only appended. Deduplication via SHA256.
- **Fully flattened schema**: V5 schema has no JSON blobs - every field has its own column.
- **Provider fallback chain**: yfinance → yahooquery → stooq with per-provider rate limits.
- **FIFO lot reconstruction**: Holdings reconstructed from transactions using FIFO cost basis.
- **On-demand diffs**: Comparisons computed on request, not stored.
- **Distributed locking**: Prevents concurrent syncs via `locks` table.
- **Time budget**: Sync aborts if `sync_time_budget_seconds` exceeded (default: 300s).

## Database Schema

**Current Schema Version**: v5 (fully flattened)

**Key Tables**:
- `lm_raw` - Append-only Lunch Money API data
- `investment_transactions` - Normalized transactions
- `daily_portfolio`, `daily_holdings`, etc. - Daily snapshot data
- `period_summary`, `period_intervals`, etc. - Period aggregates
- `alert_messages` - Alert history
- `runs` - Sync run history

See `migrations/004_consolidated_flat_schema.sql` for complete schema.

## License

MIT

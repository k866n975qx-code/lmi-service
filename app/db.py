import sqlite3
from pathlib import Path
from datetime import datetime, timezone

def get_conn(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None)  # autocommit
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

DDL = [
    # Raw LM data (append-only)
    """
CREATE TABLE IF NOT EXISTS lm_raw (
  run_id TEXT NOT NULL,
  pulled_at_utc TEXT NOT NULL,
  endpoint TEXT NOT NULL,
  object_id TEXT,
  payload_json TEXT NOT NULL,
  payload_sha256 TEXT NOT NULL,
  PRIMARY KEY (run_id, endpoint, payload_sha256)
);
""",
    "CREATE INDEX IF NOT EXISTS ix_lm_raw_endpoint_id_time ON lm_raw(endpoint, object_id, pulled_at_utc DESC);",

    # CUSIP map (ingested once)
    """
CREATE TABLE IF NOT EXISTS cusip_map (
  cusip TEXT PRIMARY KEY,
  symbol TEXT NOT NULL,
  description TEXT
);
""",

    # Metadata
    """
CREATE TABLE IF NOT EXISTS metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
""",

    # Facts table: every sourced field (provider-backed)
    """
CREATE TABLE IF NOT EXISTS facts_source_daily (
  as_of_date_local TEXT NOT NULL,
  scope TEXT NOT NULL,         -- 'portfolio'|'symbol'
  symbol TEXT,                 -- NULL for portfolio-scope
  field_path TEXT NOT NULL,
  value_text TEXT,
  value_num REAL,
  value_int INTEGER,
  value_type TEXT NOT NULL,    -- 'text'|'num'|'int'|'bool'|'date'
  source_type TEXT NOT NULL,   -- 'market'|'fundamental'|'derived'
  provider TEXT NOT NULL,
  endpoint TEXT,
  params TEXT,
  source_ref TEXT,
  commit_sha TEXT,
  provider_rank INTEGER DEFAULT 1,
  run_id TEXT NOT NULL,
  created_at_utc TEXT NOT NULL,
  PRIMARY KEY (as_of_date_local, scope, symbol, field_path, source_type, provider, endpoint)
);
""",
    "CREATE INDEX IF NOT EXISTS ix_facts_symbol_field ON facts_source_daily(symbol, field_path);",

    # Persisted snapshots (non-to-date)
    """
CREATE TABLE IF NOT EXISTS snapshots (
  snapshot_id TEXT PRIMARY KEY,
  period_type TEXT NOT NULL,  -- 'WEEK'|'MONTH'|'QUARTER'|'YEAR'
  period_start_date TEXT NOT NULL,
  period_end_date TEXT NOT NULL,
  built_from_run_id TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  payload_sha256 TEXT NOT NULL,
  created_at_utc TEXT NOT NULL
);
""",
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_snapshots_period ON snapshots(period_type, period_start_date, period_end_date);",

    # Current daily snapshot
    """
CREATE TABLE IF NOT EXISTS snapshot_daily_current (
  as_of_date_local TEXT PRIMARY KEY,
  built_from_run_id TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  payload_sha256 TEXT NOT NULL,
  updated_at_utc TEXT NOT NULL
);
""",

    # Runs table
    """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  started_at_utc TEXT NOT NULL,
  finished_at_utc TEXT,
  status TEXT NOT NULL,   -- 'running'|'succeeded'|'failed'
  error_message TEXT,
  lm_window_start TEXT,
  lm_window_end TEXT,
  git_head_sha TEXT
);
""",

    # Account balances (plaid accounts via Lunch Money)
    """
CREATE TABLE IF NOT EXISTS account_balances (
  as_of_date_local TEXT NOT NULL,
  plaid_account_id TEXT NOT NULL,
  name TEXT,
  institution_name TEXT,
  type TEXT,
  subtype TEXT,
  balance REAL,
  credit_limit REAL,
  currency TEXT,
  balance_last_update TEXT,
  source TEXT NOT NULL,
  run_id TEXT NOT NULL,
  pulled_at_utc TEXT NOT NULL,
  PRIMARY KEY (as_of_date_local, plaid_account_id)
);
""",
    "CREATE INDEX IF NOT EXISTS ix_account_balances_plaid_id ON account_balances(plaid_account_id, as_of_date_local);",

    # Normalized investment transactions (from LM raw)
    """
CREATE TABLE IF NOT EXISTS investment_transactions (
  lm_transaction_id TEXT PRIMARY KEY,
  plaid_account_id TEXT,
  date TEXT,
  transaction_datetime TEXT,
  amount REAL,
  currency TEXT,
  name TEXT,
  category_name TEXT,
  transaction_type TEXT,
  quantity REAL,
  price REAL,
  fees REAL,
  security_id TEXT,
  cusip TEXT,
  symbol TEXT,
  plaid_investment_transaction_id TEXT,
  plaid_cancel_transaction_id TEXT,
  source TEXT NOT NULL,
  run_id TEXT NOT NULL,
  pulled_at_utc TEXT NOT NULL
);
""",
    "CREATE INDEX IF NOT EXISTS ix_investment_tx_account_date ON investment_transactions(plaid_account_id, date);",
    "CREATE INDEX IF NOT EXISTS ix_investment_tx_symbol_date ON investment_transactions(symbol, date);",

    # Dividend events (LM truth)
    """
CREATE TABLE IF NOT EXISTS dividend_events_lm (
  symbol TEXT NOT NULL,
  ex_date TEXT NOT NULL,
  pay_date TEXT,
  ex_date_est TEXT,
  pay_date_est TEXT,
  match_source TEXT,
  match_method TEXT,
  match_days_delta INTEGER,
  amount REAL NOT NULL,
  currency TEXT,
  frequency TEXT,
  source TEXT NOT NULL,
  run_id TEXT NOT NULL,
  fetched_at_utc TEXT NOT NULL,
  PRIMARY KEY (symbol, ex_date, amount, source)
);
""",
    "CREATE INDEX IF NOT EXISTS ix_dividend_events_lm_symbol_date ON dividend_events_lm(symbol, ex_date);",

    # Dividend events (provider history)
    """
CREATE TABLE IF NOT EXISTS dividend_events_provider (
  symbol TEXT NOT NULL,
  ex_date TEXT NOT NULL,
  pay_date TEXT,
  amount REAL NOT NULL,
  currency TEXT,
  frequency TEXT,
  source TEXT NOT NULL,
  run_id TEXT NOT NULL,
  fetched_at_utc TEXT NOT NULL,
  PRIMARY KEY (symbol, ex_date, amount, source)
);
""",
    "CREATE INDEX IF NOT EXISTS ix_dividend_events_provider_symbol_date ON dividend_events_provider(symbol, ex_date);",

    # Split events (provider)
    """
CREATE TABLE IF NOT EXISTS split_events (
  symbol TEXT NOT NULL,
  ex_date TEXT NOT NULL,
  ratio REAL NOT NULL,
  source TEXT NOT NULL,
  run_id TEXT NOT NULL,
  fetched_at_utc TEXT NOT NULL,
  PRIMARY KEY (symbol, ex_date, ratio, source)
);
""",
    "CREATE INDEX IF NOT EXISTS ix_split_events_symbol_date ON split_events(symbol, ex_date);",
]

def migrate(conn: sqlite3.Connection):
    cur = conn.cursor()
    for stmt in DDL:
        cur.execute(stmt)
    cols = {row[1] for row in cur.execute("PRAGMA table_info(facts_source_daily)").fetchall()}
    if cols:
        if "source_type" not in cols:
            cur.execute("ALTER TABLE facts_source_daily ADD COLUMN source_type TEXT NOT NULL DEFAULT 'unknown'")
        if "provider_rank" not in cols:
            cur.execute("ALTER TABLE facts_source_daily ADD COLUMN provider_rank INTEGER NOT NULL DEFAULT 1")
    div_cols = {row[1] for row in cur.execute("PRAGMA table_info(dividend_events_lm)").fetchall()}
    if div_cols:
        if "ex_date_est" not in div_cols:
            cur.execute("ALTER TABLE dividend_events_lm ADD COLUMN ex_date_est TEXT")
        if "pay_date_est" not in div_cols:
            cur.execute("ALTER TABLE dividend_events_lm ADD COLUMN pay_date_est TEXT")
        if "match_source" not in div_cols:
            cur.execute("ALTER TABLE dividend_events_lm ADD COLUMN match_source TEXT")
        if "match_method" not in div_cols:
            cur.execute("ALTER TABLE dividend_events_lm ADD COLUMN match_method TEXT")
        if "match_days_delta" not in div_cols:
            cur.execute("ALTER TABLE dividend_events_lm ADD COLUMN match_days_delta INTEGER")

    legacy = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='dividend_events'"
    ).fetchone()
    if legacy:
        lm_count = cur.execute("SELECT COUNT(*) FROM dividend_events_lm").fetchone()[0]
        if lm_count == 0:
            cur.execute(
                """
                INSERT OR IGNORE INTO dividend_events_lm (
                  symbol, ex_date, pay_date, ex_date_est, pay_date_est, match_source,
                  match_method, match_days_delta, amount, currency, frequency, source,
                  run_id, fetched_at_utc
                )
                SELECT symbol, ex_date, pay_date, ex_date_est, pay_date_est, match_source,
                       match_method, match_days_delta, amount, currency, frequency, source,
                       run_id, fetched_at_utc
                FROM dividend_events
                WHERE source='lunchmoney'
                """
            )
        provider_count = cur.execute("SELECT COUNT(*) FROM dividend_events_provider").fetchone()[0]
        if provider_count == 0:
            cur.execute(
                """
                INSERT OR IGNORE INTO dividend_events_provider (
                  symbol, ex_date, pay_date, amount, currency, frequency, source, run_id, fetched_at_utc
                )
                SELECT symbol, ex_date, pay_date, amount, currency, frequency, source, run_id, fetched_at_utc
                FROM dividend_events
                WHERE source!='lunchmoney'
                """
            )
    conn.commit()

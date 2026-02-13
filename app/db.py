import sqlite3
from pathlib import Path
from datetime import datetime, timezone

_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"
_DB_PATH = None  # Will be set when get_conn is called


def get_conn(db_path: str) -> sqlite3.Connection:
    global _DB_PATH
    _DB_PATH = db_path  # Store for migration use
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
    # Consolidated schema migration (004) - replaces 002 and 003
    if _run_consolidated_schema_migration(conn, str(_DB_PATH)):
        print("Consolidated schema migration completed")
        # Connection remains open, no need to reopen
    
    # Drop legacy snapshot tables only if they exist (replaced by daily_portfolio, period_summary).
    # NEVER drop or truncate account_balances - it holds per-date balance history from sync.
    # Note: 004 creates fresh schema, so these drops may be no-ops
    cur.execute("DROP TABLE IF EXISTS snapshot_daily_current")
    cur.execute("DROP TABLE IF EXISTS snapshots")
    conn.commit()


def _run_flat_tables_migration(conn: sqlite3.Connection):
    """Execute migrations/002_flat_tables.sql to create flat schema tables."""
    path = _MIGRATIONS_DIR / "002_flat_tables.sql"
    if not path.exists():
        return
    sql = path.read_text()
    conn.executescript(sql)


def _run_period_interval_migration(conn: sqlite3.Connection):
    """Run migrations/003: add period_intervals columns and create child tables (idempotent)."""
    path = _MIGRATIONS_DIR / "003_period_interval_cols.sql"
    if not path.exists():
        return
    cur = conn.cursor()
    # Create child tables and indexes (IF NOT EXISTS)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS period_interval_holdings (
          period_type TEXT NOT NULL,
          period_start_date TEXT NOT NULL,
          period_end_date TEXT NOT NULL,
          interval_label TEXT NOT NULL,
          symbol TEXT NOT NULL,
          weight_pct REAL,
          market_value REAL,
          pnl_pct REAL,
          pnl_dollar REAL,
          projected_monthly_dividend REAL,
          current_yield_pct REAL,
          sharpe_1y REAL,
          sortino_1y REAL,
          risk_quality_category TEXT,
          PRIMARY KEY (period_type, period_start_date, period_end_date, interval_label, symbol)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS period_interval_attribution (
          period_type TEXT NOT NULL,
          period_start_date TEXT NOT NULL,
          period_end_date TEXT NOT NULL,
          interval_label TEXT NOT NULL,
          window TEXT NOT NULL,
          total_return_pct REAL,
          income_contribution_pct REAL,
          price_contribution_pct REAL,
          top_json TEXT,
          bottom_json TEXT,
          PRIMARY KEY (period_type, period_start_date, period_end_date, interval_label, window)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS ix_pih_parent ON period_interval_holdings(period_type, period_start_date, period_end_date)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_pia_parent ON period_interval_attribution(period_type, period_start_date, period_end_date)")
    # Add columns to period_intervals only if missing (idempotent)
    pi_cols = {row[1] for row in cur.execute("PRAGMA table_info(period_intervals)").fetchall()}
    new_cols = [
        ("forward_12m_total", "REAL"),
        ("yield_pct", "REAL"),
        ("yield_on_cost_pct", "REAL"),
        ("twr_6m_pct", "REAL"),
        ("vol_30d_pct", "REAL"),
        ("vol_90d_pct", "REAL"),
        ("calmar_1y", "REAL"),
        ("max_drawdown_1y_pct", "REAL"),
        ("var_90_1d_pct", "REAL"),
        ("var_95_1d_pct", "REAL"),
        ("cvar_90_1d_pct", "REAL"),
        ("omega_ratio_1y", "REAL"),
        ("ulcer_index_1y", "REAL"),
        ("income_stability_score", "REAL"),
        ("beta_portfolio", "REAL"),
        ("annual_interest_expense", "REAL"),
        ("margin_call_buffer_pct", "REAL"),
        ("goal_progress_pct", "REAL"),
        ("goal_months_to_goal", "REAL"),
        ("goal_projected_monthly", "REAL"),
        ("goal_net_progress_pct", "REAL"),
    ]
    for col_name, col_type in new_cols:
        if col_name not in pi_cols:
            cur.execute(f"ALTER TABLE period_intervals ADD COLUMN {col_name} {col_type}")
    conn.commit()


def _run_consolidated_schema_migration(conn: sqlite3.Connection, db_path: str) -> bool:
    """
    Execute migrations/004_consolidated_flat_schema.sql.
    This creates the full flattened schema.
    Returns True if migration was run, False if already at version 5.
    """
    cur = conn.cursor()
    
    # Check current schema version
    cur.execute("PRAGMA user_version")
    current_version = cur.fetchone()[0]
    
    if current_version >= 5:
        print(f"Schema already at version {current_version}, skipping migration")
        return False
    
    print(f"Current schema version: {current_version}")
    print(f"Running consolidated schema migration to version 5...")
    
    # Create backup before migration using SQLite backup API (doesn't require closing connection)
    import shutil
    import os
    from datetime import datetime
    
    backup_dir = os.path.dirname(db_path) + "/backups"
    os.makedirs(backup_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"pre_migration_v{current_version}_to_v5_{timestamp}.db"
    backup_path = os.path.join(backup_dir, backup_filename)
    
    print(f"Creating backup before migration...")
    print(f"  Source: {db_path}")
    print(f"  Target: {backup_path}")
    
    # Use SQLite backup API to create consistent backup
    backup_conn = sqlite3.connect(backup_path, isolation_level=None)
    backup_conn.execute("PRAGMA journal_mode=WAL;")
    conn.backup(backup_conn)
    backup_conn.commit()
    backup_conn.close()
    
    print(f"Backup created successfully")
    
    # Run consolidated migration (using existing open connection)
    path = _MIGRATIONS_DIR / "004_consolidated_flat_schema.sql"
    if not path.exists():
        print(f"Warning: Migration file not found: {path}")
        return False
    
    sql = path.read_text()
    conn.executescript(sql)
    
    # Set schema version to 5
    conn.execute("PRAGMA user_version=5")
    conn.commit()
    
    print(f"Migration to schema version 5 complete")
    print(f"Backup saved to: {backup_path}")
    
    return True

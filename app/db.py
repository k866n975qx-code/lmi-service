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
    if _run_realized_schema_migration(conn):
        print("Realized P&L schema migration completed")
    if _run_account_schema_migration(conn):
        print("Account-level flat schema migration completed")
    
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

    existing_tables = {
        row[0]
        for row in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    consolidated_tables = {
        "daily_portfolio",
        "daily_holdings",
        "period_summary",
        "period_activity",
        "period_trades",
    }
    has_consolidated_schema = consolidated_tables.issubset(existing_tables)

    if has_consolidated_schema and current_version < 5:
        conn.execute("PRAGMA user_version=5")
        conn.commit()
        print("Detected existing consolidated schema; updated user_version to 5")
        return False

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


def _column_names(cur: sqlite3.Cursor, table_name: str) -> set[str]:
    try:
        return {row[1] for row in cur.execute(f"PRAGMA table_info({table_name})").fetchall()}
    except sqlite3.OperationalError:
        return set()


def _run_realized_schema_migration(conn: sqlite3.Connection) -> bool:
    cur = conn.cursor()
    changed = False

    investment_tx_columns = {
        "economic_bucket": "TEXT",
        "cash_flow_direction": "TEXT",
        "cash_flow_amount": "REAL",
        "external_flow_amount": "REAL",
        "trading_flow_amount": "REAL",
        "income_flow_amount": "REAL",
        "financing_flow_amount": "REAL",
        "fee_flow_amount": "REAL",
    }
    period_activity_columns = {
        "buys_total": "REAL",
        "sells_total": "REAL",
        "trading_net_cash": "REAL",
        "realized_capital_pnl": "REAL",
        "fees_total": "REAL",
        "interest_income_total": "REAL",
        "realized_total_return": "REAL",
    }
    period_trades_columns = {
        "shares_bought": "REAL",
        "shares_sold": "REAL",
        "buy_amount_total": "REAL",
        "sell_amount_total": "REAL",
        "net_trade_cash": "REAL",
        "realized_cost_basis": "REAL",
        "realized_capital_pnl": "REAL",
    }

    for table_name, columns in (
        ("investment_transactions", investment_tx_columns),
        ("period_activity", period_activity_columns),
        ("period_trades", period_trades_columns),
    ):
        existing = _column_names(cur, table_name)
        for column_name, column_type in columns.items():
            if column_name in existing:
                continue
            cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
            changed = True

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS realized_trade_ledger (
          lm_transaction_id TEXT PRIMARY KEY,
          date TEXT NOT NULL,
          transaction_datetime TEXT,
          symbol TEXT NOT NULL,
          shares_sold REAL NOT NULL,
          sell_price REAL,
          gross_proceeds REAL,
          fees REAL,
          net_proceeds REAL,
          realized_cost_basis REAL,
          realized_pnl REAL,
          realized_pnl_pct REAL,
          matched_shares REAL,
          unmatched_shares REAL,
          matched_complete INTEGER NOT NULL DEFAULT 1,
          lots_closed_count INTEGER NOT NULL DEFAULT 0,
          weighted_holding_period_days REAL,
          source TEXT NOT NULL,
          run_id TEXT NOT NULL,
          pulled_at_utc TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS realized_trade_lots (
          lm_transaction_id TEXT NOT NULL,
          lot_index INTEGER NOT NULL,
          symbol TEXT NOT NULL,
          acquisition_lm_transaction_id TEXT,
          acquisition_date TEXT,
          disposal_date TEXT NOT NULL,
          shares_closed REAL NOT NULL,
          buy_price_effective REAL,
          sell_price REAL,
          gross_proceeds REAL,
          sell_fees REAL,
          net_proceeds REAL,
          realized_cost_basis REAL,
          realized_pnl REAL,
          holding_period_days REAL,
          PRIMARY KEY (lm_transaction_id, lot_index)
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS ix_realized_trade_ledger_date_symbol ON realized_trade_ledger(date, symbol)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS ix_realized_trade_lots_symbol_disposal ON realized_trade_lots(symbol, disposal_date)"
    )

    cur.execute("PRAGMA user_version")
    current_version = cur.fetchone()[0]
    if current_version < 6:
        conn.execute("PRAGMA user_version=6")
        changed = True

    conn.commit()
    return changed


def _run_account_schema_migration(conn: sqlite3.Connection) -> bool:
    cur = conn.cursor()
    changed = False

    statements = [
        """
        CREATE TABLE IF NOT EXISTS portfolio_accounts (
          plaid_account_id TEXT PRIMARY KEY,
          display_name TEXT,
          short_name TEXT,
          institution_name TEXT,
          account_role TEXT NOT NULL,
          account_type TEXT,
          account_subtype TEXT,
          mask TEXT,
          status TEXT,
          include_in_portfolio INTEGER NOT NULL DEFAULT 0,
          include_in_income INTEGER NOT NULL DEFAULT 0,
          include_in_margin INTEGER NOT NULL DEFAULT 0,
          is_primary INTEGER NOT NULL DEFAULT 0,
          first_seen_date TEXT,
          last_seen_utc TEXT,
          source TEXT NOT NULL DEFAULT 'lunchmoney'
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS daily_account_portfolio (
          as_of_date_local TEXT NOT NULL,
          plaid_account_id TEXT NOT NULL,
          account_role TEXT NOT NULL,
          display_name TEXT,
          short_name TEXT,
          market_value REAL,
          cost_basis REAL,
          net_liquidation_value REAL,
          unrealized_pnl REAL,
          unrealized_pct REAL,
          margin_loan_balance REAL,
          ltv_pct REAL,
          projected_monthly_income REAL,
          forward_12m_total REAL,
          portfolio_yield_pct REAL,
          portfolio_yield_on_cost_pct REAL,
          holdings_count INTEGER,
          account_weight_pct REAL,
          income_weight_pct REAL,
          is_primary INTEGER NOT NULL DEFAULT 0,
          built_from_run_id TEXT NOT NULL,
          created_at_utc TEXT NOT NULL,
          PRIMARY KEY (as_of_date_local, plaid_account_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS daily_account_holdings (
          as_of_date_local TEXT NOT NULL,
          plaid_account_id TEXT NOT NULL,
          account_role TEXT NOT NULL,
          account_short_name TEXT,
          symbol TEXT NOT NULL,
          shares REAL,
          cost_basis REAL,
          avg_cost_per_share REAL,
          last_price REAL,
          market_value REAL,
          unrealized_pnl REAL,
          unrealized_pct REAL,
          account_weight_pct REAL,
          portfolio_weight_pct REAL,
          forward_12m_dividend REAL,
          projected_monthly_dividend REAL,
          projected_annual_dividend REAL,
          current_yield_pct REAL,
          yield_on_cost_pct REAL,
          dividends_30d REAL,
          dividends_qtd REAL,
          dividends_ytd REAL,
          PRIMARY KEY (as_of_date_local, plaid_account_id, symbol)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS period_account_summary (
          period_type TEXT NOT NULL,
          period_start_date TEXT NOT NULL,
          period_end_date TEXT NOT NULL,
          is_rolling INTEGER NOT NULL DEFAULT 0,
          plaid_account_id TEXT NOT NULL,
          account_role TEXT NOT NULL,
          short_name TEXT,
          market_value_start REAL,
          market_value_end REAL,
          market_value_delta REAL,
          cost_basis_start REAL,
          cost_basis_end REAL,
          projected_monthly_income_start REAL,
          projected_monthly_income_end REAL,
          projected_monthly_income_delta REAL,
          forward_12m_total_start REAL,
          forward_12m_total_end REAL,
          portfolio_yield_pct_start REAL,
          portfolio_yield_pct_end REAL,
          margin_loan_balance_start REAL,
          margin_loan_balance_end REAL,
          ltv_pct_start REAL,
          ltv_pct_end REAL,
          holdings_count_start INTEGER,
          holdings_count_end INTEGER,
          account_weight_pct_start REAL,
          account_weight_pct_end REAL,
          built_from_run_id TEXT NOT NULL,
          created_at_utc TEXT NOT NULL,
          PRIMARY KEY (period_type, period_start_date, period_end_date, is_rolling, plaid_account_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS period_account_activity (
          period_type TEXT NOT NULL,
          period_start_date TEXT NOT NULL,
          period_end_date TEXT NOT NULL,
          is_rolling INTEGER NOT NULL DEFAULT 0,
          plaid_account_id TEXT NOT NULL,
          account_role TEXT,
          short_name TEXT,
          contributions_total REAL,
          contributions_count INTEGER,
          withdrawals_total REAL,
          withdrawals_count INTEGER,
          dividends_total_received REAL,
          dividends_count INTEGER,
          interest_total_paid REAL,
          trades_total_count INTEGER,
          buy_count INTEGER,
          sell_count INTEGER,
          buys_total REAL,
          sells_total REAL,
          realized_capital_pnl REAL,
          realized_total_return REAL,
          PRIMARY KEY (period_type, period_start_date, period_end_date, is_rolling, plaid_account_id)
        )
        """,
    ]
    for stmt in statements:
        cur.execute(stmt)

    activity_cols = {row[1] for row in cur.execute("PRAGMA table_info(period_account_activity)").fetchall()}
    if activity_cols:
        if "account_role" not in activity_cols:
            cur.execute("ALTER TABLE period_account_activity ADD COLUMN account_role TEXT")
        if "short_name" not in activity_cols:
            cur.execute("ALTER TABLE period_account_activity ADD COLUMN short_name TEXT")
        if "realized_total_return" not in activity_cols:
            cur.execute("ALTER TABLE period_account_activity ADD COLUMN realized_total_return REAL")

    indexes = [
        "CREATE INDEX IF NOT EXISTS ix_portfolio_accounts_role ON portfolio_accounts(account_role, include_in_portfolio)",
        "CREATE INDEX IF NOT EXISTS ix_dap_date_role ON daily_account_portfolio(as_of_date_local DESC, account_role)",
        "CREATE INDEX IF NOT EXISTS ix_dah_symbol_date ON daily_account_holdings(symbol, as_of_date_local DESC)",
        "CREATE INDEX IF NOT EXISTS ix_dah_account_date ON daily_account_holdings(plaid_account_id, as_of_date_local DESC)",
        "CREATE INDEX IF NOT EXISTS ix_pas_period ON period_account_summary(period_type, period_end_date DESC, is_rolling)",
        "CREATE INDEX IF NOT EXISTS ix_paa_period ON period_account_activity(period_type, period_end_date DESC, is_rolling)",
    ]
    for stmt in indexes:
        cur.execute(stmt)

    cur.execute("PRAGMA user_version")
    current_version = cur.fetchone()[0]
    if current_version < 7:
        conn.execute("PRAGMA user_version=7")
        changed = True

    conn.commit()
    return changed

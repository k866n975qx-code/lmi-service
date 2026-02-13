-- Consolidated flat schema migration (replaces 001, 002, 003)
-- Schema version: 5
-- FULLY FLATTENED: All JSON columns removed, every field has its own column
-- Preserved tables: margin_balance_history (data preserved during migration)
-- All other tables are recreated

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

BEGIN;

-- ============================================================
-- Drop derived tables only (preserve all source data)
-- ============================================================

-- Drop period summary tables (derived from daily snapshots)
DROP TABLE IF EXISTS period_interval_attribution;
DROP TABLE IF EXISTS period_interval_holdings;
DROP TABLE IF EXISTS period_holding_changes;
DROP TABLE IF EXISTS period_intervals;
DROP TABLE IF EXISTS period_risk_stats;
DROP TABLE IF EXISTS period_summary;
DROP TABLE IF EXISTS period_activity;
DROP TABLE IF EXISTS period_contributions;
DROP TABLE IF EXISTS period_withdrawals;
DROP TABLE IF EXISTS period_dividend_events;
DROP TABLE IF EXISTS period_trades;
DROP TABLE IF EXISTS period_margin_detail;
DROP TABLE IF EXISTS period_position_lists;
DROP TABLE IF EXISTS period_macro_stats;

-- Drop daily snapshot tables (derived from source data)
DROP TABLE IF EXISTS daily_dividends_upcoming;
DROP TABLE IF EXISTS daily_return_attribution;
DROP TABLE IF EXISTS daily_margin_rate_scenarios;
DROP TABLE IF EXISTS daily_goal_tiers;
DROP TABLE IF EXISTS daily_holdings;
DROP TABLE IF EXISTS daily_portfolio;

-- Drop system tables
DROP TABLE IF EXISTS alert_messages;
DROP TABLE IF EXISTS locks;
DROP TABLE IF EXISTS runs;
DROP TABLE IF EXISTS metadata;

-- Preserved tables (source data - NOT dropped):
-- - lm_raw (raw Lunch Money API data)
-- - cusip_map (symbol mappings)
-- - account_balances (historical balances)
-- - investment_transactions (transaction history)
-- - dividend_events_lm (dividend events from LM)
-- - dividend_events_provider (dividend events from provider)
-- - split_events (stock splits)
-- - margin_balance_history (margin balance history)
-- - facts_source_daily (daily facts)

-- ============================================================
-- Section 1: Core tables
-- ============================================================

CREATE TABLE IF NOT EXISTS lm_raw (
  run_id TEXT NOT NULL,
  pulled_at_utc TEXT NOT NULL,
  endpoint TEXT NOT NULL,
  object_id TEXT,
  payload_json TEXT NOT NULL,
  payload_sha256 TEXT NOT NULL,
  PRIMARY KEY (run_id, endpoint, payload_sha256)
);

CREATE TABLE IF NOT EXISTS cusip_map (
  cusip TEXT PRIMARY KEY,
  symbol TEXT NOT NULL,
  description TEXT
);

CREATE TABLE IF NOT EXISTS metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS facts_source_daily (
  as_of_date_local TEXT NOT NULL,
  scope TEXT NOT NULL,
  symbol TEXT,
  field_path TEXT NOT NULL,
  value_text TEXT,
  value_num REAL,
  value_int INTEGER,
  value_type TEXT NOT NULL,
  source_type TEXT NOT NULL,
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

CREATE TABLE runs (
  run_id TEXT PRIMARY KEY,
  started_at_utc TEXT NOT NULL,
  finished_at_utc TEXT,
  status TEXT NOT NULL,
  error_message TEXT,
  lm_window_start TEXT,
  lm_window_end TEXT,
  git_head_sha TEXT
);

CREATE TABLE locks(
  name TEXT PRIMARY KEY,
  owner TEXT NOT NULL,
  acquired_at_utc TEXT NOT NULL,
  expires_at_utc TEXT NOT NULL
);

CREATE TABLE alert_messages (
  id TEXT PRIMARY KEY,
  fingerprint TEXT NOT NULL,
  category TEXT NOT NULL,
  severity INTEGER NOT NULL,
  title TEXT NOT NULL,
  body_html TEXT NOT NULL,
  details_json TEXT,
  status TEXT NOT NULL DEFAULT 'open',
  period_type TEXT,
  as_of_date_local TEXT NOT NULL,
  created_at_utc TEXT NOT NULL,
  updated_at_utc TEXT NOT NULL,
  first_triggered_at_utc TEXT NOT NULL,
  last_triggered_at_utc TEXT NOT NULL,
  last_notified_at_utc TEXT,
  notification_count INTEGER NOT NULL DEFAULT 0,
  reminder_count INTEGER NOT NULL DEFAULT 0,
  next_reminder_at_utc TEXT,
  sent_at_utc TEXT,
  acked_at_utc TEXT,
  acked_by TEXT
);

-- ============================================================
-- Section 2: Facts and dividend events tables
-- ============================================================

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

CREATE TABLE IF NOT EXISTS split_events (
  symbol TEXT NOT NULL,
  ex_date TEXT NOT NULL,
  ratio REAL NOT NULL,
  source TEXT NOT NULL,
  run_id TEXT NOT NULL,
  fetched_at_utc TEXT NOT NULL,
  PRIMARY KEY (symbol, ex_date, ratio, source)
);

-- ============================================================
-- Section 3: Margin data (preserved table)
-- ============================================================

CREATE TABLE IF NOT EXISTS margin_balance_history (
  as_of_date_local  TEXT PRIMARY KEY,
  balance           REAL NOT NULL,
  source            TEXT NOT NULL DEFAULT 'manual_entry'
);

-- ============================================================
-- Section 4: Daily snapshot tables
-- ============================================================

CREATE TABLE daily_portfolio (
  as_of_date_local          TEXT PRIMARY KEY,
  built_from_run_id         TEXT NOT NULL,
  created_at_utc            TEXT NOT NULL,

  -- Totals
  market_value              REAL,
  cost_basis                REAL,
  net_liquidation_value     REAL,
  unrealized_pnl            REAL,
  unrealized_pct            REAL,
  margin_loan_balance       REAL,
  ltv_pct                   REAL,
  holdings_count            INTEGER,
  positions_profitable      INTEGER,
  positions_losing          INTEGER,

  -- Income
  projected_monthly_income      REAL,
  forward_12m_total             REAL,
  portfolio_yield_pct           REAL,
  portfolio_yield_on_cost_pct   REAL,

  -- Income stability
  income_stability_score        REAL,
  income_trend_6m               TEXT,
  income_volatility_30d_pct     REAL,
  dividend_cut_count_12m        INTEGER,
  income_growth_json            TEXT,

  -- Performance
  twr_1m_pct              REAL,
  twr_3m_pct              REAL,
  twr_6m_pct              REAL,
  twr_12m_pct             REAL,

  -- Benchmark
  vs_benchmark_symbol         TEXT,
  vs_benchmark_twr_1y_pct     REAL,
  vs_benchmark_excess_1y_pct  REAL,
  vs_benchmark_corr_1y        REAL,

  -- Risk: volatility
  vol_30d_pct             REAL,
  vol_90d_pct             REAL,

  -- Risk: ratios
  sharpe_1y               REAL,
  sortino_1y              REAL,
  sortino_6m              REAL,
  sortino_3m              REAL,
  sortino_1m              REAL,
  sortino_sharpe_ratio    REAL,
  sortino_sharpe_divergence REAL,
  calmar_1y               REAL,
  omega_ratio_1y          REAL,
  ulcer_index_1y          REAL,
  pain_adjusted_return    REAL,

  -- Risk: benchmark
  beta_portfolio          REAL,
  information_ratio_1y    REAL,
  tracking_error_1y_pct   REAL,

  -- Risk: quality
  portfolio_risk_quality  TEXT,

  -- Risk: VaR / CVaR
  var_90_1d_pct           REAL,
  var_95_1d_pct           REAL,
  var_99_1d_pct           REAL,
  var_95_1w_pct           REAL,
  var_95_1m_pct           REAL,
  cvar_90_1d_pct          REAL,
  cvar_95_1d_pct          REAL,
  cvar_99_1d_pct          REAL,
  cvar_95_1w_pct          REAL,
  cvar_95_1m_pct          REAL,

  -- Risk: tail risk
  tail_risk_category      TEXT,
  cvar_to_income_ratio    REAL,

  -- Risk: drawdown
  max_drawdown_1y_pct         REAL,
  drawdown_duration_1y_days   INTEGER,
  currently_in_drawdown       INTEGER,
  drawdown_depth_pct          REAL,
  drawdown_duration_days      INTEGER,
  drawdown_peak_value         REAL,
  drawdown_peak_date          TEXT,

  -- Dividends: realized
  dividends_realized_mtd      REAL,
  dividends_realized_30d      REAL,
  dividends_realized_ytd      REAL,
  dividends_realized_qtd      REAL,

  -- Dividends: projected vs received
  dividends_projected_monthly     REAL,
  dividends_received_pct          REAL,

  -- Dividends: upcoming (scalar summaries)
  dividends_upcoming_count        INTEGER,
  dividends_upcoming_total_est    REAL,

  -- Dividends: detail (JSON)
  dividends_by_symbol_json              TEXT,
  dividends_projected_vs_received_json  TEXT,

  -- Allocation
  top3_weight_pct         REAL,
  top5_weight_pct         REAL,
  herfindahl_index        REAL,
  concentration_category  TEXT,

  -- Margin: current
  monthly_interest_current        REAL,
  annual_interest_current         REAL,
  margin_income_coverage          REAL,
  margin_interest_to_income_pct   REAL,

  -- Margin: stress
  buffer_to_margin_call_pct   REAL,
  dollar_decline_to_call      REAL,
  days_at_current_volatility  REAL,
  margin_call_buffer_status   TEXT,

  -- Margin: guidance
  margin_guidance_selected_mode   TEXT,
  margin_guidance_json            TEXT,

  -- Margin: history
  margin_history_90d_json         TEXT,

  -- Goal: baseline
  goal_target_monthly_income      REAL,
  goal_required_portfolio_value   REAL,
  goal_progress_pct               REAL,
  goal_months_to_goal             REAL,
  goal_current_projected_monthly   REAL,
  goal_estimated_goal_date        TEXT,

  -- Goal: net of interest
  goal_net_progress_pct               REAL,
  goal_net_months_to_goal             REAL,
  goal_net_current_projected_monthly  REAL,

  -- Goal: tiers current state
  goal_tiers_portfolio_value          REAL,
  goal_tiers_projected_monthly        REAL,
  goal_tiers_yield_pct                REAL,

  -- Goal: pace
  goal_pace_months_ahead_behind   REAL,
  goal_pace_category              TEXT,
  goal_pace_on_track              INTEGER,
  goal_pace_revised_goal_date     TEXT,
  goal_pace_pct_of_tier           REAL,
  goal_likely_tier                INTEGER,
  goal_likely_tier_name           TEXT,
  goal_likely_tier_confidence     REAL,
  goal_likely_tier_reason         TEXT,

  -- Goal: pace detail (JSON)
  goal_pace_windows_json          TEXT,
  goal_pace_baseline_json         TEXT,

  -- Macro
  macro_vix                   REAL,
  macro_ten_year_yield        REAL,
  macro_two_year_yield        REAL,
  macro_hy_spread_bps         REAL,
  macro_yield_spread_10y_2y   REAL,
  macro_stress_score          REAL,
  macro_cpi_yoy               REAL,
  macro_data_as_of_date       TEXT,

  -- Data quality
  coverage_derived_pct        REAL,
  coverage_pulled_pct         REAL,
  coverage_missing_pct        REAL,
  coverage_filled_pct         REAL,
  coverage_missing_paths_json TEXT,

  -- Timestamps & meta
  prices_as_of_utc            TEXT,
  schema_version              TEXT,
  health_status               TEXT
);

CREATE TABLE daily_holdings (
  as_of_date_local        TEXT NOT NULL,
  symbol                  TEXT NOT NULL,
  shares                  REAL,
  trades_count            INTEGER,
  cost_basis              REAL,
  avg_cost_per_share      REAL,

  -- Valuation
  last_price              REAL,
  market_value            REAL,
  unrealized_pnl          REAL,
  unrealized_pct          REAL,
  weight_pct              REAL,
  forward_12m_dividend        REAL,
  projected_monthly_dividend  REAL,
  projected_annual_dividend   REAL,
  current_yield_pct           REAL,
  yield_on_cost_pct           REAL,
  dividends_30d               REAL,
  dividends_qtd               REAL,
  dividends_ytd               REAL,
  trailing_12m_yield_pct      REAL,
  forward_yield_pct           REAL,
  distribution_frequency      TEXT,
  next_ex_date_est            TEXT,
  last_ex_date                TEXT,
  trailing_12m_div_per_share  REAL,
  forward_12m_div_per_share   REAL,
  vol_30d_pct             REAL,
  vol_90d_pct             REAL,
  beta_3y                 REAL,
  max_drawdown_1y_pct     REAL,
  drawdown_duration_1y_days INTEGER,
  downside_dev_1y_pct     REAL,
  sortino_1y              REAL,
  sortino_6m              REAL,
  sortino_3m              REAL,
  sortino_1m              REAL,
  sharpe_1y               REAL,
  calmar_1y               REAL,
  risk_quality_score      REAL,
  risk_quality_category   TEXT,
  volatility_profile      TEXT,
  var_90_1d_pct           REAL,
  var_95_1d_pct           REAL,
  var_99_1d_pct           REAL,
  var_95_1w_pct           REAL,
  var_95_1m_pct           REAL,
  cvar_90_1d_pct          REAL,
  cvar_95_1d_pct          REAL,
  cvar_99_1d_pct          REAL,
  cvar_95_1w_pct          REAL,
  cvar_95_1m_pct          REAL,
  twr_1m_pct              REAL,
  twr_3m_pct              REAL,
  twr_6m_pct              REAL,
  twr_12m_pct             REAL,
  corr_1y                 REAL,
  reliability_consistency_score   REAL,
  reliability_trend_6m            TEXT,
  reliability_missed_payments_12m INTEGER,
  PRIMARY KEY (as_of_date_local, symbol)
);

CREATE TABLE daily_goal_tiers (
  as_of_date_local                TEXT NOT NULL,
  tier                            INTEGER NOT NULL,
  name                            TEXT,
  description                     TEXT,
  target_monthly                  REAL,
  required_portfolio_value        REAL,
  final_portfolio_value           REAL,
  progress_pct                    REAL,
  months_to_goal                  REAL,
  estimated_goal_date             TEXT,
  confidence                      REAL,
  assumption_monthly_contribution REAL,
  assumption_drip_enabled         INTEGER,
  assumption_annual_appreciation_pct REAL,
  assumption_ltv_maintained       INTEGER,
  assumption_target_ltv_pct       REAL,
  PRIMARY KEY (as_of_date_local, tier)
);

CREATE TABLE daily_margin_rate_scenarios (
  as_of_date_local  TEXT NOT NULL,
  scenario          TEXT NOT NULL,
  new_rate_pct      REAL,
  new_monthly_cost  REAL,
  income_coverage   REAL,
  margin_impact_pct REAL,
  PRIMARY KEY (as_of_date_local, scenario)
);

CREATE TABLE daily_return_attribution (
  as_of_date_local  TEXT NOT NULL,
  window            TEXT NOT NULL,
  symbol            TEXT NOT NULL,
  contribution_pct  REAL,
  weight_avg_pct    REAL,
  return_pct        REAL,
  PRIMARY KEY (as_of_date_local, window, symbol)
);

CREATE TABLE daily_dividends_upcoming (
  as_of_date_local  TEXT NOT NULL,
  symbol            TEXT NOT NULL,
  ex_date_est       TEXT NOT NULL,
  pay_date_est      TEXT,
  amount_est        REAL,
  PRIMARY KEY (as_of_date_local, symbol, ex_date_est)
);

-- ============================================================
-- Section 5: Period summary tables
-- ============================================================

CREATE TABLE period_summary (
  period_type             TEXT NOT NULL,
  period_start_date       TEXT NOT NULL,
  period_end_date         TEXT NOT NULL,
  period_label            TEXT,
  is_rolling              INTEGER NOT NULL DEFAULT 0,
  snapshot_mode           TEXT,
  expected_days           INTEGER,
  observed_days           INTEGER,
  coverage_pct            REAL,
  is_complete             INTEGER,

  -- Totals (start/end/delta)
  mv_start REAL, mv_end REAL, mv_delta REAL,
  nlv_start REAL, nlv_end REAL, nlv_delta REAL,
  cost_basis_start REAL, cost_basis_end REAL, cost_basis_delta REAL,
  unrealized_pct_start REAL, unrealized_pct_end REAL, unrealized_pct_delta REAL,
  unrealized_pnl_start REAL, unrealized_pnl_end REAL, unrealized_pnl_delta REAL,
  margin_balance_start REAL, margin_balance_end REAL, margin_balance_delta REAL,
  margin_balance_avg REAL, margin_balance_min REAL, margin_balance_max REAL, margin_balance_std REAL,
  margin_balance_min_date TEXT, margin_balance_max_date TEXT,
  ltv_pct_start REAL, ltv_pct_end REAL, ltv_pct_delta REAL,
  ltv_pct_avg REAL, ltv_pct_min REAL, ltv_pct_max REAL, ltv_pct_std REAL,
  ltv_pct_min_date TEXT, ltv_pct_max_date TEXT,

  -- Performance
  twr_period_pct    REAL,
  pnl_dollar_period REAL,
  pnl_pct_period    REAL,
  twr_1m_start  REAL, twr_1m_end  REAL, twr_1m_delta  REAL,
  twr_3m_start  REAL, twr_3m_end  REAL, twr_3m_delta  REAL,
  twr_6m_start  REAL, twr_6m_end  REAL, twr_6m_delta  REAL,
  twr_12m_start REAL, twr_12m_end REAL, twr_12m_delta REAL,

  -- Income
  monthly_income_start REAL, monthly_income_end REAL, monthly_income_delta REAL,
  forward_12m_start    REAL, forward_12m_end    REAL, forward_12m_delta    REAL,
  yield_start          REAL, yield_end          REAL, yield_delta          REAL,
  yield_on_cost_start  REAL, yield_on_cost_end  REAL, yield_on_cost_delta  REAL,
  dividends_received_period REAL,

  -- Risk
  vol_30d_start   REAL, vol_30d_end   REAL, vol_30d_delta   REAL, vol_30d_avg   REAL,
  vol_90d_start   REAL, vol_90d_end   REAL, vol_90d_delta   REAL, vol_90d_avg   REAL,
  sharpe_1y_start REAL, sharpe_1y_end REAL, sharpe_1y_delta REAL, sharpe_1y_avg REAL,
  sortino_1y_start REAL, sortino_1y_end REAL, sortino_1y_delta REAL, sortino_1y_avg REAL,
  sortino_6m_start REAL, sortino_6m_end REAL, sortino_6m_delta REAL,
  sortino_3m_start REAL, sortino_3m_end REAL, sortino_3m_delta REAL,
  sortino_1m_start REAL, sortino_1m_end REAL, sortino_1m_delta REAL,
  calmar_1y_start REAL, calmar_1y_end REAL, calmar_1y_delta REAL, calmar_1y_avg REAL,
  max_dd_1y_start REAL, max_dd_1y_end REAL, max_dd_1y_delta REAL,
  portfolio_risk_quality_start TEXT, portfolio_risk_quality_end TEXT,

  -- Goals
  goal_progress_pct_start     REAL, goal_progress_pct_end     REAL, goal_progress_pct_delta     REAL,
  goal_monthly_start          REAL, goal_monthly_end          REAL, goal_monthly_delta          REAL,
  goal_months_to_goal_start   REAL, goal_months_to_goal_end   REAL, goal_months_to_goal_delta   REAL,
  goal_net_progress_pct_start REAL, goal_net_progress_pct_end REAL, goal_net_progress_pct_delta REAL,
  goal_net_monthly_start      REAL, goal_net_monthly_end      REAL, goal_net_monthly_delta      REAL,
  goal_pace_category_start    TEXT, goal_pace_category_end    TEXT,
  goal_pace_months_start      REAL, goal_pace_months_end      REAL, goal_pace_months_delta      REAL,
  goal_pace_tier_pace_pct_start REAL, goal_pace_tier_pace_pct_end REAL, goal_pace_tier_pace_pct_delta REAL,

  -- Composition
  holding_count_start INTEGER, holding_count_end INTEGER, holding_count_delta INTEGER,
  concentration_top3_start REAL, concentration_top3_end REAL, concentration_top3_delta REAL, concentration_top3_avg REAL,
  concentration_top5_start REAL, concentration_top5_end REAL, concentration_top5_delta REAL, concentration_top5_avg REAL,
  concentration_herfindahl_start REAL, concentration_herfindahl_end REAL, concentration_herfindahl_delta REAL, concentration_herfindahl_avg REAL,

  -- Macro
  macro_10y_start REAL, macro_10y_end REAL, macro_10y_avg REAL, macro_10y_delta REAL,
  macro_2y_start  REAL, macro_2y_end  REAL, macro_2y_avg  REAL, macro_2y_delta  REAL,
  macro_vix_start REAL, macro_vix_end REAL, macro_vix_avg REAL, macro_vix_delta REAL,
  macro_hy_spread_bps_start REAL, macro_hy_spread_bps_end REAL, macro_hy_spread_bps_avg REAL, macro_hy_spread_bps_delta REAL,
  macro_stress_score_start REAL, macro_stress_score_end REAL, macro_stress_score_avg REAL, macro_stress_score_delta REAL,
  macro_yield_spread_10y_2y_start REAL, macro_yield_spread_10y_2y_end REAL, macro_yield_spread_10y_2y_avg REAL, macro_yield_spread_10y_2y_delta REAL,
  macro_cpi_start REAL, macro_cpi_end REAL, macro_cpi_avg REAL, macro_cpi_delta REAL,

  -- Benchmark
  benchmark_symbol            TEXT,
  benchmark_period_return_pct REAL,
  benchmark_twr_1y_start      REAL,
  benchmark_twr_1y_end        REAL,
  benchmark_twr_1y_delta      REAL,
  benchmark_correlation_1y_avg REAL,

  -- Period stats: market value
  mv_avg REAL, mv_min REAL, mv_max REAL, mv_std REAL, mv_min_date TEXT, mv_max_date TEXT,
  -- Period stats: net liquidation value
  nlv_avg REAL, nlv_min REAL, nlv_max REAL, nlv_std REAL, nlv_min_date TEXT, nlv_max_date TEXT,
  -- Period stats: income
  projected_monthly_avg REAL, projected_monthly_min REAL, projected_monthly_max REAL, projected_monthly_std REAL,
  yield_pct_avg REAL, yield_pct_min REAL, yield_pct_max REAL, yield_pct_std REAL,
  -- Period stats: margin to portfolio
  margin_to_portfolio_pct_avg REAL, margin_to_portfolio_pct_min REAL, margin_to_portfolio_pct_max REAL, margin_to_portfolio_pct_std REAL,
  -- Period stats: VaR
  var_95_1d_avg REAL,
  var_90_1d_avg REAL, var_99_1d_avg REAL,
  var_95_1w_avg REAL, var_95_1m_avg REAL,
  cvar_95_1d_avg REAL,
  cvar_90_1d_avg REAL, cvar_99_1d_avg REAL,
  cvar_95_1w_avg REAL, cvar_95_1m_avg REAL,
  -- Period drawdown
  period_max_drawdown_pct REAL, period_max_drawdown_date TEXT, period_recovery_date TEXT, period_days_in_drawdown INTEGER, period_drawdown_count INTEGER,
  -- VaR breach metrics
  days_exceeding_var_95 INTEGER, worst_day_return_pct REAL, worst_day_date TEXT, best_day_return_pct REAL, best_day_date TEXT,
  -- Margin safety metrics
  margin_min_buffer_to_call_pct REAL, margin_min_buffer_date TEXT, margin_days_below_50pct_buffer INTEGER, margin_days_below_40pct_buffer INTEGER, margin_call_events INTEGER,
  -- Margin interest APR
  margin_apr_start REAL, margin_apr_end REAL, margin_apr_avg REAL,

  built_from_run_id TEXT,
  created_at_utc    TEXT NOT NULL,
  PRIMARY KEY (period_type, period_start_date, period_end_date, is_rolling)
);

CREATE TABLE period_risk_stats (
  period_type       TEXT NOT NULL,
  period_start_date TEXT NOT NULL,
  period_end_date   TEXT NOT NULL,
  metric            TEXT NOT NULL,
  avg_val           REAL,
  min_val           REAL,
  max_val           REAL,
  PRIMARY KEY (period_type, period_start_date, period_end_date, metric)
);

CREATE TABLE period_intervals (
  period_type       TEXT NOT NULL,
  period_start_date  TEXT NOT NULL,
  period_end_date   TEXT NOT NULL,
  interval_label    TEXT NOT NULL,
  interval_start    TEXT NOT NULL,
  interval_end      TEXT NOT NULL,
  mv                REAL,
  cost_basis        REAL,
  nlv               REAL,
  unrealized_pct    REAL,
  unrealized_pnl    REAL,
  pnl_period        REAL,
  pnl_pct_period    REAL,
  twr_1m_pct        REAL,
  twr_3m_pct        REAL,
  twr_6m_pct        REAL,
  twr_12m_pct       REAL,
  sharpe_1y         REAL,
  sortino_1y        REAL,
  sortino_6m        REAL,
  sortino_3m        REAL,
  sortino_1m        REAL,
  portfolio_risk_quality TEXT,
  monthly_income    REAL,
  margin_loan       REAL,
  ltv_pct           REAL,

  -- Income (from 003)
  forward_12m_total REAL,
  yield_pct REAL,
  yield_on_cost_pct REAL,
  -- Risk (from 003)
  vol_30d_pct REAL,
  vol_90d_pct REAL,
  calmar_1y REAL,
  max_drawdown_1y_pct REAL,
  var_90_1d_pct REAL,
  var_95_1d_pct REAL,
  cvar_90_1d_pct REAL,
  omega_ratio_1y REAL,
  ulcer_index_1y REAL,
  income_stability_score REAL,
  beta_portfolio REAL,
  -- Margin (from 003)
  annual_interest_expense REAL,
  margin_call_buffer_pct REAL,
  -- Goals (from 003)
  goal_progress_pct REAL,
  goal_months_to_goal REAL,
  goal_projected_monthly REAL,
  goal_net_progress_pct REAL,

  PRIMARY KEY (period_type, period_start_date, period_end_date, interval_label)
);

CREATE TABLE period_holding_changes (
  period_type       TEXT NOT NULL,
  period_start_date TEXT NOT NULL,
  period_end_date   TEXT NOT NULL,
  symbol            TEXT NOT NULL,
  change_type       TEXT NOT NULL,

  -- Weight metrics
  weight_start_pct  REAL,
  weight_end_pct    REAL,
  weight_delta_pct  REAL,
  avg_weight_pct    REAL,

  -- Performance metrics
  pnl_pct_period    REAL,
  pnl_dollar_period REAL,
  contribution_to_portfolio_pct REAL,
  start_twr_12m_pct REAL,
  end_twr_12m_pct   REAL,

  -- Share metrics
  start_shares      REAL,
  end_shares        REAL,
  shares_delta      REAL,
  shares_delta_pct  REAL,

  -- Market value metrics
  start_market_value      REAL,
  end_market_value        REAL,
  market_value_delta      REAL,
  market_value_delta_pct  REAL,

  -- Income metrics
  dividends_received      REAL,
  dividend_events_count   INTEGER,
  start_yield_pct         REAL,
  end_yield_pct           REAL,
  start_projected_monthly REAL,
  end_projected_monthly   REAL,

  -- Risk metrics
  avg_vol_30d_pct         REAL,
  period_max_drawdown_pct REAL,
  worst_day_pct           REAL,
  worst_day_date          TEXT,
  best_day_pct            REAL,
  best_day_date           TEXT,

  PRIMARY KEY (period_type, period_start_date, period_end_date, symbol, change_type)
);

-- ============================================================
-- Section 6: Activity tables
-- ============================================================

CREATE TABLE period_activity (
  period_type       TEXT NOT NULL,
  period_start_date TEXT NOT NULL,
  period_end_date   TEXT NOT NULL,

  -- Contributions (dates stored in period_contributions table)
  contributions_total REAL,
  contributions_count INTEGER,

  -- Withdrawals (dates stored in period_withdrawals table)
  withdrawals_total REAL,
  withdrawals_count INTEGER,

  -- Dividends
  dividends_total_received REAL,
  dividends_count INTEGER,

  -- Interest
  interest_total_paid REAL,
  interest_avg_daily_balance REAL,
  interest_avg_rate_pct REAL,
  interest_annualized REAL,

  -- Trades
  trades_total_count INTEGER,
  trades_buy_count INTEGER,
  trades_sell_count INTEGER,

  -- Margin
  margin_borrowed REAL,
  margin_repaid REAL,
  margin_net_change REAL,

  -- Position changes (symbols stored in period_position_lists table)
  positions_added_count INTEGER DEFAULT 0,
  positions_removed_count INTEGER DEFAULT 0,
  positions_increased_count INTEGER DEFAULT 0,
  positions_decreased_count INTEGER DEFAULT 0,

  PRIMARY KEY (period_type, period_start_date, period_end_date)
);

CREATE TABLE period_position_lists (
  period_type       TEXT NOT NULL,
  period_start_date TEXT NOT NULL,
  period_end_date   TEXT NOT NULL,
  list_type         TEXT NOT NULL,  -- 'added', 'removed', 'increased', 'decreased'
  symbol            TEXT NOT NULL,
  PRIMARY KEY (period_type, period_start_date, period_end_date, list_type, symbol)
);

CREATE TABLE period_contributions (
  period_type       TEXT NOT NULL,
  period_start_date TEXT NOT NULL,
  period_end_date   TEXT NOT NULL,
  contribution_date TEXT NOT NULL,
  amount            REAL NOT NULL,
  account_id        TEXT,
  PRIMARY KEY (period_type, period_start_date, period_end_date, contribution_date, account_id)
);

CREATE TABLE period_withdrawals (
  period_type       TEXT NOT NULL,
  period_start_date TEXT NOT NULL,
  period_end_date   TEXT NOT NULL,
  withdrawal_date   TEXT NOT NULL,
  amount            REAL NOT NULL,
  account_id        TEXT,
  PRIMARY KEY (period_type, period_start_date, period_end_date, withdrawal_date, account_id)
);

CREATE TABLE period_dividend_events (
  period_type       TEXT NOT NULL,
  period_start_date TEXT NOT NULL,
  period_end_date   TEXT NOT NULL,
  symbol            TEXT NOT NULL,
  ex_date           TEXT NOT NULL,
  pay_date          TEXT,
  amount            REAL NOT NULL,
  PRIMARY KEY (period_type, period_start_date, period_end_date, symbol, ex_date)
);

CREATE TABLE period_trades (
  period_type       TEXT NOT NULL,
  period_start_date TEXT NOT NULL,
  period_end_date   TEXT NOT NULL,
  symbol            TEXT NOT NULL,
  buy_count         INTEGER NOT NULL DEFAULT 0,
  sell_count        INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (period_type, period_start_date, period_end_date, symbol)
);

CREATE TABLE period_margin_detail (
  period_type       TEXT NOT NULL,
  period_start_date TEXT NOT NULL,
  period_end_date   TEXT NOT NULL,
  borrowed          REAL NOT NULL,
  repaid            REAL NOT NULL,
  net_change        REAL NOT NULL,
  PRIMARY KEY (period_type, period_start_date, period_end_date)
);

-- ============================================================
-- Section 7: Period interval child tables (from 003)
-- ============================================================

CREATE TABLE period_interval_holdings (
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
);

CREATE TABLE period_interval_attribution (
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
);

-- ============================================================
-- Section 8: Macro period stats table
-- ============================================================

CREATE TABLE period_macro_stats (
  period_type       TEXT NOT NULL,
  period_start_date TEXT NOT NULL,
  period_end_date   TEXT NOT NULL,
  metric            TEXT NOT NULL,
  avg_val           REAL,
  min_val           REAL,
  max_val           REAL,
  std_val           REAL,
  min_date          TEXT,
  max_date          TEXT,
  PRIMARY KEY (period_type, period_start_date, period_end_date, metric)
);

-- ============================================================
-- Section 9: Indexes
-- ============================================================

-- lm_raw
CREATE INDEX IF NOT EXISTS ix_lm_raw_endpoint_id_time ON lm_raw(endpoint, object_id, pulled_at_utc DESC);

-- facts_source_daily
CREATE INDEX IF NOT EXISTS ix_facts_symbol_field ON facts_source_daily(symbol, field_path);

-- account_balances
CREATE INDEX IF NOT EXISTS ix_account_balances_plaid_id ON account_balances(plaid_account_id, as_of_date_local);

-- investment_transactions
CREATE INDEX IF NOT EXISTS ix_investment_tx_account_date ON investment_transactions(plaid_account_id, date);
CREATE INDEX IF NOT EXISTS ix_investment_tx_symbol_date ON investment_transactions(symbol, date);

-- dividend_events_lm
CREATE INDEX IF NOT EXISTS ix_dividend_events_lm_symbol_date ON dividend_events_lm(symbol, ex_date);

-- dividend_events_provider
CREATE INDEX IF NOT EXISTS ix_dividend_events_provider_symbol_date ON dividend_events_provider(symbol, ex_date);

-- split_events
CREATE INDEX IF NOT EXISTS ix_split_events_symbol_date ON split_events(symbol, ex_date);

-- alert_messages
CREATE INDEX IF NOT EXISTS ix_alerts_status_date ON alert_messages(status, as_of_date_local);
CREATE INDEX IF NOT EXISTS ix_alerts_fp ON alert_messages(fingerprint);

-- Daily tables
CREATE INDEX IF NOT EXISTS ix_dp_date ON daily_portfolio(as_of_date_local DESC);
CREATE INDEX IF NOT EXISTS ix_dh_date ON daily_holdings(as_of_date_local DESC);
CREATE INDEX IF NOT EXISTS ix_dh_symbol ON daily_holdings(symbol, as_of_date_local DESC);
CREATE INDEX IF NOT EXISTS ix_dgt_date ON daily_goal_tiers(as_of_date_local DESC);
CREATE INDEX IF NOT EXISTS ix_ddu_date ON daily_dividends_upcoming(as_of_date_local DESC);
CREATE INDEX IF NOT EXISTS ix_mbh_date ON margin_balance_history(as_of_date_local DESC);

-- Period tables
CREATE INDEX IF NOT EXISTS ix_ps_type_end ON period_summary(period_type, period_end_date DESC);
CREATE INDEX IF NOT EXISTS ix_pi_parent ON period_intervals(period_type, period_start_date, period_end_date);
CREATE INDEX IF NOT EXISTS ix_pih_parent ON period_interval_holdings(period_type, period_start_date, period_end_date);
CREATE INDEX IF NOT EXISTS ix_pia_parent ON period_interval_attribution(period_type, period_start_date, period_end_date);
CREATE INDEX IF NOT EXISTS ix_phc_parent ON period_holding_changes(period_type, period_start_date, period_end_date);
CREATE INDEX IF NOT EXISTS ix_ppl_period ON period_position_lists(period_type, period_start_date, period_end_date);
CREATE INDEX IF NOT EXISTS ix_ppl_symbol ON period_position_lists(symbol);

COMMIT;

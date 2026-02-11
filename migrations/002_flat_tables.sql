-- Flat schema migration: 11 new tables (daily + period + margin_balance_history)
-- Legacy snapshot_daily_current and snapshots are dropped in app/db.py migrate() (Phase 18).

-- ============================================================
-- Daily portfolio-level snapshot (one row per date)
-- ============================================================
CREATE TABLE IF NOT EXISTS daily_portfolio (
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

-- ============================================================
-- Per-symbol per-date holdings
-- ============================================================
CREATE TABLE IF NOT EXISTS daily_holdings (
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

-- ============================================================
-- Goal tiers per day (always 6 tiers)
-- ============================================================
CREATE TABLE IF NOT EXISTS daily_goal_tiers (
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

-- ============================================================
-- Margin rate shock scenarios per day
-- ============================================================
CREATE TABLE IF NOT EXISTS daily_margin_rate_scenarios (
  as_of_date_local  TEXT NOT NULL,
  scenario          TEXT NOT NULL,
  new_rate_pct      REAL,
  new_monthly_cost  REAL,
  income_coverage   REAL,
  margin_impact_pct REAL,
  PRIMARY KEY (as_of_date_local, scenario)
);

-- ============================================================
-- Return attribution per window per symbol per day
-- ============================================================
CREATE TABLE IF NOT EXISTS daily_return_attribution (
  as_of_date_local  TEXT NOT NULL,
  window            TEXT NOT NULL,
  symbol            TEXT NOT NULL,
  contribution_pct  REAL,
  weight_avg_pct    REAL,
  return_pct        REAL,
  PRIMARY KEY (as_of_date_local, window, symbol)
);

-- ============================================================
-- Upcoming dividend events per day
-- ============================================================
CREATE TABLE IF NOT EXISTS daily_dividends_upcoming (
  as_of_date_local  TEXT NOT NULL,
  symbol            TEXT NOT NULL,
  ex_date_est       TEXT NOT NULL,
  pay_date_est      TEXT,
  amount_est        REAL,
  PRIMARY KEY (as_of_date_local, symbol, ex_date_est)
);

-- ============================================================
-- Historical margin balance values (from margin balances.md)
-- ============================================================
CREATE TABLE IF NOT EXISTS margin_balance_history (
  as_of_date_local  TEXT PRIMARY KEY,
  balance           REAL NOT NULL,
  source            TEXT NOT NULL DEFAULT 'manual_entry'
);

-- ============================================================
-- Period summaries (replaces snapshots table)
-- ============================================================
CREATE TABLE IF NOT EXISTS period_summary (
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
  ltv_pct_start REAL, ltv_pct_end REAL, ltv_pct_delta REAL,

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
  vol_30d_start   REAL, vol_30d_end   REAL, vol_30d_delta   REAL,
  vol_90d_start   REAL, vol_90d_end   REAL, vol_90d_delta   REAL,
  sharpe_1y_start REAL, sharpe_1y_end REAL, sharpe_1y_delta REAL,
  sortino_1y_start REAL, sortino_1y_end REAL, sortino_1y_delta REAL,
  sortino_6m_start REAL, sortino_6m_end REAL, sortino_6m_delta REAL,
  sortino_3m_start REAL, sortino_3m_end REAL, sortino_3m_delta REAL,
  sortino_1m_start REAL, sortino_1m_end REAL, sortino_1m_delta REAL,
  calmar_1y_start REAL, calmar_1y_end REAL, calmar_1y_delta REAL,
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

  -- Composition
  holding_count_start INTEGER, holding_count_end INTEGER, holding_count_delta INTEGER,
  concentration_top5_start REAL, concentration_top5_end REAL, concentration_top5_delta REAL,

  -- Macro
  macro_10y_start REAL, macro_10y_end REAL, macro_10y_avg REAL, macro_10y_delta REAL,
  macro_2y_start  REAL, macro_2y_end  REAL, macro_2y_avg  REAL, macro_2y_delta  REAL,
  macro_vix_start REAL, macro_vix_end REAL, macro_vix_avg REAL, macro_vix_delta REAL,
  macro_cpi_start REAL, macro_cpi_end REAL, macro_cpi_avg REAL, macro_cpi_delta REAL,

  -- Benchmark
  benchmark_symbol            TEXT,
  benchmark_period_return_pct REAL,
  benchmark_twr_1y_start      REAL,
  benchmark_twr_1y_end        REAL,
  benchmark_twr_1y_delta      REAL,

  built_from_run_id TEXT,
  created_at_utc    TEXT NOT NULL,
  PRIMARY KEY (period_type, period_start_date, period_end_date)
);

-- ============================================================
-- Min/avg/max risk stats per period
-- ============================================================
CREATE TABLE IF NOT EXISTS period_risk_stats (
  period_type       TEXT NOT NULL,
  period_start_date TEXT NOT NULL,
  period_end_date   TEXT NOT NULL,
  metric            TEXT NOT NULL,
  avg_val           REAL,
  min_val           REAL,
  max_val           REAL,
  PRIMARY KEY (period_type, period_start_date, period_end_date, metric)
);

-- ============================================================
-- Sub-period interval rows
-- ============================================================
CREATE TABLE IF NOT EXISTS period_intervals (
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
  PRIMARY KEY (period_type, period_start_date, period_end_date, interval_label)
);

-- ============================================================
-- Holdings changed per period
-- ============================================================
CREATE TABLE IF NOT EXISTS period_holding_changes (
  period_type       TEXT NOT NULL,
  period_start_date TEXT NOT NULL,
  period_end_date   TEXT NOT NULL,
  symbol            TEXT NOT NULL,
  change_type       TEXT NOT NULL,
  weight_start_pct  REAL,
  weight_end_pct    REAL,
  weight_delta_pct  REAL,
  pnl_pct_period    REAL,
  pnl_dollar_period REAL,
  PRIMARY KEY (period_type, period_start_date, period_end_date, symbol, change_type)
);

-- ============================================================
-- Indexes
-- ============================================================
CREATE INDEX IF NOT EXISTS ix_dp_date ON daily_portfolio(as_of_date_local DESC);
CREATE INDEX IF NOT EXISTS ix_dh_date ON daily_holdings(as_of_date_local DESC);
CREATE INDEX IF NOT EXISTS ix_dh_symbol ON daily_holdings(symbol, as_of_date_local DESC);
CREATE INDEX IF NOT EXISTS ix_dgt_date ON daily_goal_tiers(as_of_date_local DESC);
CREATE INDEX IF NOT EXISTS ix_ddu_date ON daily_dividends_upcoming(as_of_date_local DESC);
CREATE INDEX IF NOT EXISTS ix_ps_type_end ON period_summary(period_type, period_end_date DESC);
CREATE INDEX IF NOT EXISTS ix_pi_parent ON period_intervals(period_type, period_start_date, period_end_date);
CREATE INDEX IF NOT EXISTS ix_phc_parent ON period_holding_changes(period_type, period_start_date, period_end_date);
CREATE INDEX IF NOT EXISTS ix_mbh_date ON margin_balance_history(as_of_date_local DESC);

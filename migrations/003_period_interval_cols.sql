-- Enrich period_intervals with income, performance, risk, margin, goals columns;
-- add period_interval_holdings and period_interval_attribution child tables.

-- ============================================================
-- New columns on period_intervals
-- ============================================================
-- Income (3)
ALTER TABLE period_intervals ADD COLUMN forward_12m_total REAL;
ALTER TABLE period_intervals ADD COLUMN yield_pct REAL;
ALTER TABLE period_intervals ADD COLUMN yield_on_cost_pct REAL;
-- Performance (1)
ALTER TABLE period_intervals ADD COLUMN twr_6m_pct REAL;
-- Risk (11)
ALTER TABLE period_intervals ADD COLUMN vol_30d_pct REAL;
ALTER TABLE period_intervals ADD COLUMN vol_90d_pct REAL;
ALTER TABLE period_intervals ADD COLUMN calmar_1y REAL;
ALTER TABLE period_intervals ADD COLUMN max_drawdown_1y_pct REAL;
ALTER TABLE period_intervals ADD COLUMN var_90_1d_pct REAL;
ALTER TABLE period_intervals ADD COLUMN var_95_1d_pct REAL;
ALTER TABLE period_intervals ADD COLUMN cvar_90_1d_pct REAL;
ALTER TABLE period_intervals ADD COLUMN omega_ratio_1y REAL;
ALTER TABLE period_intervals ADD COLUMN ulcer_index_1y REAL;
ALTER TABLE period_intervals ADD COLUMN income_stability_score REAL;
ALTER TABLE period_intervals ADD COLUMN beta_portfolio REAL;
-- Margin (2)
ALTER TABLE period_intervals ADD COLUMN annual_interest_expense REAL;
ALTER TABLE period_intervals ADD COLUMN margin_call_buffer_pct REAL;
-- Goals (4)
ALTER TABLE period_intervals ADD COLUMN goal_progress_pct REAL;
ALTER TABLE period_intervals ADD COLUMN goal_months_to_goal REAL;
ALTER TABLE period_intervals ADD COLUMN goal_projected_monthly REAL;
ALTER TABLE period_intervals ADD COLUMN goal_net_progress_pct REAL;

-- ============================================================
-- Per-interval per-symbol holdings
-- ============================================================
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
);

-- ============================================================
-- Per-interval per-window attribution summary
-- ============================================================
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
);

-- ============================================================
-- Indexes
-- ============================================================
CREATE INDEX IF NOT EXISTS ix_pih_parent ON period_interval_holdings(period_type, period_start_date, period_end_date);
CREATE INDEX IF NOT EXISTS ix_pia_parent ON period_interval_attribution(period_type, period_start_date, period_end_date);

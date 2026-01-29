from __future__ import annotations

# Thresholds and knobs (unit: percent unless noted)
MARGIN_LTV_CRITICAL = 45.0
MARGIN_LTV_RED = 40.0
MARGIN_LTV_YELLOW = 30.0
MARGIN_COVERAGE_MIN = 2.0
MARGIN_INTEREST_INCOME_WARN_PCT = 15.0

INCOME_CONCENTRATION_WARNING = 80.0  # top3 income %  (was 65; realistic for <10 holdings)
INCOME_SINGLE_SOURCE_WARNING = 35.0  # single source % (was 30)
INCOME_MISS_CRITICAL = 0.5           # 50% of expected by day threshold
INCOME_MISS_DAY_THRESHOLD = 20       # after this day: apply miss rule
INCOME_STABILITY_MIN = 0.30          # (was 0.70; 0.00 normal with <6mo data)
INCOME_VOLATILITY_30D_WARN = 50.0    # (was 15; sparse dividends inflate this metric)

DIVIDEND_CUT_THRESHOLD = 0.10        # 10%
MISSING_DIVIDEND_GRACE_DAYS = 3
MISSING_DIVIDEND_MIN_PAYMENTS = 1   # require at least N payouts before missing alerts

VOL_EXPANSION_RATIO = 1.5
PORTFOLIO_VOL_WARNING = 18.0          # (was 12; too tight for vol-product portfolios)
VIX_WARNING = 25.0
VIX_CRITICAL = 30.0
TREASURY_SPIKE = 0.5
HY_SPREAD_WARNING = 400.0
HY_SPREAD_CRITICAL = 500.0

PORTFOLIO_SORTINO_MIN = 0.5
PORTFOLIO_SORTINO_DROP = 0.3
PORTFOLIO_SORTINO_SHARPE_GAP = 0.2
POSITION_SORTINO_NEGATIVE = 0.0

TAIL_RISK_CVAR_1D_CRITICAL = -4.0
TAIL_RISK_CVAR_1W_CRITICAL = -10.0
TAIL_RISK_INCOME_RATIO = 20.0

MARGIN_BUFFER_WARNING = 40.0
MARGIN_BUFFER_CRITICAL = 20.0

POSITION_LOSS_WARNING = -15.0
POSITION_LOSS_CRITICAL = -20.0
POSITION_LOSS_SEVERE = -25.0
SINGLE_POSITION_MAX = 30.0            # (was 25; allows intentional overweight)

GOAL_SLIPPAGE_MONTHS = 3
GOAL_REQUIRED_INVESTMENT_DELTA = 10000.0
YIELD_COMPRESSION_WARNING = 1.0      # percentage point drop
INCOME_BUNCHING_WEEK_PCT = 70.0      # % of month in one ISO week (was 50; normal w/ few holdings)
EXTENDED_DRAWDOWN_DAYS = 180
MAX_DRAWDOWN_WARNING = -25.0          # (was -20; too sensitive for vol products)

# Spam control
MAX_NOTIFICATIONS_PER_DAY = 20
MIN_HOURS_BETWEEN_SAME_ALERT = 24
ESCALATION_REMINDER_HOURS = [24, 72]

# Milestones
MILESTONE_NET_VALUES = [15000, 20000, 25000]
MILESTONE_MONTHLY_INCOME = [250, 300, 500]
MILESTONE_PROGRESS_PCT = [15, 20, 25]

# Pace alerts
PACE_SLIPPAGE_PCT = 90.0            # alert if pace < this % of tier pace
PACE_SLIPPAGE_CONSECUTIVE = 2       # require N consecutive snapshots below threshold
PACE_SURPLUS_DECLINE_PCT = 20.0     # alert if surplus drops by this % week-over-week
PACE_GOAL_MILESTONES = [25, 50, 75, 90]  # goal progress milestones
PACE_ACCEL_MONTHS = 0.5             # alert on pace change >= this many months
VOL_SPIKE_RATIO = 1.3               # alert if vol increases by 30%+

# Alert grouping
ALERT_GROUPING_ENABLED = True
ALERT_GROUPING_MIN_SIZE = 2         # min alerts in same category to trigger grouping

from typing import Tuple, List

# V4 paths (legacy) and V5 paths (new schema)
CRITICAL_PATHS_V4 = [
    "as_of",
    "as_of_utc",
    "as_of_date_local",
]

CRITICAL_PATHS_V5 = [
    "timestamps.portfolio_data_as_of_utc",
    "timestamps.portfolio_data_as_of_local",
]

# Check both V4 and V5 - at least one must have the required data
CRITICAL_PATHS = CRITICAL_PATHS_V4 + CRITICAL_PATHS_V5

PERIOD_CRITICAL_PATHS = [
    "summary_id",
    "snapshot_type",
    "snapshot_mode",
    "as_of",
    "period.start_date",
    "period.end_date",
    "period.label",
    "period.expected_days",
    "period.observed_days",
    "period.coverage_pct",
    "period.is_complete",
    "period.missing_dates",
    "period_summary",
    "portfolio_changes",
    "intervals",
]

def _get(path: str, obj: dict):
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur

def validate_daily_snapshot(snap: dict, max_missing_pct: float = 0.20, critical_paths: List[str] | None = None) -> Tuple[bool, List[str]]:
    reasons = []
    paths = critical_paths or CRITICAL_PATHS
    
    # For V5 schema, check timestamps instead of top-level fields
    timestamps = snap.get("timestamps") or {}
    coverage = snap.get("coverage") or snap.get("data_quality") or {}
    
    # Check critical paths - for V5, accept either V4 or V5 format
    if timestamps:
        # V5 schema - check timestamps
        v5_checks = [
            ("timestamps.portfolio_data_as_of_utc", timestamps.get("portfolio_data_as_of_utc")),
            ("timestamps.portfolio_data_as_of_local", timestamps.get("portfolio_data_as_of_local")),
        ]
        for path, val in v5_checks:
            if val is None:
                reasons.append(f"missing {path}")
    else:
        # V4 schema - check top-level fields
        for path in CRITICAL_PATHS_V4:
            if _get(path, snap) is None:
                reasons.append(f"missing {path}")
    
    cov = snap.get("coverage") or snap.get("data_quality") or {}
    missing_pct = cov.get("missing_pct")
    if missing_pct is not None and missing_pct > max_missing_pct:
        reasons.append(f"coverage.missing_pct {missing_pct:.2f} > {max_missing_pct:.2f}")
    holdings = snap.get("holdings")
    if not isinstance(holdings, list):
        reasons.append("holdings is not a list")
    else:
        count_val = snap.get("count")
        count = None
        # Handle V5: portfolio.totals.holdings_count or V4: count field
        if isinstance(count_val, dict):
            count = count_val.get("holdings")
        elif isinstance(count_val, int):
            count = count_val
        # Also check V5: portfolio.totals.holdings_count
        if count is None:
            portfolio = snap.get("portfolio") or {}
            totals = portfolio.get("totals") or {}
            count = totals.get("holdings_count")
        if count is not None and count != len(holdings):
            reasons.append("count.holdings mismatch")
    return (len(reasons) == 0), reasons

def validate_period_snapshot(snap: dict, critical_paths: List[str] | None = None) -> Tuple[bool, List[str]]:
    reasons = []
    paths = critical_paths or PERIOD_CRITICAL_PATHS
    for path in paths:
        if _get(path, snap) is None:
            reasons.append(f"missing {path}")
    intervals = snap.get("intervals")
    if not isinstance(intervals, list):
        reasons.append("intervals is not a list")
    return (len(reasons) == 0), reasons

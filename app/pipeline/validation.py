from typing import Tuple, List

CRITICAL_PATHS = [
    "as_of",
    "as_of_utc",
    "as_of_date_local",
]

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
    for path in paths:
        if _get(path, snap) is None:
            reasons.append(f"missing {path}")
    cov = snap.get("coverage", {})
    missing_pct = cov.get("missing_pct")
    if missing_pct is not None and missing_pct > max_missing_pct:
        reasons.append(f"coverage.missing_pct {missing_pct:.2f} > {max_missing_pct:.2f}")
    holdings = snap.get("holdings")
    if not isinstance(holdings, list):
        reasons.append("holdings is not a list")
    else:
        count_val = snap.get("count")
        count = None
        if isinstance(count_val, dict):
            count = count_val.get("holdings")
        elif isinstance(count_val, int):
            count = count_val
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

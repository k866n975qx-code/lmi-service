import hashlib, json
import time as time_module
from datetime import datetime, date, time, timedelta, timezone
from dateutil import tz

def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def to_local_datetime_iso(dt_utc: datetime, local_tz: str) -> str:
    tzinfo = tz.gettz(local_tz)
    return dt_utc.astimezone(tzinfo).isoformat()

def to_local_date(dt_utc: datetime, local_tz: str, cutover_hhmm: str) -> date:
    tzinfo = tz.gettz(local_tz)
    loc = dt_utc.astimezone(tzinfo)
    hh, mm = cutover_hhmm.split(":"); cut = time(int(hh), int(mm))
    # If before cutover treat as previous local date
    if loc.timetz() < cut.replace(tzinfo=loc.tzinfo):
        loc = (loc - timedelta(days=1))
    return loc.date()

def retry_call(
    fn,
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 5.0,
    deadline: float | None = None,
    retry_on_result=None,
):
    last_exc = None
    last_result = None
    for attempt in range(1, attempts + 1):
        if deadline is not None and time_module.monotonic() >= deadline:
            raise TimeoutError("time_budget_exceeded")
        try:
            result = fn()
            last_result = result
            if retry_on_result and retry_on_result(result) and attempt < attempts:
                _sleep_with_deadline(base_delay, attempt, max_delay, deadline)
                continue
            return result
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                raise
            _sleep_with_deadline(base_delay, attempt, max_delay, deadline)
    if last_exc:
        raise last_exc
    return last_result

def _sleep_with_deadline(base_delay: float, attempt: int, max_delay: float, deadline: float | None):
    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
    if deadline is not None:
        remaining = deadline - time_module.monotonic()
        if remaining <= 0:
            raise TimeoutError("time_budget_exceeded")
        delay = min(delay, max(0.0, remaining))
    if delay > 0:
        time_module.sleep(delay)


class RateLimiter:
    def __init__(self, min_interval_seconds: float):
        self.min_interval_seconds = float(min_interval_seconds or 0.0)
        self._last_call = None

    def wait(self, deadline: float | None = None):
        if self.min_interval_seconds <= 0:
            return
        now = time_module.monotonic()
        if self._last_call is None:
            self._last_call = now
            return
        elapsed = now - self._last_call
        sleep_for = self.min_interval_seconds - elapsed
        if sleep_for > 0:
            if deadline is not None and now + sleep_for > deadline:
                raise TimeoutError("time_budget_exceeded")
            time_module.sleep(sleep_for)
        self._last_call = time_module.monotonic()

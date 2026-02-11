import sqlite3, json, hashlib, os, time
from datetime import datetime, timezone, timedelta, date
import httpx, structlog
from ..config import settings
from ..utils import now_utc_iso, to_local_date, retry_call

log = structlog.get_logger()

def _allowed_plaid_ids():
    raw = settings.lm_plaid_account_ids
    if not raw:
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}

def _filter_by_plaid_ids(items: list, id_keys: list[str]):
    allowed = _allowed_plaid_ids()
    if not allowed:
        return items
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = None
        for key in id_keys:
            val = item.get(key)
            if val is not None:
                item_id = str(val)
                break
        if item_id in allowed:
            out.append(item)
    return out

def _scrub_account_numbers(obj):
    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            if "account_number" in key.lower():
                continue
            cleaned[key] = _scrub_account_numbers(value)
        return cleaned
    if isinstance(obj, list):
        return [_scrub_account_numbers(value) for value in obj]
    return obj

def _coerce_float(val):
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

def _upsert_account_balances(conn: sqlite3.Connection, run_id: str, items: list[dict]):
    """Upsert account balances. Only advances as_of_date_local when balance actually changes,
    so day-over-day snapshots use the same margin when nothing changed (no phantom margin delta)."""
    if not items:
        return
    today = to_local_date(datetime.now(timezone.utc), settings.local_tz, settings.daily_cutover).isoformat()
    pulled_at = now_utc_iso()
    cur = conn.cursor()
    for item in items:
        if not isinstance(item, dict):
            continue
        plaid_account_id = item.get("id")
        if plaid_account_id is None:
            continue
        plaid_account_id = str(plaid_account_id)
        new_balance = _coerce_float(item.get("balance"))
        row = cur.execute(
            """
            SELECT as_of_date_local, balance FROM account_balances
            WHERE plaid_account_id = ?
            ORDER BY as_of_date_local DESC LIMIT 1
            """,
            (plaid_account_id,),
        ).fetchone()
        prev_balance = _coerce_float(row[1]) if row and row[1] is not None else None
        if row and new_balance is not None and prev_balance is not None and abs(prev_balance - new_balance) < 0.01:
            as_of_date_local = row[0]
        else:
            as_of_date_local = today
        cur.execute(
            """
            INSERT OR REPLACE INTO account_balances (
              as_of_date_local, plaid_account_id, name, institution_name, type, subtype,
              balance, credit_limit, currency, balance_last_update, source, run_id, pulled_at_utc
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                as_of_date_local,
                plaid_account_id,
                item.get("name") or item.get("display_name") or item.get("official_name"),
                item.get("institution_name"),
                item.get("type"),
                item.get("subtype"),
                new_balance,
                _coerce_float(item.get("limit")),
                item.get("currency"),
                item.get("balance_last_update"),
                "lunchmoney",
                run_id,
                pulled_at,
            ),
        )
    conn.commit()

def start_run(conn: sqlite3.Connection, run_id: str):
    conn.execute(
        "INSERT OR REPLACE INTO runs(run_id, started_at_utc, status) VALUES(?,?,?)",
        (run_id, now_utc_iso(), 'running'),
    )

def finish_run_ok(conn: sqlite3.Connection, run_id: str):
    conn.execute(
        "UPDATE runs SET finished_at_utc=?, status=? WHERE run_id=?",
        (now_utc_iso(), 'succeeded', run_id),
    )

def finish_run_fail(conn: sqlite3.Connection, run_id: str, err: str):
    conn.execute(
        "UPDATE runs SET finished_at_utc=?, status=?, error_message=? WHERE run_id=?",
        (now_utc_iso(), 'failed', err[:1000], run_id),
    )

def get_run_status(conn: sqlite3.Connection, run_id: str):
    cur = conn.cursor()
    row = cur.execute(
        "SELECT run_id, started_at_utc, finished_at_utc, status, error_message FROM runs WHERE run_id=?",
        (run_id,),
    ).fetchone()
    if not row: return None
    return {
        'run_id': row[0], 'started_at_utc': row[1], 'finished_at_utc': row[2], 'status': row[3], 'error_message': row[4]
    }

def _lm_headers():
    return {"Authorization": f"Bearer {settings.lm_token}", "Accept": "application/json"}

def _json_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()

def _append_payloads(conn: sqlite3.Connection, run_id: str, endpoint: str, objs: list, id_field: str):
    cur = conn.cursor()
    for obj in objs:
        obj = _scrub_account_numbers(obj)
        oid = str(obj.get(id_field)) if isinstance(obj, dict) else None
        sha = _json_sha(obj)
        cur.execute(
            "INSERT OR IGNORE INTO lm_raw(run_id, pulled_at_utc, endpoint, object_id, payload_json, payload_sha256) VALUES(?,?,?,?,?,?)",
            (run_id, now_utc_iso(), endpoint, oid, json.dumps(obj), sha),
        )

def _lm_get(client: httpx.Client, path: str, params: dict, deadline: float | None = None):
    base = settings.lm_base_url.rstrip("/")

    def _call():
        r = client.get(
            f"{base}{path}",
            params=params,
            headers=_lm_headers(),
            timeout=settings.http_timeout_seconds,
        )
        r.raise_for_status()
        return r.json()

    return retry_call(
        _call,
        attempts=settings.http_retry_attempts,
        base_delay=settings.http_retry_backoff_seconds,
        deadline=deadline,
    )

def append_lm_raw(
    conn: sqlite3.Connection,
    run_id: str,
    deadline: float | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    append_only: bool = True,
):
    # Pull transactions using date windows & pagination; append to lm_raw.
    cur = conn.cursor()
    allowed_ids = _allowed_plaid_ids()
    last = cur.execute(
        "SELECT lm_window_end FROM runs WHERE status='succeeded' AND lm_window_end IS NOT NULL ORDER BY finished_at_utc DESC LIMIT 1"
    ).fetchone()
    if not end_date:
        end_date = datetime.now(timezone.utc).date().isoformat()
    if not start_date:
        if not last or not last[0]:
            start_date = settings.lm_start_date or (datetime.now(timezone.utc) - timedelta(days=90)).date().isoformat()
        else:
            start_dt = date.fromisoformat(last[0])
            lookback_days = max(0, int(settings.lm_lookback_days or 0))
            if lookback_days:
                start_dt = start_dt - timedelta(days=lookback_days)
            if settings.lm_start_date:
                start_dt = max(start_dt, date.fromisoformat(settings.lm_start_date))
            start_date = start_dt.isoformat()

    total = 0
    seen_tx_ids = None
    if append_only:
        seen_tx_ids = {
            str(row[0])
            for row in cur.execute("SELECT lm_transaction_id FROM investment_transactions").fetchall()
            if row[0] is not None
        }
    with httpx.Client() as client:
        def _pull_transactions(plaid_account_id: str | None):
            pulled = 0
            limit = 500
            offset = 0
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("time_budget_exceeded")
                params = {"start_date": start_date, "end_date": end_date, "offset": offset, "limit": limit}
                if plaid_account_id:
                    params["plaid_account_id"] = plaid_account_id
                data = _lm_get(client, "/v1/transactions", params, deadline=deadline)
                items = data.get("transactions") or data.get("data") or data
                if not isinstance(items, list):
                    items = []
                raw_count = len(items)
                items = _filter_by_plaid_ids(items, ["plaid_account_id"])
                if append_only and seen_tx_ids is not None:
                    filtered = []
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        tx_id = item.get("id")
                        if tx_id is None:
                            continue
                        tx_id = str(tx_id)
                        if tx_id in seen_tx_ids:
                            continue
                        seen_tx_ids.add(tx_id)
                        filtered.append(item)
                    items = filtered
                _append_payloads(conn, run_id, "transactions", items, "id")
                pulled += len(items)
                if raw_count < limit:
                    break
                offset += limit
            return pulled

        if allowed_ids:
            for plaid_account_id in sorted(allowed_ids):
                total += _pull_transactions(plaid_account_id)
        else:
            total += _pull_transactions(None)
        # Plaid accounts (scrub account_number before storing)
        try:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("time_budget_exceeded")
            plaid = _lm_get(client, "/v1/plaid_accounts", {}, deadline=deadline)
            items = plaid.get("plaid_accounts") or plaid.get("accounts") or plaid.get("data") or plaid
            if isinstance(items, list):
                items = _filter_by_plaid_ids(items, ["id"])
                _upsert_account_balances(conn, run_id, items)
                _append_payloads(conn, run_id, "plaid_accounts", items, "id")
        except Exception as e:
            log.error("lm_plaid_accounts_failed", err=str(e))

    conn.execute(
        "UPDATE runs SET lm_window_start=?, lm_window_end=? WHERE run_id=?",
        (start_date, end_date, run_id),
    )
    log.info("lm_pull_done", start=start_date, end=end_date, total=total)
    return {"start": start_date, "end": end_date, "count": total}

def ensure_cusip_map(conn: sqlite3.Connection):
    cur = conn.cursor()
    existing = cur.execute("SELECT COUNT(*) FROM cusip_map").fetchone()[0]
    csv_path = settings.cusip_csv
    if not os.path.exists(csv_path):
        return
    with open(csv_path, "rb") as f:
        csv_bytes = f.read()
    sha = hashlib.sha256(csv_bytes).hexdigest()
    meta = cur.execute("SELECT value FROM metadata WHERE key='cusip_csv_sha'").fetchone()
    if existing == 0 or (meta and meta[0] != sha):
        import csv as _csv
        text = csv_bytes.decode("utf-8", errors="ignore")
        cur.execute("DELETE FROM cusip_map")
        for row in _csv.DictReader(text.splitlines()):
            cusip = row.get("cusip") or row.get("CUSIP") or row.get("Cusip")
            symbol = row.get("symbol") or row.get("ticker") or row.get("Symbol")
            desc = row.get("description") or row.get("name") or ""
            if not cusip or not symbol:
                continue
            cur.execute(
                "INSERT OR REPLACE INTO cusip_map(cusip, symbol, description) VALUES(?,?,?)",
                (cusip.strip(), symbol.strip().upper(), desc.strip()),
            )
        cur.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES('cusip_csv_sha', ?)", (sha,))

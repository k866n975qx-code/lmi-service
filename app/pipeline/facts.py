import sqlite3, json
from ..utils import now_utc_iso

def upsert_facts_from_sources(conn: sqlite3.Connection, run_id: str, daily: dict, sources: list[dict]):
    cur = conn.cursor()
    created = now_utc_iso()
    # Get as_of_date_local from V5 timestamps or fall back to V4 format
    timestamps = daily.get('timestamps') or {}
    as_of_date_local = timestamps.get('portfolio_data_as_of_local') or daily.get('as_of_date_local')
    if not as_of_date_local:
        as_of_utc = timestamps.get('portfolio_data_as_of_utc') or daily.get('as_of_utc') or ''
        as_of_date_local = as_of_utc[:10] if as_of_utc else ''
    for s in sources:
        params = s.get('params')
        if params is not None and not isinstance(params, str):
            params = json.dumps(params)
        cur.execute(
            """
            INSERT OR REPLACE INTO facts_source_daily (
              as_of_date_local, scope, symbol, field_path,
              value_text, value_num, value_int, value_type,
              source_type, provider, endpoint, params, source_ref, commit_sha,
              provider_rank, run_id, created_at_utc
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                s.get('as_of_date_local', as_of_date_local), s['scope'], s.get('symbol'), s['field_path'],
                s.get('value_text'), s.get('value_num'), s.get('value_int'), s['value_type'],
                s.get('source_type', 'unknown'), s.get('provider','unknown'), s.get('endpoint'), params, s.get('source_ref'), s.get('commit_sha'),
                s.get('provider_rank', 1), run_id, created
            )
        )
    conn.commit()

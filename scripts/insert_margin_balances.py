"""Insert historical margin balances by date into account_balances (loan account only).
Uses existing account_balances to detect the loan account (type=loan); else set MARGIN_PLAID_ACCOUNT_ID.
Does not use margin_balance_history."""
import os
import sqlite3
from datetime import datetime, timezone
from app.db import get_conn, migrate
from app.config import settings
from app.utils import now_utc_iso

BALANCES = [
    ("2025-11-06", 995.21),
    ("2025-11-07", 2195.18),
    ("2025-11-08", 2195.18),
    ("2025-11-09", 2195.18),
    ("2025-11-10", 2195.18),
    ("2025-11-11", 2195.18),
    ("2025-11-12", 2195.18),
    ("2025-11-13", 2195.18),
    ("2025-11-14", 2195.18),
    ("2025-11-15", 2195.18),
    ("2025-11-16", 2195.18),
    ("2025-11-17", 1795.18),
    ("2025-11-18", 2120.01),
    ("2025-11-19", 2120.01),
    ("2025-11-20", 2120.01),
    ("2025-11-21", 2119.22),
    ("2025-11-22", 2119.22),
    ("2025-11-23", 2119.22),
    ("2025-11-24", 2119.22),
    ("2025-11-25", 2119.22),
    ("2025-11-26", 2119.22),
    ("2025-11-27", 2119.22),
    ("2025-11-28", 2119.22),
    ("2025-11-29", 2119.22),
    ("2025-11-30", 2119.22),
    ("2025-12-01", 2119.22),
    ("2025-12-02", 2119.22),
    ("2025-12-03", 2125.74),
    ("2025-12-04", 2825.74),
    ("2025-12-05", 2825.74),
    ("2025-12-06", 2825.74),
    ("2025-12-07", 2825.74),
    ("2025-12-08", 3025.74),
    ("2025-12-09", 3025.74),
    ("2025-12-10", 3025.74),
    ("2025-12-11", 3025.74),
    ("2025-12-12", 3025.74),
    ("2025-12-13", 3025.74),
    ("2025-12-14", 3025.74),
    ("2025-12-15", 3025.74),
    ("2025-12-16", 3755.74),
    ("2025-12-17", 7155.74),
    ("2025-12-18", 7155.74),
    ("2025-12-19", 7155.74),
    ("2025-12-20", 7155.74),
    ("2025-12-21", 7155.74),
    ("2025-12-22", 7155.74),
    ("2025-12-23", 7155.74),
    ("2025-12-24", 7155.74),
    ("2025-12-25", 7155.74),
    ("2025-12-26", 7155.74),
    ("2025-12-27", 7155.74),
    ("2025-12-28", 7155.74),
    ("2025-12-29", 7155.74),
    ("2025-12-30", 7155.74),
    ("2025-12-31", 7155.74),
    ("2026-01-01", 7155.74),
    ("2026-01-02", 7155.74),
    ("2026-01-03", 7155.74),
    ("2026-01-04", 7155.74),
    ("2026-01-05", 7155.74),
    ("2026-01-06", 7173.56),
    ("2026-01-07", 7173.56),
    ("2026-01-08", 8642.31),
    ("2026-01-09", 9355.27),
    ("2026-01-10", 9355.27),
    ("2026-01-11", 9355.27),
    ("2026-01-12", 11409.23),
    ("2026-01-13", 11409.23),
    ("2026-01-14", 12167.15),
    ("2026-01-15", 12167.15),
    ("2026-01-16", 13052.31),
    ("2026-01-17", 13052.31),
    ("2026-01-18", 13052.31),
    ("2026-01-19", 13052.31),
    ("2026-01-20", 16772.23),
    ("2026-01-21", 16772.23),
    ("2026-01-22", 16772.23),
    ("2026-01-23", 16772.23),
    ("2026-01-24", 16772.23),
    ("2026-01-25", 16772.23),
    ("2026-01-26", 17066.95),
    ("2026-01-27", 17066.95),
    ("2026-01-28", 17356.93),
    ("2026-01-29", 17356.93),
    ("2026-01-30", 17786.93),
    ("2026-01-31", 17786.93),
    ("2026-02-01", 17786.93),
    ("2026-02-02", 17786.93),
    ("2026-02-03", 17786.93),
    ("2026-02-04", 17832.02),
    ("2026-02-05", 17832.02),
    ("2026-02-06", 17832.02),
    ("2026-02-07", 17832.02),
]


def _get_loan_account(cur: sqlite3.Cursor) -> tuple[str, str]:
    row = cur.execute(
        "SELECT plaid_account_id, name FROM account_balances WHERE type = 'loan' LIMIT 1"
    ).fetchone()
    if row:
        return (row[0], row[1] or "M1 Borrow")
    plaid_id = os.environ.get("MARGIN_PLAID_ACCOUNT_ID", "317632")
    return (plaid_id, "M1 Borrow")


def main():
    conn = get_conn(settings.db_path)
    migrate(conn)
    cur = conn.cursor()
    plaid_account_id, name = _get_loan_account(cur)
    pulled_at = now_utc_iso()
    run_id = f"manual_margin_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    cur.executemany(
        """
        INSERT OR REPLACE INTO account_balances (
          as_of_date_local, plaid_account_id, name, institution_name, type, subtype,
          balance, credit_limit, currency, balance_last_update, source, run_id, pulled_at_utc
        ) VALUES (?, ?, ?, NULL, 'loan', NULL, ?, NULL, 'USD', NULL, 'manual_entry', ?, ?)
        """,
        [(d, plaid_account_id, name, b, run_id, pulled_at) for d, b in BALANCES],
    )
    conn.commit()
    count = cur.execute(
        "SELECT COUNT(*) FROM account_balances WHERE plaid_account_id = ? AND source = 'manual_entry'",
        (plaid_account_id,),
    ).fetchone()[0]
    print(f"Inserted {len(BALANCES)} margin balance rows into account_balances (loan account {plaid_account_id}). Total manual_entry rows for this account: {count}")


if __name__ == "__main__":
    main()

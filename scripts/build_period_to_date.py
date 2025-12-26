from pathlib import Path
import os
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from app.db import get_conn
from app.config import settings
from app.pipeline.periods import build_period_snapshot


def main(snapshot_type: str, as_of: str | None, mode: str):
    conn = get_conn(settings.db_path)
    snap = build_period_snapshot(conn, snapshot_type=snapshot_type, as_of=as_of, mode=mode)
    out_name = f"period-{snapshot_type}-{snap['as_of']}.json"
    with open(out_name, "w") as f:
        json.dump(snap, f, indent=2, sort_keys=False)
    print("Wrote", out_name)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/build_period_to_date.py <weekly|monthly|quarterly|yearly> [as_of=YYYY-MM-DD] [mode=to_date|final]")
        raise SystemExit(2)
    snapshot_type = sys.argv[1]
    as_of = sys.argv[2] if len(sys.argv) > 2 else None
    mode = sys.argv[3] if len(sys.argv) > 3 else "to_date"
    main(snapshot_type, as_of, mode)

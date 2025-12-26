from pathlib import Path
import os
import sys

# Ensure repo root is on sys.path and is the working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from app.db import get_conn, migrate
from app.config import settings
from app.pipeline.utils import ensure_cusip_map

if __name__ == '__main__':
    conn = get_conn(settings.db_path)
    migrate(conn)
    ensure_cusip_map(conn)
    cusip_count = conn.execute("SELECT COUNT(*) FROM cusip_map").fetchone()[0]
    print('DB ready at', settings.db_path, '| CUSIP rows:', cusip_count)

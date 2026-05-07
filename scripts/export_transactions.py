from pathlib import Path
import os
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from app.config import settings
from app.db import get_conn, migrate
from app.pipeline.transaction_export import export_transactions_csv


if __name__ == "__main__":
    output_path = sys.argv[1] if len(sys.argv) > 1 else settings.transaction_export_path
    conn = get_conn(settings.db_path)
    migrate(conn)
    result = export_transactions_csv(conn, output_path)
    print(f"Exported {result.row_count} transactions to {result.output_path}")

from pathlib import Path
import os
import sys
import uuid

# Ensure repo root is on sys.path and is the working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from app.pipeline.orchestrator import _sync_impl

if __name__ == '__main__':
    run_id = str(uuid.uuid4())
    print('Run', run_id)
    _sync_impl(run_id)
    print('Done.')

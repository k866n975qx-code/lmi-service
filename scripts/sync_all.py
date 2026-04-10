from pathlib import Path
import os
import signal
import sys
import uuid

# Ensure repo root is on sys.path and is the working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from app.config import settings
from app.pipeline.orchestrator import _sync_impl
from app.pipeline.utils import mark_run_interrupted

if __name__ == '__main__':
    run_id = str(uuid.uuid4())
    completed = False

    def _handle_signal(signum, _frame):
        signal_name = signal.Signals(signum).name
        if not completed:
            try:
                marked = mark_run_interrupted(settings.db_path, run_id, signal_name)
                if marked:
                    print(f'Interrupted {run_id} via {signal_name}.')
            except Exception as exc:
                print(f'Failed to record interruption for {run_id}: {exc}', file=sys.stderr)
        raise SystemExit(128 + signum)

    for handled_signal in (signal.SIGTERM, signal.SIGINT):
        signal.signal(handled_signal, _handle_signal)

    print('Run', run_id)
    _sync_impl(run_id)
    completed = True
    print('Done.')

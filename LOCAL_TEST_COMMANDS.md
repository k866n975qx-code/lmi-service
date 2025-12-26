# Local Testing Commands

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

## Initialize DB + seed CUSIP
```bash
python scripts/init_db.py
```

## Run one sync (CLI)
```bash
python scripts/sync_all.py
```

## Run API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8010 --reload
```

## Smoke endpoints
```bash
curl -s http://127.0.0.1:8010/health | python3 -m json.tool
curl -s http://127.0.0.1:8010/snapshots/available | python3 -m json.tool
curl -s "http://127.0.0.1:8010/period/weekly/2025-12-25/to_date?slim=true" | python3 -m json.tool
curl -s http://127.0.0.1:8010/diff/daily/2025-12-24/2025-12-25 | python3 -m json.tool
curl -s -X POST http://127.0.0.1:8010/sync-all | python3 -m json.tool
```

## Quick validation
```bash
python3 -m compileall app scripts
sqlite3 data/app.db "select 'snapshot_daily_current', count(*) from snapshot_daily_current; select 'snapshots', count(*) from snapshots; select 'dividend_events_lm', count(*) from dividend_events_lm; select 'dividend_events_provider', count(*) from dividend_events_provider;"
```

## Create schema without Python (Ubuntu)
```bash
sqlite3 data/app.db < migrations/001_init.sql
```

## Systemd (system-level, sudo)
```bash
sudo cp systemd/lmi@.service /etc/systemd/system/
sudo cp systemd/lmi-sync@.service /etc/systemd/system/
sudo cp systemd/lmi-sync@.timer /etc/systemd/system/
sudo systemctl daemon-reload

# replace <repo> with your repo folder name under /home/jose
sudo systemctl enable --now lmi@<repo>.service
sudo systemctl enable --now lmi-sync@<repo>.timer

sudo systemctl status lmi@<repo>.service
sudo systemctl list-timers | rg lmi-sync
```

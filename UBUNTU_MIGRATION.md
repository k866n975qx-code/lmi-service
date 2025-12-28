# Ubuntu Migration Steps (Port 8010)

These steps assume the old 8010 service is already removed. Commands are written for a system-level systemd setup.

## 1) Install system packages
```bash
sudo apt update
sudo apt install -y git sqlite3 build-essential python3.11 python3.11-venv python3.11-dev
```

If `python3.11` is not available on your Ubuntu version, install it first (or tell me the version and I will adjust the steps).

## 2) Clone the repo
```bash
cd /home/jose
git clone https://github.com/k866n975qx-code/lmi-service
cd lmi-service
```

## 3) Create venv + install deps
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

## 4) Configure environment
```bash
cp .env.example .env
chmod 600 .env
```

Edit `.env` and set at least:
- `LM_TOKEN`
- `LM_PLAID_ACCOUNT_IDS`
- `LM_START_DATE` (e.g., `2025-10-01`)
- `FRED_API_KEY` (if used)

Optional logging (errors only):
```
LOG_LEVEL=ERROR
LOG_ERROR_FILE=./data/logs/error.log
```

## 5) Initialize DB (schema + CUSIP)
```bash
python scripts/init_db.py
```

## 6) First sync (manual smoke)
```bash
python scripts/sync_all.py
```

## 7) Start API locally for a quick test (optional)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8010
```
Then test:
```bash
curl -s http://127.0.0.1:8010/health | python3 -m json.tool
```

## 8) Install systemd units (system-level)
```bash
sudo cp systemd/lmi@.service /etc/systemd/system/
sudo cp systemd/lmi-sync@.service /etc/systemd/system/
sudo cp systemd/lmi-sync@.timer /etc/systemd/system/
sudo systemctl daemon-reload
```

Enable services using the repo folder name as the instance:
```bash
sudo systemctl enable --now lmi@lmi-service.service
sudo systemctl enable --now lmi-sync@lmi-service.timer
```

## 9) Verify
```bash
sudo systemctl status lmi@lmi-service.service
sudo systemctl list-timers | rg lmi-sync
curl -s http://127.0.0.1:8010/health | python3 -m json.tool
```

Logs (API):
```bash
sudo journalctl -u lmi@lmi-service.service -f --no-pager
```

Logs (hourly sync):
```bash
sudo journalctl -u lmi-sync@lmi-service.service -f --no-pager
sudo journalctl -u lmi-sync@lmi-service.timer -f --no-pager
```

## 10) Common fixes
- Port 8010 already in use: stop the old service or change port in `systemd/lmi@.service`.
- Missing `.env`: ensure the repo root has `.env` with correct values.
- Permission errors: check file ownership under `/home/jose/lmi-service`.

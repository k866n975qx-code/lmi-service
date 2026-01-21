#!/usr/bin/env bash
set -euo pipefail

SERVER="jose@192.168.12.221"
REMOTE_DB="/home/jose/lmi-service/data/app.db"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
LOCAL_DB="$REPO_ROOT/data/app.db"
LOCAL_DIR=$(dirname "$LOCAL_DB")

mkdir -p "$LOCAL_DIR"

# Copy the DB down to macOS (overwrites local file).
scp "$SERVER:$REMOTE_DB" "$LOCAL_DB"

echo "Copied to $LOCAL_DB"

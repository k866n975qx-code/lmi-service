#!/usr/bin/env bash
set -euo pipefail

SERVER="jose@192.168.12.221"
REMOTE_DB="/home/jose/lmi-service/data/app.db"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
LOCAL_DB="$REPO_ROOT/data/app.db"

# Default to dry-run for safety
DRY_RUN=true

# Parse arguments
for arg in "$@"; do
  case $arg in
    --push|--execute|-f|--force)
      DRY_RUN=false
      shift
      ;;
    --dry-run|-n)
      DRY_RUN=true
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Push local database to server via Cloudflare tunnel."
      echo ""
      echo "Options:"
      echo "  --dry-run, -n       Show what would be copied (default)"
      echo "  --push, --execute   Actually perform the copy"
      echo "  -f, --force         Alias for --push"
      echo "  --help, -h          Show this help"
      echo ""
      echo "Examples:"
      echo "  $0                  # Dry-run (safe, shows what would happen)"
      echo "  $0 --dry-run        # Explicit dry-run"
      echo "  $0 --push           # Actually push the database"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Check local DB exists
if [[ ! -f "$LOCAL_DB" ]]; then
  echo "❌ Error: Local database not found at $LOCAL_DB"
  exit 1
fi

# Get local file info
LOCAL_SIZE=$(du -h "$LOCAL_DB" | cut -f1)
LOCAL_MTIME=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$LOCAL_DB" 2>/dev/null || stat -c "%y" "$LOCAL_DB" 2>/dev/null | cut -d'.' -f1)

echo "========================================="
echo "Database Push Script"
echo "========================================="
echo "Local DB:   $LOCAL_DB"
echo "Size:       $LOCAL_SIZE"
echo "Modified:   $LOCAL_MTIME"
echo ""
echo "Remote:     $SERVER:$REMOTE_DB"
echo "========================================="

# Check if remote DB exists and get info
echo "Checking remote database..."
if ssh "$SERVER" "test -f $REMOTE_DB"; then
  REMOTE_SIZE=$(ssh "$SERVER" "du -h $REMOTE_DB | cut -f1")
  REMOTE_MTIME=$(ssh "$SERVER" "stat -c '%y' $REMOTE_DB 2>/dev/null | cut -d'.' -f1")
  echo "Remote DB exists:"
  echo "  Size:     $REMOTE_SIZE"
  echo "  Modified: $REMOTE_MTIME"
  echo ""
  BACKUP_NEEDED=true
else
  echo "Remote DB does not exist (will create new)"
  echo ""
  BACKUP_NEEDED=false
fi

if [[ "$DRY_RUN" == true ]]; then
  echo "🔍 DRY RUN MODE - No changes will be made"
  echo ""
  echo "Would perform the following actions:"

  if [[ "$BACKUP_NEEDED" == true ]]; then
    BACKUP_PATH="${REMOTE_DB}.backup.$(date +%Y%m%d_%H%M%S)"
    echo "  1. Backup remote DB to: $BACKUP_PATH"
  fi

  echo "  2. Copy $LOCAL_DB"
  echo "     to   $SERVER:$REMOTE_DB"
  echo ""
  echo "To actually push, run with --push flag:"
  echo "  $0 --push"
  exit 0
fi

# EXECUTE MODE
echo "⚠️  EXECUTE MODE - Database will be pushed to server"
echo ""
read -p "Continue? (yes/no): " -r CONFIRM
echo ""

if [[ "$CONFIRM" != "yes" ]]; then
  echo "Aborted."
  exit 0
fi

# Create backup on server if DB exists
if [[ "$BACKUP_NEEDED" == true ]]; then
  BACKUP_PATH="${REMOTE_DB}.backup.$(date +%Y%m%d_%H%M%S)"
  echo "Creating backup on server..."
  ssh "$SERVER" "cp $REMOTE_DB $BACKUP_PATH"
  echo "✅ Backup created: $BACKUP_PATH"
fi

# Ensure remote directory exists
echo "Ensuring remote directory exists..."
ssh "$SERVER" "mkdir -p $(dirname $REMOTE_DB)"

# Copy the DB to server
echo "Pushing database..."
scp "$LOCAL_DB" "$SERVER:$REMOTE_DB"

echo ""
echo "✅ Database pushed successfully to $SERVER:$REMOTE_DB"

# Verify remote file
REMOTE_SIZE_NEW=$(ssh "$SERVER" "du -h $REMOTE_DB | cut -f1")
echo "Remote size after push: $REMOTE_SIZE_NEW"

#!/bin/bash
# LMI Database Optimization Script
# Safe to run while lmi-service is running
# Creates indexes and optimizes query planner

set -e

DB_PATH="/home/jose/lmi-service/data/app.db"

echo "Starting database optimization..."
echo "Database: $DB_PATH"
echo ""

# Check if database exists
if [ ! -f "$DB_PATH" ]; then
    echo "Error: Database not found at $DB_PATH"
    exit 1
fi

# Run optimizations (safe while service is running)
sqlite3 "$DB_PATH" <<EOF
-- Create indexes for faster dashboard queries (flat tables)
CREATE INDEX IF NOT EXISTS idx_daily_portfolio_as_of
  ON daily_portfolio(as_of_date_local DESC);

CREATE INDEX IF NOT EXISTS idx_period_summary_type_end
  ON period_summary(period_type, period_end_date DESC);

-- Optimize query planner
PRAGMA optimize;

-- Show current indexes
.indexes
EOF

echo ""
echo "✓ Database optimization complete!"
echo ""
echo "Indexes created:"
echo "  - idx_daily_portfolio_as_of (speeds up latest daily queries)"
echo "  - idx_period_summary_type_end (speeds up period queries)"
echo ""
echo "Note: This is safe to run while lmi-service is running."

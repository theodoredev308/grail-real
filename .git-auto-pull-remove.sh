#!/bin/bash
# Script to remove the auto-pull cron job

echo "Removing auto-pull cron job..."

# Remove the cron entry
crontab -l 2>/dev/null | grep -v "git-auto-pull" | crontab -

if [ $? -eq 0 ]; then
    echo "✓ Auto-pull cron job removed successfully"
    echo ""
    echo "Remaining cron jobs:"
    crontab -l 2>/dev/null || echo "  (no cron jobs remaining)"
else
    echo "✗ Error removing cron job"
    exit 1
fi


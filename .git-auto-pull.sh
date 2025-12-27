#!/bin/bash
# Auto-pull script for grail repository

REPO_DIR="/home/administrator/ajo_work/mining/grail/grail"
BRANCH="main"
LOG_FILE="/tmp/grail-auto-pull.log"

cd "$REPO_DIR" || exit 1

# Log the pull attempt
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Attempting to pull changes..." >> "$LOG_FILE"

# Fetch latest changes
git fetch origin >> "$LOG_FILE" 2>&1

# Check if there are updates
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse @{u})

if [ "$LOCAL" != "$REMOTE" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Changes detected, pulling..." >> "$LOG_FILE"
    git pull origin "$BRANCH" >> "$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Successfully pulled changes" >> "$LOG_FILE"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Pull failed (may have conflicts)" >> "$LOG_FILE"
    fi
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] No changes to pull" >> "$LOG_FILE"
fi


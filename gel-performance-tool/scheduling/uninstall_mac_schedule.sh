#!/bin/bash
# Removes the daily schedule installed by install_mac_schedule.sh
PLIST="$HOME/Library/LaunchAgents/com.gelperformance.daily.plist"
launchctl unload "$PLIST" 2>/dev/null || true
rm -f "$PLIST"
echo "Daily schedule removed."

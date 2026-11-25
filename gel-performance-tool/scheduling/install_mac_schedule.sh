#!/bin/bash
# Installs a daily schedule on macOS with launchd (built in, no extra software).
# The job runs scheduling/run_daily.sh every day at the time below. If the Mac is asleep at that
# time, launchd runs the job when it wakes up.
#   Install:    bash scheduling/install_mac_schedule.sh
#   Uninstall:  bash scheduling/uninstall_mac_schedule.sh
#   Logs:       output/run.log (the tool) and output/launchd.log (anything printed by the job)
set -euo pipefail

HOUR=7      # change to run at another time (24h clock)
MINUTE=10

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.gelperformance.daily"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
mkdir -p "$HOME/Library/LaunchAgents" "$PROJECT_DIR/output"

cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$PROJECT_DIR/scheduling/run_daily.sh</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>$HOUR</integer><key>Minute</key><integer>$MINUTE</integer></dict>
  <key>StandardOutPath</key><string>$PROJECT_DIR/output/launchd.log</string>
  <key>StandardErrorPath</key><string>$PROJECT_DIR/output/launchd.log</string>
</dict>
</plist>
PLIST

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "Installed: runs every day at $(printf '%02d:%02d' $HOUR $MINUTE)."
echo "Run it once now with:  launchctl start $LABEL"

#!/bin/bash
# Daily job: release the next simulated day(s) of data, then score them.
# Run it by hand to test:   bash scheduling/run_daily.sh
# Schedule it on a Mac:     bash scheduling/install_mac_schedule.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# Use the project's own Python environment if it exists
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

python -m gel_performance.feed --source data/synthetic/cycles.csv --live data/live/cycles.csv
python -m gel_performance.run --input data/live/cycles.csv --output output --config config/config.yaml

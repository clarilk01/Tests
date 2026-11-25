"""Simulated daily data feed: makes the synthetic data arrive day by day, like a real plant.

    python -m gel_performance.feed --source data/synthetic/cycles.csv --live data/live/cycles.csv

The live file holds every source cycle whose cycle_end_time is on or before the "simulated now":

    simulated now = first source time + start_days + days_per_day * (today - feed start date)

The feed start date is saved in data/live/feed_state.json on the first run. Because the content depends
only on today's date (not on how often the job ran), running it twice on the same day changes nothing
(idempotent). Use --today YYYY-MM-DD to try a future day, and --reset to start the feed again.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path

import pandas as pd

log = logging.getLogger("gel_performance.feed")
ROOT = Path(__file__).resolve().parents[2]


def simulated_now(first_time: pd.Timestamp, start_days: int, days_per_day: int,
                  feed_start: dt.date, today: dt.date) -> pd.Timestamp:
    elapsed = max((today - feed_start).days, 0)
    return first_time.normalize() + pd.Timedelta(days=start_days + days_per_day * elapsed)


def update_feed(source: Path, live: Path, start_days: int = 150, days_per_day: int = 7,
                today: dt.date | None = None, reset: bool = False) -> dict:
    """Write the live file for `today`. Returns a small summary (also logged)."""
    today = today or dt.date.today()
    state_path = live.parent / "feed_state.json"
    live.parent.mkdir(parents=True, exist_ok=True)
    if reset or not state_path.exists():
        state = {"feed_start_date": today.isoformat(), "start_days": start_days, "days_per_day": days_per_day}
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        log.info("Feed started on %s", today)
    state = json.loads(state_path.read_text())

    src = pd.read_csv(source)
    times = pd.to_datetime(src["cycle_end_time"])
    now = simulated_now(times.min(), state["start_days"], state["days_per_day"],
                        dt.date.fromisoformat(state["feed_start_date"]), today)
    live_rows = src[times <= now]

    tmp = live.with_name(live.name + ".tmp")
    live_rows.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, live)
    summary = {"simulated_now": str(now), "cycles": len(live_rows), "of_total": len(src),
               "finished": len(live_rows) == len(src)}
    log.info("Live feed up to simulated %s: %d of %d cycles%s", now.date(), len(live_rows), len(src),
             " (all data delivered)" if summary["finished"] else "")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Release synthetic cycles day by day into a live input file.")
    parser.add_argument("--source", type=Path, default=ROOT / "data/synthetic/cycles.csv")
    parser.add_argument("--live", type=Path, default=ROOT / "data/live/cycles.csv")
    parser.add_argument("--start-days", type=int, default=150, help="days of history shown on the first day")
    parser.add_argument("--days-per-day", type=int, default=7, help="days of plant data released per real day")
    parser.add_argument("--today", type=dt.date.fromisoformat, help="pretend today is this date (YYYY-MM-DD)")
    parser.add_argument("--reset", action="store_true", help="start the feed again from today")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    update_feed(args.source, args.live, args.start_days, args.days_per_day, args.today, args.reset)


if __name__ == "__main__":
    main()

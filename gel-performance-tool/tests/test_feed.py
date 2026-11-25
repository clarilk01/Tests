"""Checks for the simulated daily data feed."""

import datetime as dt

import pandas as pd

from gel_performance.feed import update_feed


def make_source(tmp_path):
    times = pd.date_range("2024-01-01 12:00", periods=40, freq="D")
    src = pd.DataFrame({"gel_batch": "GB-A01", "resin_age": range(1, 41), "cycle_end_time": times.strftime("%Y-%m-%dT%H:%M:%S")})
    path = tmp_path / "source.csv"
    src.to_csv(path, index=False)
    return path


def test_feed_releases_days_by_calendar_and_is_idempotent(tmp_path):
    source, live = make_source(tmp_path), tmp_path / "live" / "cycles.csv"
    day1 = dt.date(2026, 1, 1)
    s1 = update_feed(source, live, start_days=10, days_per_day=7, today=day1)
    assert s1["cycles"] == 10                                  # cycles on 2024-01-01 .. 2024-01-10
    first = live.read_bytes()
    update_feed(source, live, start_days=10, days_per_day=7, today=day1)
    assert live.read_bytes() == first                          # same day again -> nothing changes
    s2 = update_feed(source, live, today=day1 + dt.timedelta(days=1))
    assert s2["cycles"] == 17                                  # one real day later -> 7 more days of data
    s3 = update_feed(source, live, today=day1 + dt.timedelta(days=30))
    assert s3["finished"] and s3["cycles"] == 40


def test_reset_starts_again(tmp_path):
    source, live = make_source(tmp_path), tmp_path / "live" / "cycles.csv"
    update_feed(source, live, start_days=10, today=dt.date(2026, 1, 1))
    s = update_feed(source, live, start_days=10, today=dt.date(2026, 1, 5), reset=True)
    assert s["cycles"] == 10

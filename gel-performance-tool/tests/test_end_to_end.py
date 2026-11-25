"""End-to-end: run the real command-line entry point on a small synthetic dataset."""

from pathlib import Path

import pandas as pd
import pytest

from gel_performance.run import EXIT_CONFIG, EXIT_DATA, EXIT_OK, main
from gel_performance.simulate import SimSettings, simulate

CONFIG = Path(__file__).parents[1] / "config" / "config.yaml"
OUTPUTS = ["cycle_scores.csv", "batch_summary.csv", "reference_stats.csv"]


@pytest.fixture(scope="module")
def small_input(tmp_path_factory):
    """2 gel batches per type, generated with a fixed seed."""
    cycles, _, _ = simulate(seed=1, s=SimSettings(batches_per_type=2, n_outliers_per_type=2))
    path = tmp_path_factory.mktemp("in") / "cycles.csv"
    cycles.to_csv(path, index=False)
    return path


def run_cli(input_path, out_dir, config=CONFIG):
    return main(["--input", str(input_path), "--output", str(out_dir), "--config", str(config)])


def test_full_run_produces_all_outputs(small_input, tmp_path):
    assert run_cli(small_input, tmp_path) == EXIT_OK
    for name in OUTPUTS:
        assert (tmp_path / name).exists()

    scores = pd.read_csv(tmp_path / "cycle_scores.csv")
    n_in = len(pd.read_csv(small_input))
    assert len(scores) == n_in                                   # no cycle lost
    assert scores["score"].dropna().between(0, 1).all()          # replacement score is 0..1
    assert set(scores["decision"]) <= {"GO", "NO-GO", "NO SCORE"}
    assert (scores["flag"] == "incomplete").any()                # missing data was flagged
    # NO-GO only when BOTH conditions hold
    nogo = scores[scores["decision"] == "NO-GO"]
    assert (nogo["score"] >= 0.5).all() and nogo["at_pressure_limit"].all()
    # fresh resin is GO, worn-out resin is NO-GO
    by_batch = scores.sort_values("resin_age").groupby("gel_batch")["decision"]
    assert (by_batch.first() == "GO").all()
    assert (by_batch.last() == "NO-GO").all()


def test_running_twice_gives_identical_files(small_input, tmp_path):
    assert run_cli(small_input, tmp_path / "a") == EXIT_OK
    assert run_cli(small_input, tmp_path / "a") == EXIT_OK   # second run over the first one's output
    assert run_cli(small_input, tmp_path / "b") == EXIT_OK
    for name in OUTPUTS:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_row_order_of_the_input_does_not_change_the_output(small_input, tmp_path):
    shuffled = tmp_path / "shuffled.csv"
    pd.read_csv(small_input).sample(frac=1, random_state=0).to_csv(shuffled, index=False)
    assert run_cli(small_input, tmp_path / "a") == EXIT_OK
    assert run_cli(shuffled, tmp_path / "b") == EXIT_OK
    for name in OUTPUTS:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_log_records_what_happened(small_input, tmp_path):
    run_cli(small_input, tmp_path)
    log = (tmp_path / "run.log").read_text()
    assert "Read " in log and "rows from" in log
    assert "incomplete" in log
    assert "Run finished OK" in log


def test_bad_config_fails_with_config_exit_code(small_input, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(CONFIG.read_text().replace("decision_threshold: 0.5", "decision_threshold: 5"))
    assert run_cli(small_input, tmp_path / "out", config=bad) == EXIT_CONFIG
    assert "decision_threshold" in (tmp_path / "out" / "run.log").read_text()


def test_missing_model_file_fails_with_config_exit_code(small_input, tmp_path):
    (tmp_path / "config").mkdir()
    cfg = tmp_path / "config" / "config.yaml"           # model path is resolved next to this config
    cfg.write_text(CONFIG.read_text())
    assert run_cli(small_input, tmp_path / "out", config=cfg) == EXIT_CONFIG
    assert "Model file not found" in (tmp_path / "out" / "run.log").read_text()


def test_missing_column_fails_with_data_exit_code(small_input, tmp_path):
    broken = tmp_path / "broken.csv"
    pd.read_csv(small_input).drop(columns=["pt_80_percentile"]).to_csv(broken, index=False)
    assert run_cli(broken, tmp_path / "out") == EXIT_DATA
    assert "missing required columns" in (tmp_path / "out" / "run.log").read_text()

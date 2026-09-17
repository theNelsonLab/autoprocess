"""
Per-dataset result records (autoprocess_logs/<dataset>_status.json) and remembered failures.

Callers such as REyes run autoprocess once per movie folder and need to know what happened
without guessing from XDS.INP or CORRECT.LP, and without XDS being re-run on every monitor
restart for a dataset that is known to fail.
"""
import json
import sys

import pytest

from pyautoprocess import __version__
from pyautoprocess.autoprocess import CrystallographyProcessor
from pyautoprocess.config.config_manager import ConfigLoader

MOVIE = "sample-mov1_960_0.3_3_P50toN-50.ser"
DATASET = "sample-mov1"


def make_processor(**overrides):
    params = ConfigLoader().get_config('default')
    params.paths = []
    for key, value in overrides.items():
        setattr(params, key, value)
    return CrystallographyProcessor(params)


def fake_xds(processor, monkeypatch, produce, succeed, reason=None):
    """Stand in for conversion + XDS: create `produce` in auto_process/ and return `succeed`."""
    calls = []

    def process_movie_data(sample_movie, distance, rotation, exposure, res, test_res,
                           filename, source_file_path):
        calls.append(filename)
        auto_process = source_file_path.parent / sample_movie / "auto_process"
        auto_process.mkdir(parents=True, exist_ok=True)
        for name in produce:
            (auto_process / name.format(dataset=sample_movie)).write_text("x")
        if reason:
            processor._failure_reason = reason
        return succeed

    monkeypatch.setattr(processor, "_process_movie_data", process_movie_data)
    monkeypatch.setattr(processor, "_setup_movie_directories",
                        lambda sample_movie, distance, src: src.parent / sample_movie)
    return calls


def status(tmp_path):
    return json.loads((tmp_path / "autoprocess_logs" / f"{DATASET}_status.json").read_text())


@pytest.fixture
def movie(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / MOVIE
    path.write_bytes(b"x")
    return path


SUCCESS_FILES = ["XDS.INP", "XPARM.XDS", "INTEGRATE.HKL", "CORRECT.LP", "XDS_ASCII.HKL",
                 "{dataset}.ahkl", "{dataset}.hkl", "stats.LP"]


# ------------------------------------------------------------------ the record

def test_success_record_lists_outputs(movie, tmp_path, monkeypatch):
    processor = make_processor()
    fake_xds(processor, monkeypatch, SUCCESS_FILES, succeed=True)
    assert processor.process_movie().exit_code() == 0

    record = status(tmp_path)
    assert record["schema_version"] == 1
    assert record["dataset"] == DATASET
    assert record["status"] == "success"
    assert record["source_file"] == str(movie)
    assert record["output_dir"] == str(tmp_path / DATASET)
    assert record["pyautoprocess_version"] == __version__
    auto_process = tmp_path / DATASET / "auto_process"
    assert record["outputs"]["XDS_ASCII.HKL"] == str(auto_process / "XDS_ASCII.HKL")
    assert record["outputs"][f"{DATASET}.hkl"] == str(auto_process / f"{DATASET}.hkl")
    assert "pointless.LP" not in record["outputs"]          # only files that exist


@pytest.mark.parametrize("produced, reason", [
    (["XDS.INP"], "indexing failed: no XPARM.XDS after 10 retries"),
    (["XDS.INP", "XPARM.XDS"], "integration failed: no INTEGRATE.HKL"),
    (["XDS.INP", "XPARM.XDS", "INTEGRATE.HKL"], "CORRECT did not complete: no CORRECT.LP"),
    (["XDS.INP", "XPARM.XDS", "INTEGRATE.HKL", "CORRECT.LP", "XDS_ASCII.HKL"],
     f"scaling or conversion failed: no {DATASET}.hkl"),
])
def test_failure_reason_names_the_stage(movie, tmp_path, monkeypatch, produced, reason):
    processor = make_processor()
    fake_xds(processor, monkeypatch, produced, succeed=False)
    assert processor.process_movie().exit_code() == 1
    record = status(tmp_path)
    assert record["status"] == "failed"
    assert record["reason"] == reason
    assert sorted(record["outputs"]) == sorted(produced)


def test_an_explicit_reason_wins_over_diagnosis(movie, tmp_path, monkeypatch):
    processor = make_processor()
    fake_xds(processor, monkeypatch, [], succeed=False, reason="no good-quality frames found (--dqa)")
    processor.process_movie()
    assert status(tmp_path)["reason"] == "no good-quality frames found (--dqa)"


def test_no_record_for_names_that_are_not_datasets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "20260513_98917_0_movie.ser").write_bytes(b"x")
    make_processor().process_movie()
    assert list((tmp_path / "autoprocess_logs").glob("*_status.json")) == []


# ------------------------------------------------------------------ remembered failures

def test_known_failure_is_skipped_with_exit_1_and_no_rerun(movie, tmp_path, monkeypatch):
    first = make_processor()
    fake_xds(first, monkeypatch, ["XDS.INP"], succeed=False)
    assert first.process_movie().exit_code() == 1

    second = make_processor()
    calls = fake_xds(second, monkeypatch, SUCCESS_FILES, succeed=True)
    summary = second.process_movie()
    assert calls == [], "XDS must not run again for a known failure"
    assert summary.previously_failed == 1 and summary.attempted == 0
    assert summary.exit_code() == 1
    assert status(tmp_path)["status"] == "failed"          # record left as it was


@pytest.mark.parametrize("flag", ["retry_failed", "reprocess"])
def test_retry_failed_and_reprocess_run_it_again(movie, tmp_path, monkeypatch, flag):
    first = make_processor()
    fake_xds(first, monkeypatch, ["XDS.INP"], succeed=False)
    first.process_movie()

    second = make_processor(**{flag: True})
    calls = fake_xds(second, monkeypatch, SUCCESS_FILES, succeed=True)
    assert second.process_movie().exit_code() == 0
    assert calls == [MOVIE]
    assert status(tmp_path)["status"] == "success"


def test_a_failure_record_for_another_source_file_is_ignored(movie, tmp_path, monkeypatch):
    """Two folders can hold a dataset of the same name; only an exact source match counts."""
    first = make_processor()
    fake_xds(first, monkeypatch, ["XDS.INP"], succeed=False)
    first.process_movie()
    path = tmp_path / "autoprocess_logs" / f"{DATASET}_status.json"
    record = json.loads(path.read_text())
    record["source_file"] = "/elsewhere/" + MOVIE
    path.write_text(json.dumps(record))

    second = make_processor()
    calls = fake_xds(second, monkeypatch, SUCCESS_FILES, succeed=True)
    assert second.process_movie().exit_code() == 0
    assert calls == [MOVIE]


def test_failure_is_not_written_to_the_tracking_log(movie, tmp_path, monkeypatch):
    """0.5.1 reads any tracking-log line as 'already processed', so failures stay out of it."""
    processor = make_processor()
    fake_xds(processor, monkeypatch, ["XDS.INP"], succeed=False)
    processor.process_movie()
    tracking = tmp_path / "autoprocess_logs" / "autoprocess_tracking.log"
    assert not tracking.exists() or tracking.read_text() == ""


# ------------------------------------------------------------------ already processed

def test_a_failed_reprocess_outranks_an_older_success(movie, tmp_path, monkeypatch):
    """Success, then --reprocess fails: the tracking log still lists the success, but the
    latest outcome is the failure, so a plain run must report it, not skip with exit 0."""
    first = make_processor()
    fake_xds(first, monkeypatch, SUCCESS_FILES, succeed=True)
    assert first.process_movie().exit_code() == 0

    again = make_processor(reprocess=True)
    fake_xds(again, monkeypatch, ["XDS.INP"], succeed=False)
    assert again.process_movie().exit_code() == 1

    plain = make_processor()
    calls = fake_xds(plain, monkeypatch, SUCCESS_FILES, succeed=True)
    summary = plain.process_movie()
    assert calls == [] and summary.previously_failed == 1 and summary.skipped == 0
    assert summary.exit_code() == 1
    assert status(tmp_path)["status"] == "failed"

    retry = make_processor(retry_failed=True)
    calls = fake_xds(retry, monkeypatch, SUCCESS_FILES, succeed=True)
    assert retry.process_movie().exit_code() == 0
    assert calls == [MOVIE]
    assert status(tmp_path)["status"] == "success"


def test_already_processed_keeps_the_success_record(movie, tmp_path, monkeypatch):
    first = make_processor()
    fake_xds(first, monkeypatch, SUCCESS_FILES, succeed=True)
    first.process_movie()
    before = status(tmp_path)

    second = make_processor()
    calls = fake_xds(second, monkeypatch, SUCCESS_FILES, succeed=True)
    assert second.process_movie().exit_code() == 0
    assert calls == []
    assert status(tmp_path) == before


def test_already_processed_by_an_older_version_gets_a_skipped_record(movie, tmp_path, monkeypatch):
    processor = make_processor()
    output = tmp_path / DATASET
    (output / "auto_process").mkdir(parents=True)
    (output / "auto_process" / "XDS_ASCII.HKL").write_text("x")
    processor._add_to_processed_files_log(movie, output)      # as 0.5.1 would have left it

    calls = fake_xds(processor, monkeypatch, SUCCESS_FILES, succeed=True)
    assert processor.process_movie().exit_code() == 0
    assert calls == []
    record = status(tmp_path)
    assert record["status"] == "skipped"
    assert record["reason"].startswith("already processed")
    assert "XDS_ASCII.HKL" in record["outputs"]


# ------------------------------------------------------------------ CLI

def test_retry_failed_flag_reaches_params(monkeypatch):
    from pyautoprocess.ui.cli_parser import parse_arguments
    monkeypatch.setattr(sys, "argv", ["autoprocess", "--retry-failed"])
    assert parse_arguments("autoprocess").retry_failed is True
    monkeypatch.setattr(sys, "argv", ["autoprocess"])
    assert parse_arguments("autoprocess").retry_failed is False

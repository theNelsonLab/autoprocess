"""
autoprocess and image_process must report failure through their exit status.

Both previously returned None from main(), so the shell always saw 0 -- including
after a dataset failed to index and produced no CORRECT.LP. monitorED checks the
child's return code, so it logged "Successfully processed" for datasets that had
produced nothing at all.

Codes: 0 success, 1 something failed, 2 the command itself was wrong, 3 nothing usable was
found. These are a stable contract for callers (REyes, monitorED): do not renumber.
"""
import sys

import pytest

from pyautoprocess.autoprocess import CrystallographyProcessor, ProcessingSummary
from pyautoprocess.config.config_manager import ConfigLoader


def make_processor(**overrides):
    params = ConfigLoader().get_config('default')
    params.paths = []
    for key, value in overrides.items():
        setattr(params, key, value)
    return CrystallographyProcessor(params)


# ------------------------------------------------------------------ the mapping

@pytest.mark.parametrize("summary, expected", [
    (ProcessingSummary(succeeded=3), 0),
    (ProcessingSummary(succeeded=2, failed=1), 1),
    (ProcessingSummary(failed=1), 1),
    (ProcessingSummary(skipped=4), 0),                       # all already done
    (ProcessingSummary(succeeded=1, skipped=2), 0),
    (ProcessingSummary(unparsable=1), 3),                    # nothing usable found
    (ProcessingSummary(), 3),                                # nothing at all
    (ProcessingSummary(previously_failed=1), 1),             # known failure, not retried
    (ProcessingSummary(succeeded=2, previously_failed=1), 1),
    # Skipped non-conventional names alongside real work are ordinary, not a failure:
    # in the lab archive only ~0.4% of .mrc names follow the convention.
    (ProcessingSummary(succeeded=1, unparsable=99), 0),
    (ProcessingSummary(skipped=1, unparsable=99), 0),
    (ProcessingSummary(failed=1, unparsable=99), 1),
    (ProcessingSummary(usage_error=True), 2),
    (ProcessingSummary(usage_error=True, succeeded=1), 2),   # usage error dominates
])
def test_exit_code_mapping(summary, expected):
    assert summary.exit_code() == expected


def test_exit_code_values_are_stable():
    from pyautoprocess import autoprocess as ap
    assert (ap.EXIT_OK, ap.EXIT_FAILED, ap.EXIT_USAGE, ap.EXIT_NO_INPUT) == (0, 1, 2, 3)


def test_attempted_counts_only_real_attempts():
    summary = ProcessingSummary(succeeded=2, failed=1, skipped=5, unparsable=3)
    assert summary.attempted == 3


# ------------------------------------------------------------------ end to end

def test_bare_sweep_of_an_empty_directory_is_no_input(tmp_path, monkeypatch):
    """REyes's call: cwd is the movie folder, no path. An empty folder must not look like success."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["autoprocess"])
    from pyautoprocess.autoprocess import main
    assert main() == 3


def test_being_pointed_at_a_missing_path_is_no_input(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["autoprocess", str(tmp_path / "nope.mrc")])
    from pyautoprocess.autoprocess import main
    assert main() == 3


def test_id_with_multiple_files_is_a_usage_error(tmp_path, monkeypatch):
    for name in ("a_960_0.3_3.mrc", "b_960_0.3_3.mrc"):
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["autoprocess", "--id", "X",
                                      str(tmp_path / "a_960_0.3_3.mrc"),
                                      str(tmp_path / "b_960_0.3_3.mrc")])
    from pyautoprocess.autoprocess import main
    assert main() == 2


def test_a_failed_dataset_is_counted_and_reported(tmp_path, monkeypatch):
    movie = tmp_path / "sample-mov1_960_0.3_3_P50toN-50.ser"
    movie.write_bytes(b"x")
    processor = make_processor(paths=[str(movie)])
    monkeypatch.setattr(processor, "_process_single_movie", lambda *a, **k: False)

    summary = processor.process_movie()
    assert summary.failed == 1 and summary.succeeded == 0
    assert summary.exit_code() == 1


def test_a_successful_dataset_exits_zero(tmp_path, monkeypatch):
    movie = tmp_path / "sample-mov1_960_0.3_3_P50toN-50.ser"
    movie.write_bytes(b"x")
    processor = make_processor(paths=[str(movie)])
    monkeypatch.setattr(processor, "_process_single_movie", lambda *a, **k: True)

    summary = processor.process_movie()
    assert summary.succeeded == 1 and summary.failed == 0
    assert summary.exit_code() == 0


def test_unparsable_filename_alone_is_no_input_not_success(tmp_path):
    """The user asked for this file; refusing to parse it is not success."""
    movie = tmp_path / "movie.ser"          # no underscore fields at all
    movie.write_bytes(b"x")
    summary = make_processor(paths=[str(movie)]).process_movie()
    assert summary.unparsable == 1
    assert summary.exit_code() == 3


def test_non_numeric_fields_are_rejected_up_front(tmp_path):
    """`20260513_98917_0_movie.ser` has four fields, but its "exposure" is `movie`.

    It is now refused at parse time rather than proceeding and crashing downstream,
    so it counts as unparsable rather than failed, and with nothing else usable the exit
    code is 3 (no input).
    """
    movie = tmp_path / "20260513_98917_0_movie.ser"
    movie.write_bytes(b"x")
    summary = make_processor(paths=[str(movie)]).process_movie()
    assert summary.unparsable == 1
    assert summary.failed == 0, "it should never reach processing"
    assert summary.exit_code() == 3


# --------------------------------------------------- tracking log vs failure

def test_a_failed_dataset_is_not_recorded_as_processed(tmp_path, monkeypatch):
    """Otherwise the next run skips it and reports success having done nothing.

    That combination -- run once and fail, run again and 'succeed' without work --
    is exactly what makes a failure exit code untrustworthy.
    """
    monkeypatch.chdir(tmp_path)
    movie = tmp_path / "sample-mov1_960_0.3_3_P50toN-50.ser"
    movie.write_bytes(b"x")

    processor = make_processor(paths=[str(movie)])
    recorded = []
    monkeypatch.setattr(processor, "_add_to_processed_files_log",
                        lambda *a, **k: recorded.append(a))
    monkeypatch.setattr(processor, "_setup_movie_directories", lambda *a, **k: tmp_path)
    monkeypatch.setattr(processor, "_reset_auto_process_for_reprocessing", lambda *a, **k: None)
    monkeypatch.setattr(processor, "_process_movie_data", lambda *a, **k: False)

    assert processor._process_single_movie(
        "sample-mov1", "960", "0.3", "3", 0.8, 1.0, movie.name, movie) is False
    assert recorded == [], "a failure must not be recorded as processed"


def test_a_successful_dataset_is_recorded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    movie = tmp_path / "sample-mov1_960_0.3_3_P50toN-50.ser"
    movie.write_bytes(b"x")

    processor = make_processor(paths=[str(movie)])
    recorded = []
    monkeypatch.setattr(processor, "_add_to_processed_files_log",
                        lambda *a, **k: recorded.append(a))
    monkeypatch.setattr(processor, "_setup_movie_directories", lambda *a, **k: tmp_path)
    monkeypatch.setattr(processor, "_reset_auto_process_for_reprocessing", lambda *a, **k: None)
    monkeypatch.setattr(processor, "_process_movie_data", lambda *a, **k: True)

    assert processor._process_single_movie(
        "sample-mov1", "960", "0.3", "3", 0.8, 1.0, movie.name, movie) is True
    assert len(recorded) == 1


# ------------------------------------------------------------- image_process

def test_image_process_returns_a_code_when_pointed_at_nothing(tmp_path, monkeypatch):
    from pyautoprocess.image_process import PreConvertedProcessor
    from pyautoprocess.ui.cli_parser import parse_arguments

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["image_process", str(tmp_path / "missing")])
    processor = PreConvertedProcessor(parse_arguments('image_process'))
    assert processor.process_all() == 3


def test_image_process_bare_sweep_of_empty_dir_is_no_input(tmp_path, monkeypatch):
    from pyautoprocess.image_process import PreConvertedProcessor
    from pyautoprocess.ui.cli_parser import parse_arguments

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["image_process"])
    processor = PreConvertedProcessor(parse_arguments('image_process'))
    assert processor.process_all() == 3


def test_extension_mismatch_explains_itself(tmp_path, capsys):
    """A .mrc under a .ser config is skipped -- the log must say why.

    This path now surfaces as a non-zero exit code, so "Could not parse filename"
    on its own would send someone hunting the wrong problem.
    """
    movie = tmp_path / "sample-mov1_960_0.3_3_P50toN-50.mrc"
    movie.write_bytes(b"x")
    processor = make_processor(paths=[str(movie)])
    messages = []
    processor.display.log_print = messages.append

    assert processor.parse_filename(movie.name) is None
    joined = "\n".join(messages)
    assert "extension does not match" in joined
    assert ".ser" in joined and "--file-extension" in joined

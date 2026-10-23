"""
The space group / unit cell lines and the XSCALE statistics in autoprocess.log.

Since 0.5.5 the "I found space group ... / cell" lines are printed AFTER XSCALE, directly above
"At <res> Å total statistics are:" and the totals read from XSCALE.LP. "Proceeding with scaling"
stays where it was. Every way out of the scaling step still logs the space group and cell.

XDS, XSCALE and XDSCONV are replaced by fakes; the processor runs in a prepared working dir.
"""
import logging
import shutil
from pathlib import Path

import pytest

from pyautoprocess import autoprocess as ap
from pyautoprocess.autoprocess import CrystallographyProcessor
from pyautoprocess.config.config_manager import ConfigLoader

FIXTURE = Path(__file__).parent / "data" / "xscale" / "XSCALE.LP"
SG = "16"
CELL = "3.72 3.96 13.88 90.000 90.000 90.000"

EXPECTED_BLOCK = f"""\
I scaled the data in XSCALE.

I found space group 16 and a unit cell of
{CELL}

At 0.80 Å total statistics are:
Observed reflections: 800
Unique reflections: 209
Possible: 288
Completeness: 72.6%
R-factor: 20.2%
R-meas: 23.1%
I/sigma: 4.43
CC1/2: 95.1%

I converted it for use in shelx!
Pointless analysis skipped (--pointless flag not set)"""


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "CORRECT.LP").write_text("fake CORRECT.LP\n")
    (tmp_path / "XDS.INP").write_text("INCLUDE_RESOLUTION_RANGE= 40 0.78\n")
    return tmp_path


@pytest.fixture
def log(caplog):
    caplog.set_level(logging.INFO)

    def text():
        return "\n".join(record.getMessage() for record in caplog.records)
    return text


def make_processor():
    params = ConfigLoader().get_config('default')
    params.paths = []
    return CrystallographyProcessor(params)


def fake_run(monkeypatch, xscale="copy"):
    """Replace subprocess.run inside autoprocess. xscale: 'copy' the fixture to XSCALE.LP,
    'nothing' (XSCALE ran but wrote no XSCALE.LP), or an exception instance to raise."""
    calls = []

    def run(command, *args, **kwargs):
        calls.append(command)
        if command == "xscale":
            if isinstance(xscale, BaseException):
                raise xscale
            if xscale == "copy":
                shutil.copy(FIXTURE, "XSCALE.LP")
        return None
    monkeypatch.setattr(ap, "run", run)
    return calls


def test_success_block_text_and_order(workdir, log, monkeypatch):
    calls = fake_run(monkeypatch)
    assert make_processor().scale_conv("sample-mov1", SG, CELL) is True
    assert calls == ["xscale", "xdsconv"]
    text = log()
    assert EXPECTED_BLOCK in text
    # the cell line is immediately followed by a blank line and the statistics header
    assert f"{CELL}\n\nAt 0.80 Å total statistics are:" in text


def test_xscale_inp_is_unchanged(workdir, log, monkeypatch):
    fake_run(monkeypatch)
    make_processor().scale_conv("sample-mov1", SG, CELL)
    assert Path("XSCALE.INP").read_text() == (
        "OUTPUT_FILE= sample-mov1.ahkl\n"
        "INPUT_FILE= XDS_ASCII.HKL\n"
        "RESOLUTION_SHELLS= 10 8 5 3 2.3 2.0 1.7 1.5 1.3 1.2 1.1 1.0 0.9 0.8 !0.75 !0.7 !0.65 !0.6\n")


def test_full_flow_from_iterate_opt(workdir, log, monkeypatch):
    """Drive iterate_opt -> scale_conv: 'Proceeding with scaling' keeps its place, the
    space group / cell move below 'I scaled the data in XSCALE.', stats.LP is unchanged."""
    fake_run(monkeypatch)
    Path("XDS.LP").write_text(
        "     a        b          ISa\n"
        " 1.000E+00  1.000E-03    10.00\n")
    processor = make_processor()
    monkeypatch.setattr(processor, "_run_xds", lambda message: None)
    monkeypatch.setattr(processor, "check_space_group_before_scaling", lambda movie: True)
    monkeypatch.setattr(processor, "_get_crystal_parameters", lambda: (SG, CELL))

    assert processor.iterate_opt("sample-mov1") is True
    text = log()
    order = [text.index(s) for s in (
        "Optimized beam divergence values.",
        "\nProceeding with scaling",
        "Using resolution from XDS.INP for XSCALE: 0.78 Å",
        "I scaled the data in XSCALE.",
        "I found space group 16 and a unit cell of",
        "At 0.80 Å total statistics are:",
        "CC1/2: 95.1%",
        "I converted it for use in shelx!")]
    assert order == sorted(order)
    assert text.count("I found space group") == 1
    assert EXPECTED_BLOCK in text
    assert Path("stats.LP").read_text() == f"{SG}\n{CELL}"


def test_xscale_wrote_no_xscale_lp(workdir, log, monkeypatch):
    fake_run(monkeypatch, xscale="nothing")
    assert make_processor().scale_conv("sample-mov1", SG, CELL) is True   # behaviour unchanged
    assert (f"I scaled the data in XSCALE.\n\nI found space group 16 and a unit cell of\n{CELL}\n\n"
            "XSCALE statistics are not available: XSCALE.LP not found\n\n"
            "I converted it for use in shelx!") in log()


def test_stale_xscale_lp_is_not_reported(workdir, log, monkeypatch):
    shutil.copy(FIXTURE, "XSCALE.LP")           # left over from an earlier run
    fake_run(monkeypatch, xscale="nothing")
    make_processor().scale_conv("sample-mov1", SG, CELL)
    text = log()
    assert "XSCALE statistics are not available: XSCALE.LP was not updated by this XSCALE run" in text
    assert "total statistics are" not in text
    assert f"I found space group 16 and a unit cell of\n{CELL}" in text


def test_unparsable_xscale_lp(workdir, log, monkeypatch):
    def run(command, *args, **kwargs):
        if command == "xscale":
            Path("XSCALE.LP").write_text(" ***** XSCALE *****\n !!! ERROR !!! nothing scaled\n")
    monkeypatch.setattr(ap, "run", run)
    make_processor().scale_conv("sample-mov1", SG, CELL)
    assert (f"{CELL}\n\nXSCALE statistics are not available: no statistics table in XSCALE.LP"
            in log())


def test_xscale_cannot_start(workdir, log, monkeypatch):
    """xscale not on PATH: the exception still propagates as before, but SG/cell are logged."""
    fake_run(monkeypatch, xscale=FileNotFoundError("xscale"))
    with pytest.raises(FileNotFoundError):
        make_processor().scale_conv("sample-mov1", SG, CELL)
    text = log()
    assert f"I found space group 16 and a unit cell of\n{CELL}" in text
    assert "XSCALE statistics are not available: XSCALE did not run (FileNotFoundError: xscale)" in text
    assert "I scaled the data in XSCALE." not in text


def test_no_correct_lp_still_logs_space_group(workdir, log, monkeypatch):
    calls = fake_run(monkeypatch)
    Path("CORRECT.LP").unlink()
    assert make_processor().scale_conv("sample-mov1", SG, CELL) is False
    assert calls == []
    assert (f"I found space group 16 and a unit cell of\n{CELL}\n\n"
            "XSCALE statistics are not available: CORRECT.LP not found, so XSCALE was not run"
            in log())


def test_parser_crash_never_fails_the_run(workdir, log, monkeypatch):
    fake_run(monkeypatch)

    def boom(path):
        raise RuntimeError("boom")
    monkeypatch.setattr(ap, "parse_xscale_lp", boom)
    assert make_processor().scale_conv("sample-mov1", SG, CELL) is True
    text = log()
    assert "XSCALE statistics are not available: unexpected error (boom)" in text
    assert "I converted it for use in shelx!" in text


def test_scale_conv_old_signature_still_works(workdir, log, monkeypatch):
    """External callers using scale_conv(sample_movie) get the statistics without SG lines."""
    fake_run(monkeypatch)
    assert make_processor().scale_conv("sample-mov1") is True
    text = log()
    assert "I found space group" not in text
    assert "I scaled the data in XSCALE.\n\nAt 0.80 Å total statistics are:" in text


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), SystemExit(1)])
def test_interrupt_during_xscale_still_logs_space_group(workdir, log, monkeypatch, interrupt):
    """Ctrl-C (or SystemExit) while XSCALE runs: SG/cell are logged, the exception passes on."""
    fake_run(monkeypatch, xscale=interrupt)
    with pytest.raises(type(interrupt)):
        make_processor().scale_conv("sample-mov1", SG, CELL)
    text = log()
    assert f"I found space group 16 and a unit cell of\n{CELL}" in text
    assert f"XSCALE statistics are not available: XSCALE did not run ({type(interrupt).__name__}" in text

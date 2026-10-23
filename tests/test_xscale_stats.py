"""
The XSCALE.LP parser shared by the autoprocess.log statistics block and autoprocess-summary.

Fixtures (tests/data/xscale/) are trimmed XSCALE.LP files: XSCALE.LP has XSCALE's layout and
realistic numbers, XSCALE_weak.LP is synthetic weak data (-99.9% R-factors, CC(1/2) with
no significance mark, last shell at 1.00 A). The edge cases are built from them in the tests.
"""
from pathlib import Path

import pytest

from pyautoprocess.core.xscale_stats import (
    XscaleRow, format_log_block, parse_xscale_lp)

DATA = Path(__file__).parent / "data" / "xscale"
FULL = DATA / "XSCALE.LP"
WEAK = DATA / "XSCALE_weak.LP"

EXPECTED_BLOCK = """\
At 0.80 Å total statistics are:
Observed reflections: 800
Unique reflections: 209
Possible: 288
Completeness: 72.6%
R-factor: 20.2%
R-meas: 23.1%
I/sigma: 4.43
CC1/2: 95.1%"""


def write(tmp_path, text, name="XSCALE.LP"):
    path = tmp_path / name
    path.write_text(text)
    return path


def without_lines(text, predicate):
    return "\n".join(line for line in text.splitlines() if not predicate(line)) + "\n"


# ------------------------------------------------------------------ normal files

def test_full_file_values():
    s = parse_xscale_lp(FULL)
    assert s.ok and s.error is None
    assert s.sg == "16"
    assert s.cell == "3.72     3.96    13.88  90.000  90.000  90.000"
    assert s.resolution == "0.80"
    assert len(s.shells) == 13
    assert (s.observed, s.unique, s.possible, s.compared) == (800, 209, 288, 792)
    assert (s.completeness, s.r_obs, s.r_exp) == (72.6, 20.2, 21.2)
    assert (s.i_sigma, s.r_meas, s.cc_half, s.cc_significant) == (4.43, 23.1, 95.1, True)
    assert s.total.resolution == "total"


def test_full_file_log_block_matches_expected_format():
    assert "\n".join(format_log_block(parse_xscale_lp(FULL))) == EXPECTED_BLOCK


def test_table_text_is_the_signal_noise_table_only():
    s = parse_xscale_lp(FULL)
    lines = s.table_text.splitlines()
    assert lines[0].strip().startswith("SUBSET OF INTENSITY DATA WITH SIGNAL/NOISE >= -3.0")
    assert lines[-1].split()[:2] == ["total", "800"]
    assert "R-FACTORS FOR INTENSITIES" not in s.table_text


def test_weak_data_minus_99_9_and_cc_without_star():
    s = parse_xscale_lp(WEAK)
    assert s.ok
    assert s.sg == "1"
    assert s.resolution == "1.00"          # the last shell reported, not a fixed 0.80
    assert s.r_obs == -99.9 and s.r_meas == -99.9
    assert s.cc_half == -12.0 and s.cc_significant is False
    block = format_log_block(s)
    assert block[0] == "At 1.00 Å total statistics are:"
    assert "R-factor: -99.9%" in block
    assert "R-meas: -99.9%" in block
    assert "I/sigma: 0.00" in block          # printed as XSCALE printed it, not 0.0
    assert block[-1] == "CC1/2: -12.0%"


def test_significance_is_per_row():
    s = parse_xscale_lp(FULL)
    assert s.shells[2].cc_significant is False   # 3.00 A: 85.5, no '*'
    assert s.shells[3].cc_significant is True    # 2.30 A: 99.1*
    assert s.shells[3].text("cc_half") == "99.1*"
    assert s.shells[3].percent("cc_half") == "99.1%"


# ------------------------------------------------------------------ edge cases

def test_missing_file(tmp_path):
    s = parse_xscale_lp(tmp_path / "XSCALE.LP")
    assert not s.ok
    assert s.error == "XSCALE.LP not found"
    with pytest.raises(ValueError):
        format_log_block(s)


def test_unreadable_path(tmp_path):
    folder = tmp_path / "XSCALE.LP"
    folder.mkdir()
    s = parse_xscale_lp(folder)
    assert not s.ok and s.error.startswith("could not read XSCALE.LP")


def test_no_table(tmp_path):
    text = FULL.read_text().split(" SUBSET OF INTENSITY")[0]
    s = parse_xscale_lp(write(tmp_path, text))
    assert s.error == "no statistics table in XSCALE.LP"
    assert s.sg == "16"                       # header values still read
    assert s.total is None and s.shells == []


def test_truncated_table_at_end_of_file(tmp_path):
    text = FULL.read_text().split("    total         800")[0]
    s = parse_xscale_lp(write(tmp_path, text))
    assert not s.ok
    assert "no total row" in s.error
    assert s.resolution == "0.80" and len(s.shells) == 13
    assert s.observed is None


def test_table_without_total_does_not_borrow_a_later_total(tmp_path):
    """The R-FACTORS table further down has its own 'total' row; it must not be used."""
    text = without_lines(FULL.read_text(), lambda line: line.strip().startswith("total         800"))
    s = parse_xscale_lp(write(tmp_path, text))
    assert s.total is None and "no total row" in s.error


def test_truncated_mid_row(tmp_path):
    text = FULL.read_text().split("     0.90         150")[0] + "     0.90         150      38"
    s = parse_xscale_lp(write(tmp_path, text))
    assert s.total is None
    assert s.shells[-1].resolution == "0.90"
    assert s.shells[-1].unique == 38 and s.shells[-1].possible is None


def test_multiple_tables_uses_the_last(tmp_path):
    s = parse_xscale_lp(write(tmp_path, FULL.read_text() + WEAK.read_text()))
    assert s.ok
    assert s.observed == 41450 and s.resolution == "1.00"
    assert s.sg == "1"     # the last 'REPORTED BELOW ASSUMES' section, matching the last table


def test_total_row_with_fewer_columns(tmp_path):
    full_total = next(line for line in FULL.read_text().splitlines()
                      if line.strip().startswith("total         800"))
    text = FULL.read_text().replace(full_total, "    total         800     209       288       72.6%")
    s = parse_xscale_lp(write(tmp_path, text))
    assert s.ok
    assert s.observed == 800 and s.completeness == 72.6
    assert s.r_obs is None and s.cc_half is None and s.cc_significant is None
    block = format_log_block(s)
    assert "Completeness: 72.6%" in block
    assert "R-factor: n/a" in block and "I/sigma: n/a" in block and block[-1] == "CC1/2: n/a"


def test_garbled_token_keeps_text_but_no_number():
    row = XscaleRow.from_line("total  800  209  288  72.6%  ****%  21.2%  792  4.43  23.1%  95.1*")
    assert row.r_obs is None
    assert row.text("r_obs") == "****%"
    assert row.observed == 800


def test_header_without_assumes_section(tmp_path):
    text = FULL.read_text().replace(" THE DATA COLLECTION STATISTICS REPORTED BELOW ASSUMES:\n", "")
    s = parse_xscale_lp(write(tmp_path, text))
    assert s.sg == "16" and s.cell.startswith("3.72")

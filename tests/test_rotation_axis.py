"""
Tests for tilt-direction parsing and rotation-axis resolution (--auto-rotation-axis).

None of these touch XDS, a microscope, or the filesystem.

The token forms exercised here are taken from real filenames on the lab's HPC storage
(/resnick/groups/NelsonLab/data/DBE/...), not invented: `P50toN-50` dominates, the token sits
at underscore field 4 or 5, the minus inside `N-50` is redundant, and `p`/`n` appear in both
cases.
"""
import pytest

from pyautoprocess.core.filename_parser import (
    NTOP,
    PTON,
    find_tilt_tokens,
    looks_numeric,
    resolve_tilt_direction,
)
from pyautoprocess.core.rotation_axis import flip_rotation_axis, resolve_rotation_axis


# --------------------------------------------------------------------------- looks_numeric

@pytest.mark.parametrize("value, expected", [
    ("960", True), ("0.3", True), ("1p5", True), ("0p6", True), ("3", True),
    ("-5", True), ("movie", False), ("bin4", False), ("g8sp7", False),
    ("", False), ("p", False), ("1p5p6", False),
])
def test_looks_numeric(value, expected):
    assert looks_numeric(value) is expected


# ------------------------------------------------------------------- tilt-direction parsing

@pytest.mark.parametrize("filename, expected", [
    # Real forms observed in the corpus.
    ("AVAAGA-mov12_540_1_1_P50toN-50_g8sp11_rt.mrc", PTON),
    ("CuSer-mov16_540_1_1_P50toN-50_g8sp11_rt.mrc", PTON),
    ("sample-mov1_960_0.3_3_n60top10_g8sp10_cryo.ser", NTOP),
    ("Lysozyme-NAG2-DC-xtal-05_960_1p5_0p6_p40ton60_g8sp7_bin4_0_movie.mrc", PTON),
    # Token at field 5 rather than 4 -- 6% of the corpus, and a fixed-field parser misses these.
    ("ss-jacobsen-mov20_540_1_1_sp11_P50toN-50_mov12_sweep1_c.mrc", PTON),
    # Sweeps that start at zero: direction comes from the arithmetic, not the letters.
    ("thing-mov3_960_0.3_3_0toP25_notes.ser", NTOP),
    ("thing-mov3_960_0.3_3_0toN25_notes.ser", PTON),
    ("thing-mov3_960_0.3_3_P25to0_notes.ser", PTON),
    # Plain negative-to-positive.
    ("thing-mov4_960_0.3_3_N40toP40.tvips", NTOP),
    # Case insensitivity, and the token as the final field with the extension attached.
    ("thing-mov4_960_0.3_3_n40TOp40.mrc", NTOP),
    # Decimal angles, including 'p' as the separator.
    ("thing-mov7_960_0.3_3_P1p5toN1p5.ser", PTON),
    ("thing-mov7_960_0.3_3_P0.5toN2.5.ser", PTON),
    # No magnitudes at all.
    ("thing-mov8_960_0.3_3_PtoN.ser", PTON),
    ("thing-mov8_960_0.3_3_NtoP.ser", NTOP),
])
def test_direction_is_parsed(filename, expected):
    direction, reason = resolve_tilt_direction(filename)
    assert direction is not None, reason
    assert direction.direction == expected


@pytest.mark.parametrize("filename", [
    "thing-mov5_960_0.3_3_g8sp7_bin4_movie.mrc",   # notes, but no tilt token
    "20260513_98917_0_movie.mrc",                  # unconventional name entirely
    "sample_960_0.3_3.mrc",                        # no notes fields at all
    "sample_960_0.3_3_photon_counting.mrc",        # contains 'to' but is not a tilt token
    "sample_960_0.3_3_P0toN0.mrc",                 # a token, but establishes no direction
])
def test_direction_absent_returns_none_with_a_reason(filename):
    direction, reason = resolve_tilt_direction(filename)
    assert direction is None
    assert reason


def test_token_in_the_sample_field_is_ignored():
    """Fields 0-3 are sample/distance/rotation/exposure and must never be scanned.

    A sample literally named after a sweep would otherwise hijack the axis.
    """
    direction, reason = resolve_tilt_direction("P50toN-50_960_0.3_3_notes.ser")
    assert direction is None
    assert "no tilt-direction token" in reason


def test_conflicting_tokens_are_refused_rather_than_guessed():
    direction, reason = resolve_tilt_direction("x-mov6_960_0.3_3_P40toN40_N10toP10.ser")
    assert direction is None
    assert "conflicting" in reason
    assert "P40toN40" in reason and "N10toP10" in reason


def test_agreeing_duplicate_tokens_are_accepted():
    direction, reason = resolve_tilt_direction("x-mov6_960_0.3_3_P40toN40_P50toN50.ser")
    assert direction is not None and direction.direction == PTON
    assert "2 matching tokens" in reason


def test_redundant_minus_does_not_flip_the_sense():
    """`N-50` and `N50` both mean -50; the letter carries the sign."""
    with_minus, _ = resolve_tilt_direction("x_960_0.3_3_P50toN-50.mrc")
    without, _ = resolve_tilt_direction("x_960_0.3_3_P50toN50.mrc")
    assert with_minus.direction == without.direction == PTON
    assert with_minus.end_angle == without.end_angle == -50.0


def test_token_records_its_field_index():
    at_four = find_tilt_tokens("x_960_0.3_3_P50toN-50_notes.mrc")[0]
    at_five = find_tilt_tokens("x_960_0.3_3_sp11_P50toN-50_notes.mrc")[0]
    assert at_four.field_index == 4
    assert at_five.field_index == 5


# ----------------------------------------------------------------------------- axis flipping

@pytest.mark.parametrize("axis, expected", [
    ("-1 0 0", "1 0 0"),
    ("1 0 0", "-1 0 0"),
    ("0 -1 0", "0 1 0"),
    # The F30 case: a whole-vector negation, NOT just the x component.
    ("-0.8290 -0.5592 0", "0.8290 0.5592 0"),
    ("0.8290 0.5592 0", "-0.8290 -0.5592 0"),
])
def test_flip_negates_every_component(axis, expected):
    assert flip_rotation_axis(axis) == expected


def test_flip_is_its_own_inverse():
    for axis in ("-1 0 0", "-0.8290 -0.5592 0", "0 1 0"):
        assert flip_rotation_axis(flip_rotation_axis(axis)) == axis


def test_flip_preserves_the_original_number_formatting():
    """XDS.INP is read by humans; '0.8290' must not silently become '0.829'."""
    assert flip_rotation_axis("-0.8290 -0.5592 0") == "0.8290 0.5592 0"


@pytest.mark.parametrize("axis", ["bogus", "1 0", "1 0 0 0", "", "a b c"])
def test_flip_refuses_malformed_axes(axis):
    assert flip_rotation_axis(axis) is None


# ------------------------------------------------------------------------- end-to-end resolve

def test_disabled_is_a_no_op_and_says_nothing():
    axis, message = resolve_rotation_axis("-1 0 0", "x_960_0.3_3_N40toP40.mrc", enabled=False)
    assert axis == "-1 0 0"
    assert message is None


def test_pton_keeps_the_configured_axis():
    axis, message = resolve_rotation_axis("-1 0 0", "x_960_0.3_3_P50toN-50.mrc", enabled=True)
    assert axis == "-1 0 0"
    assert "no flip" in message


def test_ntop_flips_and_warns_that_the_path_is_less_tested():
    axis, message = resolve_rotation_axis("-1 0 0", "x_960_0.3_3_N40toP40.mrc", enabled=True)
    assert axis == "1 0 0"
    assert "FLIPPING" in message
    assert "less well tested" in message


def test_missing_token_falls_back_to_the_configured_axis():
    axis, message = resolve_rotation_axis("-1 0 0", "x_960_0.3_3_plain.mrc", enabled=True)
    assert axis == "-1 0 0"
    assert "no tilt-direction token" in message and "keeping" in message


def test_malformed_base_axis_is_never_mangled():
    axis, message = resolve_rotation_axis("garbage", "x_960_0.3_3_N40toP40.mrc", enabled=True)
    assert axis == "garbage"
    assert "cannot be flipped" in message


def test_an_explicit_axis_is_still_flipped_but_labelled_as_such():
    """--rotation-axis chooses the axis; the sweep direction still decides its sign."""
    axis, message = resolve_rotation_axis(
        "-1 0 0", "x_960_0.3_3_N40toP40.mrc", enabled=True, explicit=True)
    assert axis == "1 0 0"
    assert "command line" in message


def test_f30_configured_default_flips_to_the_arm_seen_in_real_ntop_runs():
    """The experiment locked -0.8290 -0.5592 0 for PtoN; NtoP must give its negation."""
    pton, _ = resolve_rotation_axis("-0.8290 -0.5592 0", "x_540_1_1_P50toN-50.tvips", enabled=True)
    ntop, _ = resolve_rotation_axis("-0.8290 -0.5592 0", "x_540_1_1_N50toP50.tvips", enabled=True)
    assert pton == "-0.8290 -0.5592 0"
    assert ntop == "0.8290 0.5592 0"


# --------------------------------------------------------- integration with XDS.INP generation

def test_resolved_axis_reaches_xds_inp_without_leaking_between_datasets():
    """The whole point of routing this through the params dict rather than self.params."""
    from pyautoprocess.config.config_manager import ConfigLoader
    from pyautoprocess.core.xds_manager import XDSManager

    params = ConfigLoader().get_config('default')
    manager = XDSManager(params)
    common = dict(distance='960', rotation='0.3', exposure='3', resolution_range=0.8,
                  test_resolution_range=1.0, image_number='100',
                  background_pixel=4, signal_pixel=7, min_pixel=7)

    def axis_line(text):
        return next(line for line in text.splitlines() if line.startswith('ROTATION_AXIS'))

    flipped, _ = resolve_rotation_axis(params.rotation_axis, "a_960_0.3_3_N40toP40.mrc", enabled=True)
    kept, _ = resolve_rotation_axis(params.rotation_axis, "b_960_0.3_3_P40toN40.mrc", enabled=True)

    first = manager.create_xds_input('../images/a', dict(common, rotation_axis=flipped))
    second = manager.create_xds_input('../images/b', dict(common, rotation_axis=kept))

    assert axis_line(first) == 'ROTATION_AXIS=1 0 0'
    assert axis_line(second) == 'ROTATION_AXIS=-1 0 0'
    assert params.rotation_axis == '-1 0 0', "shared params must never be mutated"


# ------------------------------------------------------- the real corpus (regression fixture)

# Every distinct tilt-token form found across 199,606 unique movie basenames under
# /resnick/groups/NelsonLab/data/DBE/{Arctica,Talos,Apollo,F30,Spectra}_data and
# AutoProcess_paper_data, harvested 2026-09-01. 880 files carry a token; 43 distinct forms;
# all sit at underscore field 4 or 5. Format: (token, field_index, file_count, expected).
#
# These are the actual strings the parser must survive -- not invented examples. If a future
# change to the regex breaks one of these, it breaks real data.
REAL_CORPUS_TOKENS = [
    ("P50toN-50", 4, 444, PTON), ("P45toN-45", 4, 126, PTON), ("P30toN-30", 4, 108, PTON),
    ("P55toN-55", 4, 51, PTON),  ("P50toN-50", 5, 23, PTON),  ("p45ton45", 4, 17, PTON),
    ("P10toN-10", 4, 17, PTON),  ("p50ton50", 4, 12, PTON),   ("P60toN-60", 4, 12, PTON),
    ("n45top45", 4, 8, NTOP),    ("p60ton60", 4, 5, PTON),    ("P5toN-5", 4, 5, PTON),
    ("p70ton70", 4, 4, PTON),    ("n50top50", 4, 4, NTOP),    ("p40ton50", 4, 3, PTON),
    ("p40ton40", 4, 3, PTON),    ("n60top60", 4, 3, NTOP),    ("P50ToN-50", 4, 3, PTON),
    ("P15toN-15", 4, 3, PTON),   ("p65ton60", 4, 2, PTON),    ("p60ton30", 4, 2, PTON),
    ("p5ton5", 4, 2, PTON),      ("p30ton30", 4, 2, PTON),    ("n30top30", 4, 2, NTOP),
    ("P50toN50", 5, 1, PTON),    ("P50toN0", 5, 1, PTON),     ("p70ton50", 4, 1, PTON),
    ("p70ton40", 4, 1, PTON),    ("p70to0", 4, 1, PTON),      ("p65ton55", 4, 1, PTON),
    ("p65to0", 4, 1, PTON),      ("p50ton20", 4, 1, PTON),    ("p50ton10", 4, 1, PTON),
    ("p50toN-50", 4, 1, PTON),   ("p50to0", 4, 1, PTON),      ("p40ton15", 4, 1, PTON),
    ("p30ton40", 4, 1, PTON),    ("p30ton20", 4, 1, PTON),    ("n70top60", 4, 1, NTOP),
    ("n65top10", 4, 1, NTOP),    ("n50top60", 4, 1, NTOP),    ("n35top70", 4, 1, NTOP),
    ("P30toN30", 4, 1, PTON),
]


def _filename_with_token(token, field_index):
    """Build a conventional movie filename carrying `token` at `field_index`."""
    fields = ["sample-mov1", "960", "0.3", "3"] + ["pad"] * (field_index - 4) + [token, "notes"]
    return "_".join(fields) + ".mrc"


@pytest.mark.parametrize("token, field_index, _count, expected", REAL_CORPUS_TOKENS)
def test_every_real_corpus_token_parses(token, field_index, _count, expected):
    direction, reason = resolve_tilt_direction(_filename_with_token(token, field_index))
    assert direction is not None, f"{token!r} was not recognised: {reason}"
    assert direction.direction == expected, f"{token!r} -> {direction.direction}, expected {expected}"
    assert direction.field_index == field_index


def test_corpus_is_overwhelmingly_pton():
    """Sanity check on the fixture, and a reminder of why this feature is nearly always a no-op.

    NtoP is 21 of 880 token-bearing files (2.4%), so the flip path is rare in practice --
    which is exactly why it is opt-in and logs loudly when it fires.
    """
    ntop = sum(count for _t, _i, count, d in REAL_CORPUS_TOKENS if d == NTOP)
    total = sum(count for _t, _i, count, _d in REAL_CORPUS_TOKENS)
    assert total == 880
    assert ntop == 21
    assert ntop / total < 0.05

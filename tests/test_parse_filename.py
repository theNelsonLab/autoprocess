"""
parse_filename: what counts as a name following the convention.

    sample-name_distance_rotation_exposure[_notes...].ext

Four underscore-separated fields are not sufficient on their own. Fields 1-3 must
actually read as numbers, checked with the same `looks_numeric` monitorED uses, so
the two parsers agree on what a valid name is. Before that check existed,
`20260513_98917_0_movie.ser` parsed with an "exposure" of the word `movie` and
carried it to a downstream crash.
"""
import pytest

from pyautoprocess.autoprocess import CrystallographyProcessor
from pyautoprocess.config.config_manager import ConfigLoader


def make_processor(**overrides):
    params = ConfigLoader().get_config('default')
    for key, value in overrides.items():
        setattr(params, key, value)
    processor = CrystallographyProcessor(params)
    processor.messages = []
    processor.display.log_print = processor.messages.append
    return processor


# ------------------------------------------------------------------- accepted

@pytest.mark.parametrize("filename, expected", [
    # Minimal: exactly the four fields and nothing else.
    ("sample_960_0.3_3.ser", ("sample", "960", "0.3", "3")),
    # With trailing notes, which is what real data mostly looks like.
    ("sample-mov1_960_0.3_3_n60top10_g8sp10_cryo.ser",
     ("sample-mov1", "960", "0.3", "3")),
    # 'p' as the decimal separator, in any field.
    ("Lyso_960_1p5_0p6_p40ton60.ser", ("Lyso", "960", "1.5", "0.6")),
    ("x_960_1p5_0p6.ser", ("x", "960", "1.5", "0.6")),
    # Hyphens in the sample name are fine; underscores are not.
    ("Lysozyme-NAG2-DC-xtal-05_960_1_1.ser", ("Lysozyme-NAG2-DC-xtal-05", "960", "1", "1")),
])
def test_conventional_names_parse(filename, expected):
    assert make_processor().parse_filename(filename) == expected


def test_minimal_name_does_not_carry_the_extension_into_the_exposure():
    """Regression: splitting the raw filename made the last field '3.ser'.

    That string was then handed to float() as the exposure. Splitting the stem
    instead fixes it -- and it is why the numeric check had to come with a stem
    split rather than on its own.
    """
    parsed = make_processor().parse_filename("sample_960_0.3_3.ser")
    assert parsed is not None
    assert parsed[3] == "3", "exposure must not include the file extension"
    float(parsed[3])  # must be usable as a number


# ------------------------------------------------------------------- rejected

def test_non_numeric_field_is_rejected_with_a_specific_message():
    processor = make_processor()
    assert processor.parse_filename("20260513_98917_0_movie.ser") is None
    log = "\n".join(processor.messages)
    assert "field 3 = 'movie'" in log
    assert "not numeric" in log
    assert "--id" in log, "the message should point at the way out"


def test_underscore_in_the_sample_name_is_rejected_and_explained():
    """It shifts every field along, so 'distance' becomes part of the name."""
    processor = make_processor()
    assert processor.parse_filename("my_sample_960_0.3_3.ser") is None
    log = "\n".join(processor.messages)
    assert "field 1 = 'sample'" in log
    assert "'_'" in log


def test_name_with_too_few_fields_is_rejected():
    processor = make_processor()
    assert processor.parse_filename("movie.ser") is None
    assert any("unexpected filename format" in m for m in processor.messages)


def test_extension_mismatch_is_reported_separately():
    """A .mrc under a .ser config is not a malformed name -- say which it is."""
    processor = make_processor()
    assert processor.parse_filename("sample_960_0.3_3.mrc") is None
    log = "\n".join(processor.messages)
    assert "extension does not match" in log
    assert "not numeric" not in log


def test_this_matches_what_monitored_would_accept():
    """The two parsers must agree, or monitorED forwards files autoprocess rejects."""
    from pyautoprocess.monitor_ed import MonitorED

    processor = make_processor()
    for filename in ("sample_960_0.3_3.ser",
                     "sample-mov1_960_0.3_3_n60top10.ser",
                     "Lyso_960_1p5_0p6_p40ton60.ser"):
        assert MonitorED.validate_movie_filename(filename)
        assert processor.parse_filename(filename) is not None

    for filename in ("20260513_98917_0_movie.ser", "my_sample_960_0.3_3.ser"):
        assert not MonitorED.validate_movie_filename(filename)
        assert processor.parse_filename(filename) is None


# ----------------------------------------------------------------- --id path

def test_id_accepts_an_unconventional_name_using_config_defaults():
    """--id says "this name does not follow the convention; take the values elsewhere"."""
    processor = make_processor(sample_id="mysample")
    parsed = processor.parse_filename("20260513_98917_0_movie.ser")
    assert parsed is not None
    sample, distance, rotation, exposure = parsed
    assert sample == "mysample"
    # Falls through to the microscope-config defaults rather than the filename's junk.
    assert (distance, rotation, exposure) == (
        processor.params.default_detector_distance,
        processor.params.default_rotation,
        processor.params.default_exposure,
    )


def test_id_with_cli_overrides_beats_the_config_defaults():
    processor = make_processor(sample_id="mysample", detector_distance="1200",
                               rotation="0.5", exposure="2")
    assert processor.parse_filename("whatever.ser") == ("mysample", "1200", "0.5", "2")


def test_id_still_fails_when_nothing_can_supply_the_numbers():
    processor = make_processor(sample_id="mysample")
    processor.params.default_detector_distance = None
    processor.params.default_rotation = None
    processor.params.default_exposure = None
    assert processor.parse_filename("whatever.ser") is None
    assert any("missing" in m for m in processor.messages)

"""
Integration tests for --beam-center: frame extraction, the resolution-limit dependency, and
the provenance record. No XDS, no microscope, no network.

The subtle failure this file exists to catch is the SER vertical flip: detection must run on
the frame orientation that reaches the TIFs XDS reads, not the raw file orientation. Getting
that wrong produces a confidently wrong ORGY rather than an obvious error.
"""
import numpy as np
import pytest

from pyautoprocess.autoprocess import CrystallographyProcessor
from pyautoprocess.config.config_manager import ConfigLoader
from pyautoprocess.core.beam_center_detector import write_provenance


def make_processor(**overrides):
    params = ConfigLoader().get_config('default')
    for key, value in overrides.items():
        setattr(params, key, value)
    return CrystallographyProcessor(params)


def synthetic_pattern(size=512, centre=(250.0, 262.0), seed=0):
    """A centrosymmetric diffraction-like frame with a beam stop, centred off the midpoint."""
    rng = np.random.default_rng(seed)
    true_x, true_y = centre
    yy, xx = np.indices((size, size), dtype=np.float32)
    dx, dy = xx - true_x, yy - true_y
    radius = np.hypot(dx, dy)

    frame = 40.0 + rng.normal(0, 3.0, (size, size)).astype(np.float32)
    frame += 900.0 * np.exp(-(radius / 65.0) ** 2)

    a1, a2 = (31.0, 7.0), (-8.0, 29.0)
    for h in range(-7, 8):
        for k in range(-7, 8):
            if h == k == 0:
                continue
            px = true_x + h * a1[0] + k * a2[0]
            py = true_y + h * a1[1] + k * a2[1]
            if 0 <= px < size and 0 <= py < size:
                amplitude = 5200.0 / (1.0 + 0.05 * (h * h + k * k))
                frame += amplitude * np.exp(-(((xx - px) ** 2 + (yy - py) ** 2) / (2 * 2.4 ** 2)))

    frame[radius < 12] = 2.0
    frame[(np.abs(dx) < 3) & (dy < 0)] = 2.0
    return frame


def blank_frame(size=512, seed=1):
    """Near-blank: constant plus faint noise. Does NOT raise in the estimator -- it is caught
    by the confidence gate, which is exactly why that gate exists."""
    return np.full((size, size), 12.0, dtype=np.float32) + \
        np.random.default_rng(seed).normal(0, 0.01, (size, size)).astype(np.float32)


# ------------------------------------------------------------------- frame extraction / flip

def test_ser_frames_are_flipped_to_match_the_converted_tifs():
    """FileHandler flips SER vertically when writing TIFs; detection must see the same thing."""
    processor = make_processor()
    data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)

    ser = processor._probe_frames(data, True, "x_960_0.3_3.ser", [2])
    mrc = processor._probe_frames(data, True, "x_960_0.3_3.mrc", [2])

    np.testing.assert_array_equal(ser[2], data[1][::-1])
    np.testing.assert_array_equal(mrc[2], data[1])
    assert not np.array_equal(ser[2], mrc[2]), "the flip must actually change the frame"


def test_tvips_is_not_flipped():
    processor = make_processor()
    data = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    frames = processor._probe_frames(data, True, "x_960_0.3_3.tvips", [1])
    np.testing.assert_array_equal(frames[1], data[0])


def test_out_of_range_frame_numbers_are_skipped_not_wrapped():
    """Negative indexing would silently return the WRONG frame instead of nothing."""
    processor = make_processor()
    data = np.zeros((3, 4, 4), dtype=np.float32)
    frames = processor._probe_frames(data, True, "x.mrc", [0, 1, 3, 4, 99])
    assert sorted(frames) == [1, 3]


def test_single_frame_movie_yields_only_frame_one():
    processor = make_processor()
    data = np.zeros((4, 4), dtype=np.float32)
    assert sorted(processor._probe_frames(data, False, "x.mrc", [1, 2])) == [1]


# --------------------------------------------------------- the resolution-limit dependency

def test_resolution_limits_depend_on_the_beam_centre():
    """A moved centre must change the limits -- this is why they are recomputed after detection."""
    processor = make_processor()
    baseline = processor.calculate_resolution_ranges("960")
    moved = processor.calculate_resolution_ranges("960", 700, 700)
    assert baseline is not None and moved is not None
    assert baseline != moved


def test_resolution_limits_default_to_the_configured_centre():
    processor = make_processor()
    explicit = processor.calculate_resolution_ranges(
        "960", processor.params.beam_center_x, processor.params.beam_center_y)
    assert processor.calculate_resolution_ranges("960") == explicit


# ------------------------------------------------------------------------------- detection

def test_explicit_coordinates_suppress_detection(tmp_path):
    processor = make_processor(beam_center_x_explicit=True, beam_center_y_explicit=True)
    messages = []
    processor.display.log_print = messages.append
    result = processor._detect_beam_center(
        np.zeros((3, 64, 64), dtype=np.float32), True, "x.mrc", 1, 3, tmp_path)
    assert result is None
    assert any("explicitly" in m for m in messages)


@pytest.mark.parametrize("extension", [".mrc", ".ser"])
def test_detection_finds_the_centre_and_records_provenance(tmp_path, extension):
    """End-to-end: three good frames -> a centre near truth, and a beam_center.LP beside it.

    The SER case is the one that matters: because _probe_frames flips it and the pattern is
    built symmetric about a known point, the flip must be accounted for rather than ignored.
    """
    size, true_x, true_y = 512, 250.0, 262.0
    frames = np.stack([synthetic_pattern(size, (true_x, true_y), seed=s) for s in (0, 1, 2)])

    # Config centre near, but not on, the truth -- the estimator refines from this prior.
    processor = make_processor(beam_center_x=256, beam_center_y=256, frame_size=size)
    result = processor._detect_beam_center(frames, True, f"x_960_0.3_3{extension}", 1, 3, tmp_path)

    assert result is not None, "detection should succeed on a clean synthetic pattern"
    x, y = result
    expected_x = true_x + 1                       # 0-based estimator -> 1-based XDS
    expected_y = (size - 1 - true_y) + 1 if extension == ".ser" else true_y + 1
    assert abs(x - expected_x) <= 3, f"ORGX {x} vs expected ~{expected_x}"
    assert abs(y - expected_y) <= 3, f"ORGY {y} vs expected ~{expected_y}"

    record = (tmp_path / "beam_center.LP").read_text()
    assert "result= DETECTED" in record
    assert f"ORGX= {x}" in record and f"ORGY= {y}" in record
    assert "confidence=" in record and "frames_tried=" in record


def test_all_blank_frames_fall_back_and_still_record_provenance(tmp_path):
    """The fallback must be recorded too: 'we tried and declined' is the useful fact later."""
    frames = np.stack([blank_frame(256, seed=s) for s in (1, 2, 3)])
    processor = make_processor(beam_center_x=130, beam_center_y=130, frame_size=256)
    messages = []
    processor.display.log_print = messages.append

    result = processor._detect_beam_center(frames, True, "x_960_0.3_3.mrc", 1, 3, tmp_path)

    assert result is None
    assert any("keeping configured" in m for m in messages)
    record = (tmp_path / "beam_center.LP").read_text()
    assert "result= FALLBACK" in record
    assert "ORGX= 130" in record and "ORGY= 130" in record


def test_provenance_never_raises_on_an_unwritable_location():
    """Provenance is a nicety; it must not be able to kill a processing run."""
    write_provenance(None, "/definitely/not/a/real/directory", 1030, 1040)

"""
Tests for pyautoprocess.core.beam_center_detector.

Everything here is deterministic: the synthetic patterns are built from seeded
``np.random.default_rng`` instances and the estimator itself is deterministic.
The whole module runs in roughly two seconds -- the real-estimator tests share
module-scoped fixtures so each pattern is analysed once.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyautoprocess.beam_center import BeamCenterEstimationError, BeamCenterResult
from pyautoprocess.core import beam_center_detector as bcd
from pyautoprocess.core.beam_center_detector import (
    DEFAULT_CONFIDENCE_MIN,
    DEFAULT_SINGLE_FRAME_CONFIDENCE_MIN,
    DEFAULT_SPREAD_TOL_PX,
    BeamCenterDetector,
    BeamCenterOutcome,
    select_fallback_frames,
    select_probe_frames,
)


# --------------------------------------------------------------------------
# Synthetic frame builders
# --------------------------------------------------------------------------

SIZE = 512
# Deliberately off-centre so a detector that just returns the frame midpoint,
# or the config prior, fails loudly.
TRUE_X = 250.0  # 0-based column
TRUE_Y = 262.0  # 0-based row
CONFIG_X = SIZE // 2  # 1-based config prior, 6 px away from the truth
CONFIG_Y = SIZE // 2


def make_diffraction_frame(
    center_x: float = TRUE_X,
    center_y: float = TRUE_Y,
    seed: int = 7,
    size: int = SIZE,
) -> np.ndarray:
    """A centrosymmetric diffraction frame with a beam stop.

    Halo ring + direct-beam bloom + a lattice of Bragg spots, then a dark stop
    disc with a support arm running to the top edge (the arm breaks the
    symmetry the estimator relies on, which is exactly what it must cope with).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((size, size)).astype(np.float64)
    dx = xx - center_x
    dy = yy - center_y
    radius = np.hypot(dx, dy)

    image = 300.0 * np.exp(-((radius - size * 0.11) ** 2) / (2 * (size * 0.05) ** 2))
    image += 4000.0 * np.exp(-(radius ** 2) / (2 * (size * 0.02) ** 2))

    spacing = size * 0.055
    sigma = max(1.2, size * 0.004)
    order = int(size * 0.42 / spacing)
    for h in range(-order, order + 1):
        for k in range(-order, order + 1):
            if h == 0 and k == 0:
                continue
            spot_x = center_x + h * spacing + k * spacing * 0.32
            spot_y = center_y + k * spacing * 0.94
            if not (0 <= spot_x < size and 0 <= spot_y < size):
                continue
            amplitude = 9000.0 * np.exp(-math.hypot(h, k) / 4.0)
            image += amplitude * np.exp(
                -(((xx - spot_x) ** 2 + (yy - spot_y) ** 2) / (2 * sigma ** 2))
            )

    image += 60.0 + rng.normal(0.0, 6.0, size=(size, size))
    image = np.maximum(image, 0.0)

    stop = radius <= size * 0.035
    support_arm = (np.abs(dx) <= size * 0.012) & (dy <= 0)
    image[stop | support_arm] = 2.0
    return image.astype(np.float32)


def make_blank_frame(seed: int = 11, size: int = SIZE) -> np.ndarray:
    """Near-blank frame: constant plus tiny noise.

    This does NOT make the estimator raise -- it happily returns a badly wrong
    centre with a very low confidence.  The confidence gate is the only thing
    standing between this and a corrupted ORGX/ORGY.
    """
    rng = np.random.default_rng(seed)
    return (np.full((size, size), 100.0) + rng.normal(0.0, 0.5, (size, size))).astype(
        np.float32
    )


@pytest.fixture(scope="module")
def good_frames() -> dict:
    return {
        1: make_diffraction_frame(seed=1),
        5: make_diffraction_frame(seed=2),
        9: make_diffraction_frame(seed=3),
    }


@pytest.fixture(scope="module")
def blank_frames() -> dict:
    return {1: make_blank_frame(seed=11), 5: make_blank_frame(seed=12)}


def make_detector(**tunables) -> BeamCenterDetector:
    """Detector wired to the synthetic geometry, logging into a list."""
    messages = []
    tunables.setdefault("max_analysis_size", SIZE)
    detector = BeamCenterDetector(
        CONFIG_X, CONFIG_Y, log_print=messages.append, **tunables
    )
    detector.messages = messages  # type: ignore[attr-defined]
    return detector


def stub_result(x: float, y: float, confidence: float, method: str = "inversion"):
    return BeamCenterResult(
        x=x, y=y, method=method, confidence=confidence, candidates=()
    )


# --------------------------------------------------------------------------
# select_probe_frames
# --------------------------------------------------------------------------


def test_probe_frames_normal_range():
    assert select_probe_frames(1, 100) == [1, 50, 100]
    assert select_probe_frames(1, 9) == [1, 5, 9]


def test_probe_frames_not_starting_at_one():
    assert select_probe_frames(10, 20) == [10, 15, 20]
    assert select_probe_frames(7, 8) == [7, 8]


def test_probe_frames_tiny_ranges_collapse():
    assert select_probe_frames(1, 1) == [1]
    assert select_probe_frames(1, 2) == [1, 2]
    assert select_probe_frames(1, 3) == [1, 2, 3]
    assert select_probe_frames(42, 42) == [42]


def test_probe_frames_inverted_range_is_empty():
    assert select_probe_frames(10, 9) == []
    assert select_probe_frames(5, 1) == []


def test_probe_frames_are_inclusive_at_both_ends():
    # Off-by-one guard: first and last of the INCLUSIVE range must appear.
    for start, end in ((1, 2), (1, 5), (3, 4), (100, 250)):
        frames = select_probe_frames(start, end)
        assert frames[0] == start
        assert frames[-1] == end
        assert frames == sorted(set(frames))


# --------------------------------------------------------------------------
# select_fallback_frames
# --------------------------------------------------------------------------


def test_fallback_frames_normal_range():
    # Positions are taken across the SPAN (end - start), not the count:
    # 1..100 has span 99, so 25% -> 1 + 25 = 26 and 75% -> 1 + 74 = 75.
    assert select_fallback_frames(1, 100) == [26, 75]
    assert select_fallback_frames(1, 101) == [26, 76]
    assert select_fallback_frames(101, 200) == [126, 175]


def test_fallback_frames_never_overlap_probe_frames():
    for start, end in ((1, 1), (1, 2), (1, 3), (1, 4), (1, 9), (1, 100), (17, 63)):
        probes = set(select_probe_frames(start, end))
        fallbacks = select_fallback_frames(start, end)
        assert not probes & set(fallbacks)
        assert fallbacks == sorted(set(fallbacks))
        assert all(start <= number <= end for number in fallbacks)


def test_fallback_frames_tiny_ranges():
    assert select_fallback_frames(1, 1) == []
    assert select_fallback_frames(1, 2) == []
    assert select_fallback_frames(1, 3) == []
    assert select_fallback_frames(1, 4) == [3]


def test_fallback_frames_inverted_range_is_empty():
    assert select_fallback_frames(10, 9) == []


# --------------------------------------------------------------------------
# Coordinate convention -- the +1 must not be "fixed" away
# --------------------------------------------------------------------------


def test_zero_based_estimator_output_becomes_one_based_outcome(monkeypatch):
    """A known 0-based estimator answer must come out as exactly 0-based + 1."""
    monkeypatch.setattr(
        bcd, "find_beam_center", lambda image, **kwargs: stub_result(100.4, 200.6, 0.9)
    )
    detector = make_detector()
    outcome = detector.detect({1: np.zeros((64, 64), dtype=np.float32)})

    assert outcome is not None
    assert outcome.raw_x == pytest.approx(101.4)
    assert outcome.raw_y == pytest.approx(201.6)
    assert outcome.x == 101
    assert outcome.y == 202
    assert isinstance(outcome.x, int) and isinstance(outcome.y, int)


def test_config_prior_is_passed_to_estimator_zero_based(monkeypatch):
    seen = {}

    def spy(image, **kwargs):
        seen.update(kwargs)
        return stub_result(10.0, 20.0, 0.9)

    monkeypatch.setattr(bcd, "find_beam_center", spy)
    detector = make_detector()
    detector.detect({1: np.zeros((64, 64), dtype=np.float32)})

    assert seen["initial_center"] == (CONFIG_X - 1, CONFIG_Y - 1)
    assert seen["max_analysis_size"] == SIZE
    assert seen["search_radius_fraction"] == detector.search_radius_fraction


# --------------------------------------------------------------------------
# Combination rules (stubbed estimator -> pure detector logic)
# --------------------------------------------------------------------------


def _stub_by_frame(monkeypatch, results):
    """Map each frame array's first pixel value to a canned estimator answer."""

    def fake(image, **kwargs):
        key = int(np.asarray(image).flat[0])
        outcome = results[key]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(bcd, "find_beam_center", fake)


def _frame(tag: int) -> np.ndarray:
    return np.full((64, 64), float(tag), dtype=np.float32)


def test_three_survivors_use_per_axis_median(monkeypatch):
    _stub_by_frame(
        monkeypatch,
        {
            1: stub_result(100.0, 200.0, 0.9),
            2: stub_result(101.0, 203.0, 0.9),
            3: stub_result(102.0, 201.0, 0.9),
        },
    )
    outcome = make_detector().detect({1: _frame(1), 5: _frame(2), 9: _frame(3)})
    assert outcome is not None
    assert outcome.raw_x == pytest.approx(102.0)  # median 101 -> +1
    assert outcome.raw_y == pytest.approx(202.0)  # median 201 -> +1
    assert outcome.n_frames_used == 3
    assert outcome.frames_tried == (1, 5, 9)
    assert "median" in outcome.method


def test_two_survivors_use_per_axis_mean(monkeypatch):
    _stub_by_frame(
        monkeypatch,
        {1: stub_result(100.0, 200.0, 0.9), 2: stub_result(103.0, 202.0, 0.9)},
    )
    outcome = make_detector().detect({1: _frame(1), 2: _frame(2)})
    assert outcome is not None
    assert outcome.raw_x == pytest.approx(102.5)  # mean 101.5 -> +1
    assert outcome.raw_y == pytest.approx(202.0)  # mean 201.0 -> +1
    assert outcome.n_frames_used == 2
    assert "mean" in outcome.method


def test_low_confidence_frames_are_dropped_before_combining(monkeypatch):
    _stub_by_frame(
        monkeypatch,
        {
            1: stub_result(100.0, 200.0, 0.9),
            2: stub_result(400.0, 400.0, DEFAULT_CONFIDENCE_MIN - 0.01),
            3: stub_result(100.0, 200.0, 0.9),
        },
    )
    outcome = make_detector().detect({1: _frame(1), 2: _frame(2), 3: _frame(3)})
    assert outcome is not None
    assert outcome.n_frames_used == 2
    assert outcome.raw_x == pytest.approx(101.0)
    assert outcome.frames_tried == (1, 2, 3)


def test_single_survivor_below_single_frame_gate_returns_none(monkeypatch):
    confidence = (DEFAULT_CONFIDENCE_MIN + DEFAULT_SINGLE_FRAME_CONFIDENCE_MIN) / 2
    assert DEFAULT_CONFIDENCE_MIN < confidence < DEFAULT_SINGLE_FRAME_CONFIDENCE_MIN
    _stub_by_frame(monkeypatch, {1: stub_result(100.0, 200.0, confidence)})
    detector = make_detector()
    assert detector.detect({1: _frame(1)}) is None
    assert any("single frame" in message for message in detector.messages)


def test_single_survivor_above_single_frame_gate_is_accepted(monkeypatch):
    _stub_by_frame(
        monkeypatch,
        {1: stub_result(100.0, 200.0, DEFAULT_SINGLE_FRAME_CONFIDENCE_MIN + 0.01)},
    )
    outcome = make_detector().detect({1: _frame(1)})
    assert outcome is not None
    assert outcome.n_frames_used == 1
    assert outcome.spread_px == 0.0


def test_disagreeing_survivors_return_none_and_log_positions(monkeypatch):
    offset = DEFAULT_SPREAD_TOL_PX + 3.0
    _stub_by_frame(
        monkeypatch,
        {
            1: stub_result(100.0, 200.0, 0.9),
            2: stub_result(100.0 + offset, 200.0, 0.9),
        },
    )
    detector = make_detector()
    assert detector.detect({1: _frame(1), 2: _frame(2)}) is None
    log = "\n".join(detector.messages)
    assert "disagree" in log
    assert "101.00" in log and f"{101.0 + offset:.2f}" in log


def test_survivors_just_inside_spread_tolerance_are_accepted(monkeypatch):
    offset = DEFAULT_SPREAD_TOL_PX - 0.5
    _stub_by_frame(
        monkeypatch,
        {
            1: stub_result(100.0, 200.0, 0.9),
            2: stub_result(100.0 + offset, 200.0, 0.9),
        },
    )
    outcome = make_detector().detect({1: _frame(1), 2: _frame(2)})
    assert outcome is not None
    assert outcome.spread_px == pytest.approx(offset)


def test_all_frames_raise_returns_none(monkeypatch):
    _stub_by_frame(
        monkeypatch,
        {
            1: ValueError("bad image"),
            2: BeamCenterEstimationError("no estimator converged"),
        },
    )
    detector = make_detector()
    assert detector.detect({1: _frame(1), 2: _frame(2)}) is None
    assert any("confidence gate" in message for message in detector.messages)


def test_one_raising_frame_is_not_fatal(monkeypatch):
    _stub_by_frame(
        monkeypatch,
        {
            1: ValueError("bad image"),
            2: stub_result(100.0, 200.0, 0.9),
            3: stub_result(100.0, 200.0, 0.9),
        },
    )
    outcome = make_detector().detect({1: _frame(1), 2: _frame(2), 3: _frame(3)})
    assert outcome is not None
    assert outcome.n_frames_used == 2
    assert outcome.frames_tried == (1, 2, 3)


def test_empty_input_returns_none():
    detector = make_detector()
    assert detector.detect({}) is None
    assert any("no frames supplied" in message for message in detector.messages)


def test_unknown_tunable_is_rejected():
    with pytest.raises(TypeError):
        BeamCenterDetector(100, 100, log_print=lambda _: None, spread_tolerance=3)


def test_outcome_is_frozen(monkeypatch):
    _stub_by_frame(monkeypatch, {1: stub_result(100.0, 200.0, 0.9)})
    outcome = make_detector().detect({1: _frame(1)})
    assert isinstance(outcome, BeamCenterOutcome)
    with pytest.raises(Exception):
        outcome.x = 5  # type: ignore[misc]


# --------------------------------------------------------------------------
# End-to-end against the real vendored estimator
# --------------------------------------------------------------------------


def test_real_estimator_finds_known_offcentre_origin(good_frames):
    detector = make_detector()
    outcome = detector.detect(good_frames)

    assert outcome is not None, "\n".join(detector.messages)
    # Estimator is 0-based; the outcome is 1-based, hence the +1 on the truth.
    error = math.hypot(outcome.raw_x - (TRUE_X + 1), outcome.raw_y - (TRUE_Y + 1))
    assert error < 1.5, f"error {error:.2f} px, detail: {outcome.detail}"
    assert isinstance(outcome.x, int) and isinstance(outcome.y, int)
    assert outcome.x == round(outcome.raw_x)
    assert outcome.y == round(outcome.raw_y)
    assert outcome.n_frames_used == 3
    assert outcome.confidence >= DEFAULT_CONFIDENCE_MIN
    assert outcome.spread_px <= DEFAULT_SPREAD_TOL_PX
    # And it must not simply have echoed the config prior back at us.
    assert (outcome.x, outcome.y) != (CONFIG_X, CONFIG_Y)


def test_real_blank_frames_are_rejected_not_trusted(blank_frames):
    detector = make_detector()
    assert detector.detect(blank_frames) is None
    log = "\n".join(detector.messages)
    assert "rejected, confidence" in log or "confidence gate" in log


def test_real_mixed_pool_takes_single_survivor_path(blank_frames):
    frames = dict(blank_frames)  # keys 1 and 5
    frames[9] = make_diffraction_frame(seed=4)
    detector = make_detector()
    outcome = detector.detect(frames)

    assert outcome is not None, "\n".join(detector.messages)
    assert outcome.n_frames_used == 1
    assert outcome.spread_px == 0.0
    assert outcome.confidence >= DEFAULT_SINGLE_FRAME_CONFIDENCE_MIN
    error = math.hypot(outcome.raw_x - (TRUE_X + 1), outcome.raw_y - (TRUE_Y + 1))
    assert error < 1.5, f"error {error:.2f} px, detail: {outcome.detail}"


def test_real_estimator_survives_a_vertically_flipped_frame(good_frames):
    """Orientation is the CALLER's contract -- a flip silently moves ORGY.

    This is a regression guard for the module docstring's warning: flipping a
    frame (as SER conversion does) produces a confident but *different* answer,
    so the detector can never be relied on to notice a mis-oriented input.
    """
    flipped = {number: frame[::-1] for number, frame in good_frames.items()}
    detector = make_detector()
    outcome = detector.detect(flipped)

    assert outcome is not None, "\n".join(detector.messages)
    assert outcome.x == pytest.approx(TRUE_X + 1, abs=1.5)
    # y mirrors about the frame midline: (SIZE - 1 - TRUE_Y) 0-based, +1.
    assert outcome.y == pytest.approx(SIZE - TRUE_Y, abs=1.5)
    assert outcome.y != pytest.approx(TRUE_Y + 1, abs=1.5)


def test_out_of_range_tunables_fail_at_construction():
    """A bad tunable must raise, not masquerade as 'every frame failed'."""
    with pytest.raises(ValueError):
        BeamCenterDetector(100, 100, log_print=lambda _: None, search_radius_fraction=0.9)
    with pytest.raises(ValueError):
        BeamCenterDetector(100, 100, log_print=lambda _: None, max_analysis_size=32)

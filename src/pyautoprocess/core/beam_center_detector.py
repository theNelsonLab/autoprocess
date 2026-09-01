"""
Beam centre detection for XDS ORGX/ORGY.

Turns a handful of diffraction frames into ONE integer beam centre suitable for
the XDS ``ORGX``/``ORGY`` keywords, or ``None`` when the estimate cannot be
trusted.  The caller is expected to fall back to the microscope-config values
whenever ``None`` comes back, so ``None`` is always the safe answer: this module
NEVER returns a guess, a midpoint, or the config value dressed up as a
measurement.

This module is deliberately format-agnostic.  It never opens a file and knows
nothing about MRC / SER / TVIPS / TIF.  It is handed in-memory 2-D arrays.

================================================================================
CRITICAL -- FRAME ORIENTATION IS THE CALLER'S CONTRACT
================================================================================
Callers MUST pass frames in the SAME orientation that will eventually be written
to the TIFs that XDS reads.

SER frames are vertically flipped during conversion (see
``core/file_handler.py`` lines 118-119: ``frame_data = frame_data[::-1]`` for
``.ser``).  Handing this detector an UNFLIPPED SER frame therefore produces a
*wrong* ORGY -- mirrored about the detector midline -- while still reporting a
high confidence.  The detector cannot possibly notice this: a flipped
diffraction pattern is still a perfectly valid diffraction pattern.

So: apply the same flip / pedestal / orientation pipeline you use for
conversion, THEN call ``detect``.  X (columns) is unaffected by the SER flip;
only Y (rows) is.
================================================================================

Coordinate conventions
----------------------
* The vendored estimator (``pyautoprocess.beam_center.find_beam_center``) works
  in ZERO-based pixel-centre coordinates: ``x`` along columns, ``y`` along rows,
  in the array's own orientation.
* XDS ``ORGX``/``ORGY`` -- and the ``beam_center_x`` / ``beam_center_y`` config
  values -- are ONE-based.
* So this module subtracts 1 from the config centre on the way in, and adds 1 to
  the estimator output on the way out.  ``BeamCenterOutcome.raw_x`` / ``raw_y``
  are the unrounded 1-based floats; ``x`` / ``y`` are those rounded to int.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..beam_center import (
    BeamCenterEstimationError,
    BeamCenterResult,
    find_beam_center,
)


# --- Tunable defaults --------------------------------------------------------
# max_analysis_size dominates accuracy: the estimator downsamples to this size,
# and the residual error is roughly 0.55 * downsample_factor pixels.  Measured
# on a synthetic 2048^2 pattern: 512 -> 2.19 px @0.18 s, 1024 -> 0.75 px
# @0.87 s, 2048 -> 0.13 px @9.23 s.  1024 is the accuracy/runtime sweet spot.
DEFAULT_MAX_ANALYSIS_SIZE = 1024

# Fraction of the FULL frame's short side that the search box extends in each
# direction from the supplied config centre: 0.05 => +/-102 px on a 2048 frame.
DEFAULT_SEARCH_RADIUS_FRACTION = 0.05

# A near-blank frame (constant + tiny noise) does NOT raise -- it was measured
# returning a centre 35 px off at confidence 0.042, versus 0.928 for a good
# frame.  That is exactly why this gate exists.
DEFAULT_CONFIDENCE_MIN = 0.35

# Max allowed pairwise disagreement between surviving frames, in pixels.
DEFAULT_SPREAD_TOL_PX = 5.0

# When only one frame survives there is no cross-check, so demand more of it.
DEFAULT_SINGLE_FRAME_CONFIDENCE_MIN = 0.60


def _validate_range(start_frame: int, end_frame: int) -> Tuple[int, int]:
    """Coerce an inclusive 1-based frame range to ints, or signal emptiness."""
    start = int(start_frame)
    end = int(end_frame)
    return start, end


def select_probe_frames(start_frame: int, end_frame: int) -> List[int]:
    """FIRST, MIDDLE and LAST frame of an inclusive 1-based range.

    Deduplicated and sorted, so a 1- or 2-frame range collapses gracefully to
    1 or 2 entries.  Returns ``[]`` for an empty/inverted range.
    """
    start, end = _validate_range(start_frame, end_frame)
    if end < start:
        return []
    middle = start + (end - start) // 2
    return sorted({start, middle, end})


def select_fallback_frames(start_frame: int, end_frame: int) -> List[int]:
    """The 25% and 75% positions of an inclusive 1-based range, for the retry pass.

    Anything :func:`select_probe_frames` already returns is excluded, so short
    ranges yield fewer frames -- or none at all -- rather than repeating work.
    """
    start, end = _validate_range(start_frame, end_frame)
    if end < start:
        return []
    span = end - start
    quarter = start + int(round(0.25 * span))
    three_quarter = start + int(round(0.75 * span))
    already_probed = set(select_probe_frames(start, end))
    return sorted({quarter, three_quarter} - already_probed)


@dataclass(frozen=True)
class BeamCenterOutcome:
    """A trusted beam centre, ready to be written to XDS ORGX/ORGY."""

    x: int
    """FINAL 1-based ORGX value (rounded)."""

    y: int
    """FINAL 1-based ORGY value (rounded)."""

    raw_x: float
    """Combined 1-based X before rounding."""

    raw_y: float
    """Combined 1-based Y before rounding."""

    method: str
    """Estimator method name, or how several frames were combined."""

    confidence: float
    """Combined confidence in [0, 1]; the mean over surviving frames."""

    n_frames_used: int
    """How many frames survived the confidence gate."""

    spread_px: float
    """Max pairwise distance between surviving frame estimates (0.0 if one)."""

    frames_tried: Tuple[int, ...]
    """1-based frame numbers actually handed to the estimator."""

    detail: str
    """One-line human-readable provenance."""


class BeamCenterDetector:
    """Estimate one beam centre from several diffraction frames.

    Parameters
    ----------
    config_x, config_y:
        The microscope-config beam centre, 1-based, as it would be written to
        ORGX/ORGY.  Used as the estimator's prior and as the centre of the
        search box -- the answer is constrained to stay near it.
    log_print:
        Progress/diagnostic sink, matching the package convention
        (``FileHandler.log_print``, ``ImageProcessor.log_print``).  Defaults to
        :func:`print`.
    **tunables:
        Optional overrides for ``max_analysis_size``, ``search_radius_fraction``,
        ``confidence_min``, ``spread_tol_px`` and
        ``single_frame_confidence_min``.
    """

    def __init__(
        self,
        config_x: int,
        config_y: int,
        log_print: Optional[Callable[[str], None]] = None,
        **tunables,
    ) -> None:
        self.config_x = int(config_x)
        self.config_y = int(config_y)
        self.log_print = log_print or print

        known = {
            "max_analysis_size": DEFAULT_MAX_ANALYSIS_SIZE,
            "search_radius_fraction": DEFAULT_SEARCH_RADIUS_FRACTION,
            "confidence_min": DEFAULT_CONFIDENCE_MIN,
            "spread_tol_px": DEFAULT_SPREAD_TOL_PX,
            "single_frame_confidence_min": DEFAULT_SINGLE_FRAME_CONFIDENCE_MIN,
        }
        unknown = set(tunables) - set(known)
        if unknown:
            raise TypeError(
                "BeamCenterDetector got unexpected tunable(s): "
                + ", ".join(sorted(unknown))
            )
        self.max_analysis_size = int(
            tunables.get("max_analysis_size", known["max_analysis_size"])
        )
        self.search_radius_fraction = float(
            tunables.get("search_radius_fraction", known["search_radius_fraction"])
        )
        self.confidence_min = float(
            tunables.get("confidence_min", known["confidence_min"])
        )
        self.spread_tol_px = float(
            tunables.get("spread_tol_px", known["spread_tol_px"])
        )
        self.single_frame_confidence_min = float(
            tunables.get(
                "single_frame_confidence_min", known["single_frame_confidence_min"]
            )
        )

        # Validate here rather than letting find_beam_center raise per frame:
        # otherwise a bad tunable would be logged as "every frame failed" and
        # silently degrade to the config centre.
        if not 0.01 <= self.search_radius_fraction <= 0.45:
            raise ValueError("search_radius_fraction must be between 0.01 and 0.45")
        if self.max_analysis_size < 64:
            raise ValueError("max_analysis_size must be at least 64")
        if self.spread_tol_px < 0:
            raise ValueError("spread_tol_px must not be negative")

    # -- internals ------------------------------------------------------------

    def _analyse_frame(self, frame: "np.ndarray") -> Optional[BeamCenterResult]:
        """Run the estimator on one frame; return ``None`` instead of raising."""
        return find_beam_center(
            frame,
            initial_center=(self.config_x - 1, self.config_y - 1),
            search_radius_fraction=self.search_radius_fraction,
            max_analysis_size=self.max_analysis_size,
        )

    @staticmethod
    def _max_pairwise_distance(points: Sequence[Tuple[float, float]]) -> float:
        worst = 0.0
        for index, (x_a, y_a) in enumerate(points):
            for x_b, y_b in points[index + 1 :]:
                worst = max(worst, math.hypot(x_a - x_b, y_a - y_b))
        return worst

    # -- public API -----------------------------------------------------------

    def detect(
        self, frames_by_number: Dict[int, "np.ndarray"]
    ) -> Optional[BeamCenterOutcome]:
        """Estimate the beam centre from ``{1-based frame number: 2-D array}``.

        Returns a :class:`BeamCenterOutcome` with 1-based integer ``x``/``y``,
        or ``None`` if the estimate cannot be trusted.  Every ``None`` path logs
        its reason.  Callers must fall back to the microscope-config values.

        Reminder: frames must already be in the orientation that will be written
        to the TIFs XDS reads (see the module docstring).
        """
        if not frames_by_number:
            self.log_print(
                "Beam center detection skipped: no frames supplied; "
                "using configured beam center"
            )
            return None

        frame_numbers = sorted(frames_by_number)
        self.log_print(
            f"Detecting beam center from {len(frame_numbers)} frame(s): "
            f"{', '.join(str(number) for number in frame_numbers)}"
        )
        self.log_print(
            f"  Prior (config, 1-based): ({self.config_x}, {self.config_y}); "
            f"search radius fraction {self.search_radius_fraction}, "
            f"analysis size {self.max_analysis_size}"
        )

        accepted: List[Tuple[int, BeamCenterResult]] = []
        for number in frame_numbers:
            try:
                result = self._analyse_frame(frames_by_number[number])
            except (ValueError, BeamCenterEstimationError) as error:
                # A failing frame is dropped, never fatal.
                self.log_print(f"  Frame {number}: estimation failed ({error})")
                continue
            except Exception as error:  # pragma: no cover - defensive
                self.log_print(
                    f"  Frame {number}: unexpected estimation error "
                    f"({type(error).__name__}: {error})"
                )
                continue

            if result.confidence < self.confidence_min:
                self.log_print(
                    f"  Frame {number}: rejected, confidence "
                    f"{result.confidence:.3f} < {self.confidence_min:.2f} "
                    f"(method {result.method})"
                )
                continue

            self.log_print(
                f"  Frame {number}: center (1-based) "
                f"({result.x + 1:.2f}, {result.y + 1:.2f}), "
                f"confidence {result.confidence:.3f}, method {result.method}"
            )
            accepted.append((number, result))

        tried = tuple(frame_numbers)

        if not accepted:
            self.log_print(
                "Beam center detection failed: no frame passed the confidence "
                f"gate ({self.confidence_min:.2f}); using configured beam center"
            )
            return None

        points = [(item[1].x, item[1].y) for item in accepted]
        confidences = [item[1].confidence for item in accepted]
        spread = self._max_pairwise_distance(points) if len(points) > 1 else 0.0

        if len(points) > 1 and spread > self.spread_tol_px:
            positions = "; ".join(
                f"frame {number}: ({result.x + 1:.2f}, {result.y + 1:.2f})"
                for number, result in accepted
            )
            self.log_print(
                "Beam center detection failed: frames disagree by "
                f"{spread:.2f} px > {self.spread_tol_px:.2f} px tolerance "
                f"[{positions}]; using configured beam center"
            )
            return None

        if len(points) == 1:
            number, result = accepted[0]
            if result.confidence < self.single_frame_confidence_min:
                self.log_print(
                    f"Beam center detection failed: only frame {number} "
                    f"survived and its confidence {result.confidence:.3f} < "
                    f"{self.single_frame_confidence_min:.2f} required for a "
                    "single frame; using configured beam center"
                )
                return None
            combined_x, combined_y = points[0]
            method = result.method
            combination = "single frame"
        elif len(points) == 2:
            combined_x = float(np.mean([point[0] for point in points]))
            combined_y = float(np.mean([point[1] for point in points]))
            method = f"mean-of-2/{accepted[0][1].method}"
            combination = "mean of 2 frames"
        else:
            combined_x = float(np.median([point[0] for point in points]))
            combined_y = float(np.median([point[1] for point in points]))
            method = f"median-of-{len(points)}/{accepted[0][1].method}"
            combination = f"median of {len(points)} frames"

        # Estimator output is 0-based; ORGX/ORGY are 1-based.
        raw_x = float(combined_x) + 1.0
        raw_y = float(combined_y) + 1.0
        final_x = int(round(raw_x))
        final_y = int(round(raw_y))
        confidence = float(np.mean(confidences))

        used = ", ".join(str(number) for number, _ in accepted)
        detail = (
            f"ORGX/ORGY ({final_x}, {final_y}) from {combination} "
            f"[{used}] of frames tried [{', '.join(str(n) for n in tried)}]; "
            f"raw ({raw_x:.2f}, {raw_y:.2f}), confidence {confidence:.3f}, "
            f"spread {spread:.2f} px, method {method}"
        )
        self.log_print(f"Beam center detected: {detail}")
        self.log_print(
            f"  Offset from config: "
            f"({final_x - self.config_x:+d}, {final_y - self.config_y:+d}) px"
        )

        return BeamCenterOutcome(
            x=final_x,
            y=final_y,
            raw_x=raw_x,
            raw_y=raw_y,
            method=method,
            confidence=confidence,
            n_frames_used=len(accepted),
            spread_px=spread,
            frames_tried=tried,
            detail=detail,
        )

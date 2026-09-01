"""
Rotation-axis resolution for the opt-in ``--auto-rotation-axis`` feature.

The XDS rotation axis depends on which way the goniometer swept. A positive-to-negative
sweep (PtoN) uses the microscope config's axis as configured; a negative-to-positive sweep
(NtoP) uses its negation. The sweep direction is recorded in the filename, so it can be
determined per dataset instead of being fixed for a whole run.

Evidence and limits, from a survey of the lab archive (199,606 movie filenames):
  * The flip is a WHOLE-VECTOR negation, not just the x component. F30 data shows
    `-0.8290 -0.5592 0` <-> `0.8290 0.5592 0`.
  * PtoN dominates real collection (~93% of token-bearing files), so this is a no-op far
    more often than not.
  * The NtoP arm is supported by 4 unique crystals in one detector family, and no NtoP
    `.mrc` dataset is known to exist, so that combination is UNTESTED. This is why the
    feature is opt-in and logs loudly whenever it actually flips.

Everything here is pure and side-effect free: it returns the axis plus an explanation, and
the caller decides what to log. Failure always yields the unmodified base axis -- the
microscope config is the fallback, never a guess.
"""
from typing import Optional, Tuple

from .filename_parser import NTOP, PTON, resolve_tilt_direction


def flip_rotation_axis(axis: str) -> Optional[str]:
    """Negate every component of an XDS rotation-axis vector.

    Returns None if `axis` is not three numeric components, so the caller can fall back
    rather than write a malformed XDS.INP.

    The original text of each component is preserved rather than reformatted, so
    '-0.8290 -0.5592 0' becomes '0.8290 0.5592 0' and not '0.829 0.5592 0.0'. Zero is left
    alone, since '-0' would be technically valid but needlessly confusing in a log.
    """
    components = axis.split()
    if len(components) != 3:
        return None

    flipped = []
    for component in components:
        try:
            value = float(component)
        except ValueError:
            return None
        if value == 0:
            flipped.append(component)
        elif component.startswith("-"):
            flipped.append(component[1:])
        else:
            flipped.append("-" + component)

    return " ".join(flipped)


def resolve_rotation_axis(base_axis: str,
                          filename: str,
                          *,
                          enabled: bool,
                          explicit: bool = False) -> Tuple[str, Optional[str]]:
    """Decide the rotation axis for one dataset.

    Args:
        base_axis: the axis in effect -- an explicit --rotation-axis, else the microscope config.
        filename:  the movie filename, which carries the tilt-direction token.
        enabled:   whether --auto-rotation-axis was given. When False this is a no-op.
        explicit:  whether base_axis came from the command line. Affects wording only; the
                   flag deliberately flips an explicit axis too, since the user is choosing
                   the axis, not the sweep direction.

    Returns:
        (axis, message). `message` is None when there is nothing worth saying (the feature is
        off); otherwise it is a line for the caller to log. `axis` falls back to `base_axis`
        for every failure path.
    """
    if not enabled:
        return base_axis, None

    origin = "command line" if explicit else "microscope config"
    direction, reason = resolve_tilt_direction(filename)

    if direction is None:
        return base_axis, (f"Auto rotation axis: {reason}; "
                           f"keeping {origin} axis '{base_axis}'")

    if direction.direction == PTON:
        return base_axis, (f"Auto rotation axis: {reason}; "
                           f"PtoN needs no flip, keeping {origin} axis '{base_axis}'")

    flipped = flip_rotation_axis(base_axis)
    if flipped is None:
        return base_axis, (f"Auto rotation axis: {reason}, but {origin} axis '{base_axis}' "
                           f"is not three numeric components and cannot be flipped; keeping it")

    return flipped, (f"Auto rotation axis: {reason}; FLIPPING {origin} axis "
                     f"'{base_axis}' -> '{flipped}'. Note: NtoP sweeps are rarer than PtoN "
                     f"and this path is less well tested -- check the indexing result.")


__all__ = ["flip_rotation_axis", "resolve_rotation_axis", "PTON", "NTOP"]

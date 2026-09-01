"""
Shared filename-parsing primitives.

MicroED movie filenames follow:

    sample-name_distance_rotation_exposure[_extra-notes...].ext

Three parsers in this package consume that convention -- ``autoprocess.parse_filename``,
``monitor_ed.validate_movie_filename`` and ``image_process._parse_source_file_metadata``.
They have genuinely different responsibilities and are deliberately NOT merged. What lives
here are the primitives they share, so the grammar itself is defined in exactly one place.

The "extra notes" fields usually record the goniometer tilt sweep and its direction, e.g.
``P50toN-50``, ``n60top10``, ``0toP25``. That direction determines the sign of the XDS
rotation axis, so it is parsed here rather than guessed at the call site.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

MOVIE_EXTENSIONS = {".mrc", ".ser", ".tvips"}

# Underscore fields 0-3 are sample / distance / rotation / exposure. Anything from field 4
# onward is free-form notes, and that is where a tilt token can appear. Real data puts it at
# index 4 most of the time and at 5 often enough to matter, so scan to the end rather than
# probing a fixed position.
FIRST_NOTE_FIELD = 4

# A tilt token is a WHOLE underscore field: <sense><angle>to<sense><angle>, where sense is
# P (positive), N (negative) or 0, and the angle is optional. Case-insensitive. The angle may
# carry a redundant minus (`N-50` means -50, same as `N50`) and may use 'p' as a decimal
# point, matching the convention used by the numeric fields.
_TILT_TOKEN_RE = re.compile(
    r"^(?P<start_sense>[PN0])(?P<start_value>-?\d+(?:[.p]\d+)?)?"
    r"to"
    r"(?P<end_sense>[PN0])(?P<end_value>-?\d+(?:[.p]\d+)?)?$",
    re.IGNORECASE,
)

PTON = "PtoN"
NTOP = "NtoP"


@dataclass(frozen=True)
class TiltDirection:
    """One parsed tilt-sweep token."""
    token: str
    field_index: int
    start_angle: float
    end_angle: float
    direction: str  # PTON (decreasing angle) or NTOP (increasing angle)


def looks_numeric(value: str) -> bool:
    """True if `value` reads as a number, accepting 'p' as the decimal separator.

    Accepts: '960', '0.3', '1p5', '0p6'.   Rejects: 'movie', 'bin4', 'g8sp7'.
    """
    try:
        float(value.replace("p", "."))
        return True
    except (ValueError, AttributeError):
        return False


def _signed_angle(sense: str, value: Optional[str]) -> float:
    """Convert a token half into a signed angle.

    The letter carries the sign; any minus inside the number is redundant, so `N-50` and
    `N50` both mean -50. A missing magnitude means "some angle in that direction", which is
    enough to establish the sweep direction.
    """
    sense = sense.lower()
    if sense == "0":
        return 0.0
    magnitude = 1.0 if value is None else abs(float(value.replace("p", ".")))
    return -magnitude if sense == "n" else magnitude


def find_tilt_tokens(filename: str) -> List[TiltDirection]:
    """Return every parseable tilt token in `filename`, in field order.

    Only fields from FIRST_NOTE_FIELD onward are considered, so a sample name can never be
    mistaken for a tilt token. The extension is stripped from the final field.
    """
    fields = Path(filename).name.split("_")
    found: List[TiltDirection] = []

    for index in range(FIRST_NOTE_FIELD, len(fields)):
        field = fields[index]
        if index == len(fields) - 1:
            suffix = Path(field).suffix.lower()
            if suffix in MOVIE_EXTENSIONS:
                field = field[: -len(suffix)]

        match = _TILT_TOKEN_RE.match(field)
        if not match:
            continue

        start = _signed_angle(match.group("start_sense"), match.group("start_value"))
        end = _signed_angle(match.group("end_sense"), match.group("end_value"))
        if start == end:
            # e.g. 'P0toN0' -- a token, but it establishes no direction.
            continue

        found.append(TiltDirection(
            token=field,
            field_index=index,
            start_angle=start,
            end_angle=end,
            direction=PTON if end < start else NTOP,
        ))

    return found


def resolve_tilt_direction(filename: str) -> Tuple[Optional[TiltDirection], str]:
    """Determine the tilt direction for one filename.

    Returns (result, reason). `result` is None whenever the direction cannot be established
    with confidence -- no token, or several tokens that disagree. `reason` is a short phrase
    for the caller to log, so the decision is always traceable.
    """
    tokens = find_tilt_tokens(filename)

    if not tokens:
        return None, "no tilt-direction token in filename"

    directions = {token.direction for token in tokens}
    if len(directions) > 1:
        detail = ", ".join(f"{t.token}->{t.direction}" for t in tokens)
        return None, f"conflicting tilt-direction tokens ({detail})"

    chosen = tokens[0]
    extra = f" ({len(tokens)} matching tokens)" if len(tokens) > 1 else ""
    return chosen, (f"token '{chosen.token}' at field {chosen.field_index}: "
                    f"{chosen.start_angle:g} -> {chosen.end_angle:g} deg = {chosen.direction}{extra}")

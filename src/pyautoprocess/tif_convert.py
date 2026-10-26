"""
Pixel conversion from camera data to the uint16 TIF frames that XDS reads.

One rule for every format: round float data to the nearest integer (half to even), add the
pedestal, clip to the uint16 range 0..65535, store as uint16. Integer data is never rounded,
so the only information lost is the fractional part of float data.

Values above 65535 are clipped to 65535, the largest value the frame can hold, instead of
wrapping around to a small number. XDS flags a pixel as overloaded when it is ABOVE OVERLOAD;
autoprocess writes OVERLOAD= 65534 (core/xds_manager.py: XDS_OVERLOAD), so every pixel at 65535
-- clipped here, or saturated by the camera -- marks its reflection as overloaded and XDS
rejects it instead of integrating it at the cap. The conversion reports how many pixels were
clipped.
Values that are still negative after the pedestal become 0, which XDS treats as invalid
(MINIMUM_VALID_PIXEL_VALUE= 1). NaN pixels in float data also become 0 (invalid), and are
counted; +inf becomes 65535 (counted as clipped) and -inf becomes 0.
"""
from typing import NamedTuple, Tuple

import numpy as np

UINT16_MAX = 65535

# .ser (TIA) frames get +200; .mrc and .tvips frames get +1
SER_PEDESTAL = 200
DEFAULT_PEDESTAL = 1


def pedestal_for(file_extension: str) -> int:
    """Pedestal added for a source file extension such as '.ser' or '.mrc'."""
    return SER_PEDESTAL if file_extension.lower() == '.ser' else DEFAULT_PEDESTAL


class Conversion(NamedTuple):
    output: np.ndarray   # the uint16 frame
    n_clipped_high: int  # pixels above 65535 after the pedestal (incl. +inf), stored as 65535
    n_nan: int           # NaN pixels (float data only), stored as 0


def convert_frame_counts(frame: np.ndarray, pedestal: int) -> Conversion:
    """Convert one frame and report what had to be clipped or replaced."""
    n_nan = 0
    if frame.dtype.kind == 'f':
        values = np.rint(frame.astype(np.float64)) + pedestal
        nan = np.isnan(values)
        n_nan = int(np.count_nonzero(nan))
        if n_nan:
            values[nan] = 0      # explicit: casting NaN to uint16 is undefined
    else:
        values = frame.astype(np.int64) + pedestal
    n_clipped_high = int(np.count_nonzero(values > UINT16_MAX))
    output = np.clip(values, 0, UINT16_MAX).astype(np.uint16)
    return Conversion(output, n_clipped_high, n_nan)


def convert_frame(frame: np.ndarray, pedestal: int) -> Tuple[np.ndarray, int]:
    """Return (uint16 frame, number of pixels clipped down to 65535)."""
    result = convert_frame_counts(frame, pedestal)
    return result.output, result.n_clipped_high


def range_after_pedestal(data: np.ndarray, pedestal: int) -> Tuple[float, float]:
    """(min, max) of ``data`` plus the pedestal as Python numbers, ignoring NaN.

    Python arithmetic, so a NumPy scalar cannot wrap (uint16 65535 + 1 is 65536, not 0).
    Returns (nan, nan) when every value is NaN.
    """
    if data.dtype.kind == 'f':
        finite_or_inf = data[~np.isnan(data)]
        if finite_or_inf.size == 0:
            return float('nan'), float('nan')
        return float(np.min(finite_or_inf)) + pedestal, float(np.max(finite_or_inf)) + pedestal
    return int(np.min(data)) + pedestal, int(np.max(data)) + pedestal

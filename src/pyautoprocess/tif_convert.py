"""
Pixel conversion from camera data to the uint16 TIF frames that XDS reads.

One rule for every format: round float data to the nearest integer (half to even), add the
pedestal, clip to the uint16 range 0..65535, store as uint16. Integer data is never rounded,
so the only information lost is the fractional part of float data.

Values above 65535 are clipped to 65535, the largest value the frame can hold, instead of
wrapping around to a small number. XDS flags a pixel as overloaded only when it is ABOVE
OVERLOAD, so with OVERLOAD= 65535 in XDS.INP a clipped pixel is integrated at 65535 (a lower
bound on its true value) and not flagged. The conversion reports how many pixels were clipped.
Values that are still negative after the pedestal become 0, which XDS treats as invalid
(MINIMUM_VALID_PIXEL_VALUE= 1).
"""
from typing import Tuple

import numpy as np

UINT16_MAX = 65535

# .ser (TIA) frames get +200; .mrc and .tvips frames get +1
SER_PEDESTAL = 200
DEFAULT_PEDESTAL = 1


def pedestal_for(file_extension: str) -> int:
    """Pedestal added for a source file extension such as '.ser' or '.mrc'."""
    return SER_PEDESTAL if file_extension.lower() == '.ser' else DEFAULT_PEDESTAL


def convert_frame(frame: np.ndarray, pedestal: int) -> Tuple[np.ndarray, int]:
    """Return (uint16 frame, number of pixels clipped down to 65535)."""
    if frame.dtype.kind == 'f':
        values = np.rint(frame.astype(np.float64))
    else:
        values = frame.astype(np.int64)
    values = values + pedestal
    n_clipped_high = int(np.count_nonzero(values > UINT16_MAX))
    output = np.clip(values, 0, UINT16_MAX).astype(np.uint16)
    return output, n_clipped_high

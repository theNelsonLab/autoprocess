"""
Reading MRC movies with the right sign for 8-bit (mode 0) data.

MRC2014 defines mode 0 as signed bytes, but IMOD -- and the DE/SerialEM writers that follow it --
store UNSIGNED counts in mode 0 and mark genuinely signed bytes with a flag in the IMOD header
words. `mrcfile` always returns mode 0 as int8, so a DE Apollo frame holding 0-255 counts came
back with every pixel above 127 as a negative number, which the TIF conversion (pedestal +1,
clip at 0) turned into 0: the brightest pixels of every reflection were erased before XDS.

Rule (same as IMOD): mode 0 is unsigned unless the header carries the IMOD stamp AND bit 0 of the
IMOD flags ("bytes are signed"). Detector counts are never negative. Every other mode is returned
exactly as `mrcfile` reads it.
"""
from pathlib import Path
from typing import Optional, Union

import mrcfile
import numpy as np

# IMOD header stamp and its "bytes are signed" flag: 32-bit words 39 and 40 (1-based),
# i.e. byte offsets 152 and 156.
IMOD_STAMP = 1146047817
IMOD_FLAG_SIGNED_BYTES = 1

_HEADER_BYTES = 224


def mode0_signed_from_header(header_bytes: bytes) -> Optional[bool]:
    """Whether a mode-0 MRC header describes signed bytes; None if not mode 0 (or too short)."""
    if len(header_bytes) < _HEADER_BYTES:
        return None
    # Machine stamp byte 0x11 = big-endian; anything else is read as little-endian
    order = '>' if header_bytes[212] == 0x11 else '<'
    words = np.frombuffer(header_bytes[:_HEADER_BYTES], dtype=f'{order}i4')
    if int(words[3]) != 0:
        return None
    stamp, flags = int(words[38]), int(words[39])
    return stamp == IMOD_STAMP and bool(flags & IMOD_FLAG_SIGNED_BYTES)


def mrc_mode0_signed(mrc_path: Union[str, Path]) -> Optional[bool]:
    """Whether an MRC file's 8-bit pixels are signed, from its header; None if not mode 0."""
    with open(mrc_path, 'rb') as handle:
        return mode0_signed_from_header(handle.read(_HEADER_BYTES))


def read_mrc_data(mrc_path: Union[str, Path]) -> np.ndarray:
    """Read an MRC file's data into memory.

    Mode 0 data come back as uint8 (or int8 when the IMOD signed-bytes flag is set);
    all other modes are exactly what `mrcfile` returns.
    """
    with mrcfile.open(str(mrc_path), mode='r') as mrc:
        data = mrc.data.copy()
    if data.dtype.itemsize == 1 and data.dtype.kind in 'iu':
        signed = mrc_mode0_signed(mrc_path)
        if signed is not None:
            data = data.view(np.int8 if signed else np.uint8)
    return data

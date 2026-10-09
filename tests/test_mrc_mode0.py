"""
MRC mode 0 (8-bit) must be read as UNSIGNED counts unless the IMOD flag says the bytes are signed.

DE Apollo movies are mode 0 without an IMOD stamp and hold counts 0-255. mrcfile returns them as
int8, and the TIF conversion (pedestal +1, clip at 0) used to turn every pixel above 127 into 0 --
e.g. Kunal-20261002 sample-H mov4 frame 1: 14 pixels above 127 (max 255) in the MRC, TIF max 121.
"""
import logging
import struct
from types import SimpleNamespace

import mrcfile
import numpy as np
import pytest
import tifffile

from pyautoprocess.core.file_handler import FileHandler
from pyautoprocess.mrc2tif import MRCConverter
from pyautoprocess.mrc_io import (IMOD_STAMP, mode0_signed_from_header, mrc_mode0_signed,
                                  read_mrc_data)
from pyautoprocess.quality_analyzer import DiffractionQualityAnalyzer

PEDESTAL = 1


def write_mrc(path, data, imod_flags=None):
    """Write an MRC with mrcfile; optionally stamp the IMOD words (flags bit 0 = signed bytes)."""
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(data)
    if imod_flags is not None:
        with open(path, 'r+b') as handle:
            handle.seek(152)
            handle.write(struct.pack('<ii', IMOD_STAMP, imod_flags))
    return path


def apollo_like_movie(n_frames=3, size=32):
    """Unsigned counts 0..255 stored the way the DE/SerialEM writer does: mode 0, no IMOD stamp."""
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 40, size=(n_frames, size, size)).astype(np.uint8)
    counts[0, 5, 5] = 255      # true 255 -> int8 -1 -> old TIF value 0
    counts[0, 6, 6] = 149      # true 149 -> int8 -107 -> old TIF value 0
    counts[1, 7, 7] = 128
    counts[2, 8, 8] = 127
    counts[0, :, 0] = np.arange(size) * 8 % 256   # a ramp across the sign boundary
    return counts


def file_handler(tmp_path):
    return FileHandler(SimpleNamespace(verbose=False), log_print_func=lambda m: None)


def test_header_rule():
    header = bytearray(224)
    struct.pack_into('<i', header, 12, 0)                # mode 0
    assert mode0_signed_from_header(bytes(header)) is False
    struct.pack_into('<ii', header, 152, IMOD_STAMP, 0)  # IMOD stamp, unsigned
    assert mode0_signed_from_header(bytes(header)) is False
    struct.pack_into('<ii', header, 152, IMOD_STAMP, 1)  # IMOD stamp, signed-bytes flag
    assert mode0_signed_from_header(bytes(header)) is True
    struct.pack_into('<ii', header, 152, 0, 1)           # flag without the stamp means nothing
    assert mode0_signed_from_header(bytes(header)) is False
    struct.pack_into('<i', header, 12, 1)                # not mode 0
    assert mode0_signed_from_header(bytes(header)) is None
    assert mode0_signed_from_header(b'short') is None


def test_big_endian_header():
    header = bytearray(224)
    header[212] = 0x11
    struct.pack_into('>i', header, 12, 0)
    struct.pack_into('>ii', header, 152, IMOD_STAMP, 1)
    assert mode0_signed_from_header(bytes(header)) is True


def test_mode0_without_imod_stamp_reads_unsigned(tmp_path):
    counts = apollo_like_movie()
    path = write_mrc(tmp_path / 'm.mrc', counts.view(np.int8))
    with mrcfile.open(str(path)) as mrc:
        assert mrc.header.mode == 0 and mrc.data.dtype == np.int8  # the library's view
    assert mrc_mode0_signed(path) is False
    data = read_mrc_data(path)
    assert data.dtype == np.uint8
    np.testing.assert_array_equal(data, counts)


def test_mode0_imod_unsigned_flag_clear_reads_unsigned(tmp_path):
    counts = apollo_like_movie()
    path = write_mrc(tmp_path / 'm.mrc', counts.view(np.int8), imod_flags=0)
    np.testing.assert_array_equal(read_mrc_data(path), counts)


def test_mode0_imod_signed_stays_signed(tmp_path):
    signed = np.array([[[-128, -1, 0, 1, 127]]], dtype=np.int8)
    path = write_mrc(tmp_path / 's.mrc', signed, imod_flags=1)
    data = read_mrc_data(path)
    assert data.dtype == np.int8
    np.testing.assert_array_equal(data, signed)


@pytest.mark.parametrize('dtype, lo, hi', [(np.int16, -300, 3000), (np.float32, -5.5, 900.25),
                                           (np.uint16, 0, 60000)])
def test_other_modes_unchanged(tmp_path, dtype, lo, hi):
    data = np.linspace(lo, hi, 2 * 16 * 16).reshape(2, 16, 16).astype(dtype)
    path = write_mrc(tmp_path / 'o.mrc', data, imod_flags=1)
    with mrcfile.open(str(path)) as mrc:
        expected = mrc.data.copy()
    got = read_mrc_data(path)
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got, expected)


def test_file_handler_tifs_keep_counts_above_127(tmp_path):
    """The autoprocess conversion path: TIF = counts + 1 for every pixel, max 256."""
    counts = apollo_like_movie()
    path = write_mrc(tmp_path / 'mov_560_4_0.25_P50toN-50_g8sp8_c.mrc', counts.view(np.int8))
    handler = file_handler(tmp_path)
    data, multiframe = handler.read_source_file_from_path(path)
    assert multiframe and data.dtype == np.uint8
    images = tmp_path / 'sample' / 'images'
    images.mkdir(parents=True)
    assert handler.convert_data_to_tif(data, multiframe, str(tmp_path / 'sample'), path.name,
                                       images_dir=images)
    tifs = [tifffile.imread(str(images / f'sample_{i:03d}.tif')) for i in (1, 2, 3)]
    for tif, frame in zip(tifs, counts):
        assert tif.dtype == np.uint16
        np.testing.assert_array_equal(tif, frame.astype(np.uint16) + PEDESTAL)
    assert tifs[0].max() == 256 and tifs[0][5, 5] == 256 and tifs[0][6, 6] == 150
    assert tifs[1][7, 7] == 129 and tifs[2][8, 8] == 128


def test_file_handler_mode1_tifs_unchanged(tmp_path):
    """CETA-style mode 1 movies convert exactly as before (int16 + 1, negatives clipped)."""
    raw = np.array([[[-5, -1, 0, 1, 2000, 32767]]], dtype=np.int16).repeat(2, axis=0)
    path = write_mrc(tmp_path / 'c.mrc', raw)
    handler = file_handler(tmp_path)
    data, multiframe = handler.read_source_file_from_path(path)
    assert data.dtype == np.int16
    images = tmp_path / 'c' / 'images'
    images.mkdir(parents=True)
    assert handler.convert_data_to_tif(data, multiframe, str(tmp_path / 'c'), path.name,
                                       images_dir=images)
    tif = tifffile.imread(str(images / 'c_001.tif'))
    np.testing.assert_array_equal(tif, [[0, 0, 1, 2, 2001, 32768]])


def test_mrc2tif_standalone_keeps_counts(tmp_path, monkeypatch):
    counts = apollo_like_movie()
    path = write_mrc(tmp_path / 'm.mrc', counts.view(np.int8))
    logger = logging.getLogger('mrc_mode0_test')
    assert MRCConverter.convert_mrc_to_tif(str(path), PEDESTAL, None, False, logger, logger)
    tif = tifffile.imread(str(tmp_path / 'images' / 'm_001.tif'))
    np.testing.assert_array_equal(tif, counts[0].astype(np.uint16) + PEDESTAL)
    assert tif.max() == 256


def test_mrc2tif_check_data_range_does_not_wrap(tmp_path):
    logger = logging.getLogger('mrc_mode0_test')
    assert MRCConverter.check_data_range(np.array([0, 255], np.uint8), PEDESTAL, logger)
    with pytest.warns(UserWarning):
        assert not MRCConverter.check_data_range(np.array([-5, 3], np.int8), PEDESTAL, logger)


def test_quality_analyzer_reads_unsigned(tmp_path):
    counts = apollo_like_movie()
    path = write_mrc(tmp_path / 'm.mrc', counts.view(np.int8))
    data, multiframe = DiffractionQualityAnalyzer().parse_mrc(str(path))
    assert multiframe
    assert data.dtype == np.float64
    np.testing.assert_array_equal(data, counts.astype(np.float64))


def test_sample_h_mov4_regression(tmp_path):
    """Synthetic stand-in for Kunal-20261002 sample-H mov4 frame 1: 14 pixels above 127, max 255.

    The old conversion produced a TIF whose maximum was 121 with all 14 of those pixels at 0.
    """
    rng = np.random.default_rng(4)
    frame = rng.integers(0, 121, size=(64, 64)).astype(np.uint8)
    hot = rng.choice(64 * 64, size=14, replace=False)
    frame.flat[hot] = np.linspace(128, 255, 14).astype(np.uint8)
    path = write_mrc(tmp_path / 'sample-H-mov4_560_4_0.25_P50toN-50_g8sp8_c.mrc',
                     np.stack([frame, frame]).view(np.int8))
    handler = file_handler(tmp_path)
    data, multiframe = handler.read_source_file_from_path(path)
    images = tmp_path / 'sample-H-mov4' / 'images'
    images.mkdir(parents=True)
    assert handler.convert_data_to_tif(data, multiframe, str(tmp_path / 'sample-H-mov4'),
                                       path.name, images_dir=images)
    tif = tifffile.imread(str(images / 'sample-H-mov4_001.tif'))
    assert tif.max() == 256
    assert int((tif > 128).sum()) == 14
    assert not (tif.flat[hot] == 0).any()

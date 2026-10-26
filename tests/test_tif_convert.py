"""
Pixel conversion to uint16 TIF: round floats, add the pedestal (.ser 200, .mrc/.tvips 1),
clip to 0..65535.

Up to 0.5.4 the cast to uint16 had no upper clip, so a value above 65535 wrapped around
(65536 -> 0, .ser raw 65535 -> 199), and float data was truncated toward zero instead of
rounded. Every value that did not overflow and was not a fractional float converts exactly
as before; the exhaustive tests below pin that.
"""
import logging
import struct
from types import SimpleNamespace

import mrcfile
import numpy as np
import pytest
import tifffile

from pyautoprocess.core.file_handler import FileHandler
from pyautoprocess.tif_convert import convert_frame, pedestal_for

from test_tvips_geometry import write_tvips


def old_054(frame, file_extension):
    """The 0.5.4 expressions, kept here to prove nothing else changed."""
    if file_extension == '.ser':
        return np.maximum(frame + 200, 0).astype(np.uint16)
    return np.maximum(frame.astype(np.int32) + 1, 0).astype(np.uint16)


def test_pedestals():
    assert pedestal_for('.ser') == 200
    assert pedestal_for('.SER') == 200
    assert pedestal_for('.mrc') == 1
    assert pedestal_for('.tvips') == 1


@pytest.mark.parametrize('frame, pedestal, expected, n_high', [
    (np.array([-201, -200, -199, -1, 0, 65335, 65336, 65535, 70000, 2**31 - 1], np.int32), 200,
     [0, 0, 1, 199, 200, 65535, 65535, 65535, 65535, 65535], 4),
    (np.array([0, 1, 65534, 65535], np.uint16), 1, [1, 2, 65535, 65535], 1),
    (np.array([-32768, -2, -1, 0, 32767], np.int16), 1, [0, 0, 0, 1, 32768], 0),
    (np.array([0, 127, 128, 255], np.uint8), 1, [1, 128, 129, 256], 0),
    (np.array([-128, -1, 0, 127], np.int8), 1, [0, 0, 1, 128], 0),
    # floats round half to even, then the pedestal
    (np.array([0.4, 0.5, 0.6, 1.5, 2.5, 1.7, -0.5, -0.6, -1.4, -1.6, 12345.5], np.float32), 1,
     [1, 1, 2, 3, 3, 3, 1, 0, 0, 0, 12347], 0),
    (np.array([65533.6, 65534.4, 65534.5, 65534.6, 70000.0, 3e9], np.float32), 1,
     [65535, 65535, 65535, 65535, 65535, 65535], 3),
    (np.array([1.5, 2.5, 65504.0], np.float16), 1, [3, 3, 65505], 0),
])
def test_values(frame, pedestal, expected, n_high):
    out, clipped = convert_frame(frame, pedestal)
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, expected)
    assert clipped == n_high


@pytest.mark.parametrize('dtype', [np.uint8, np.int8, np.int16, np.uint16])
@pytest.mark.parametrize('ext', ['.mrc', '.tvips', '.ser'])
def test_every_integer_value_matches_054_unless_it_overflowed(dtype, ext):
    info = np.iinfo(dtype)
    frame = np.arange(info.min, info.max + 1, dtype=np.int64).astype(dtype)
    if ext == '.ser':  # seremi returns int32 for .ser
        frame = frame.astype(np.int32)
    new, _ = convert_frame(frame, pedestal_for(ext))
    old = old_054(frame, ext)
    fits = frame.astype(np.int64) + pedestal_for(ext) <= 65535
    np.testing.assert_array_equal(new[fits], old[fits])
    np.testing.assert_array_equal(new[~fits], 65535)


def test_int32_ser_range_matches_054_unless_it_overflowed():
    frame = np.arange(-300_000, 300_000, dtype=np.int32)
    new, clipped = convert_frame(frame, 200)
    fits = frame <= 65335
    np.testing.assert_array_equal(new[fits], old_054(frame, '.ser')[fits])
    np.testing.assert_array_equal(new[~fits], 65535)
    assert clipped == np.count_nonzero(~fits)


def test_integral_floats_match_054():
    """Float data holding whole numbers (counts) converts exactly as before."""
    frame = np.arange(-1000, 65535, dtype=np.float32)
    new, clipped = convert_frame(frame, 1)
    np.testing.assert_array_equal(new, old_054(frame, '.mrc'))
    assert clipped == 0


def test_deterministic():
    rng = np.random.default_rng(3)
    frame = rng.normal(500, 3000, size=(64, 64)).astype(np.float32)
    a, _ = convert_frame(frame, 1)
    b, _ = convert_frame(frame.copy(), 1)
    assert a.tobytes() == b.tobytes()


# --- end to end through FileHandler, as autoprocess runs it ---------------------------------

def run_file_handler(tmp_path, source, name):
    messages = []
    handler = FileHandler(SimpleNamespace(verbose=False), log_print_func=messages.append)
    data, multiframe = handler.read_source_file_from_path(source)
    images = tmp_path / name / 'images'
    images.mkdir(parents=True)
    ok = handler.convert_data_to_tif(data, multiframe, str(tmp_path / name), source.name,
                                     images_dir=images)
    tifs = [tifffile.imread(str(p)) for p in sorted(images.glob('*.tif'))]
    return ok, tifs, messages


def write_mrc(path, data):
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(data)
    return path


def write_ser(path, frames):
    """A minimal TIA .ser series of int32 frames (data type 6), as seremi reads it."""
    n = len(frames)
    h, w = frames[0].shape
    header = struct.pack('<HHHIIII', 0x4949, 0x0197, 0x0210, 0x4122, 0x4152, n, n)
    offset_array = len(header) + 4
    body, data_offsets, pos = b'', [], offset_array + 8 * n
    for f in frames:
        element = b'\0' * 40 + struct.pack('<HII', 6, w, h) + f.astype('<i4').tobytes()
        data_offsets.append(pos)
        body += element
        pos += len(element)
    tags, tag_offsets = b'', []
    for _ in range(n):
        tag_offsets.append(pos)
        tag = struct.pack('<HHI', 0x4152, 0, 0)
        tags += tag
        pos += len(tag)
    path.write_bytes(header + struct.pack('<I', offset_array)
                     + np.array(data_offsets, '<u4').tobytes()
                     + np.array(tag_offsets, '<u4').tobytes() + body + tags)
    return path


def test_mrc_mode6_saturation_clips_and_warns(tmp_path):
    raw = np.array([[[0, 100, 65534, 65535]], [[1, 2, 3, 4]]], dtype=np.uint16)
    ok, tifs, messages = run_file_handler(tmp_path, write_mrc(tmp_path / 'm.mrc', raw), 'm')
    assert ok
    np.testing.assert_array_equal(tifs[0], [[1, 101, 65535, 65535]])
    np.testing.assert_array_equal(tifs[1], [[2, 3, 4, 5]])
    warnings = [m for m in messages if m.startswith('WARNING') and 'clipped to 65535' in m]
    assert len(warnings) == 1 and '1 pixel(s) in 1 frame(s)' in warnings[0]


def test_mrc_float_is_rounded(tmp_path):
    raw = np.array([[[0.4, 0.6, 1.5, 2.5, -0.6, 70000.0]]] * 2, dtype=np.float32)
    ok, tifs, messages = run_file_handler(tmp_path, write_mrc(tmp_path / 'f.mrc', raw), 'f')
    assert ok
    np.testing.assert_array_equal(tifs[0], [[1, 2, 3, 3, 0, 65535]])
    assert any('2 pixel(s) in 2 frame(s)' in m for m in messages)


def test_mrc_without_overflow_has_no_warning(tmp_path):
    raw = np.array([[[0, 255, 7]]] * 2, dtype=np.uint8)
    ok, tifs, messages = run_file_handler(tmp_path, write_mrc(tmp_path / 'a.mrc', raw), 'a')
    assert ok
    np.testing.assert_array_equal(tifs[0], [[1, 256, 8]])
    assert not any('clipped' in m for m in messages)


def test_ser_saturation_clips_and_keeps_flip(tmp_path):
    f1 = np.array([[-300, 0, 65335], [65336, 65535, 70000]], dtype=np.int32)
    f2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    ok, tifs, messages = run_file_handler(tmp_path, write_ser(tmp_path / 's_1.ser', [f1, f2]), 's')
    assert ok
    # rows flipped as before; 65336 + 200 and above clip to 65535 instead of wrapping
    np.testing.assert_array_equal(tifs[0], [[65535, 65535, 65535], [0, 200, 65535]])
    np.testing.assert_array_equal(tifs[1], [[204, 205, 206], [201, 202, 203]])
    assert any('3 pixel(s) in 1 frame(s)' in m for m in messages)


def test_tvips_saturation_clips(tmp_path):
    frames = [np.array([[0, 65534, 65535]], dtype=np.uint16), np.array([[5, 6, 7]], dtype=np.uint16)]
    source = tmp_path / 'scope-mov1_1320_0.3_3_p50ton50.tvips'
    write_tvips(source, frames)
    ok, tifs, messages = run_file_handler(tmp_path, source, 't')
    assert ok
    np.testing.assert_array_equal(tifs[0], [[1, 65535, 65535]])
    np.testing.assert_array_equal(tifs[1], [[6, 7, 8]])
    assert any('1 pixel(s) in 1 frame(s)' in m for m in messages)


def test_mrc2tif_cli_uses_the_same_rule(tmp_path):
    from pyautoprocess.mrc2tif import MRCConverter
    raw = np.array([[[0.6, 1.5, 70000.0]]] * 2, dtype=np.float32)
    path = write_mrc(tmp_path / 'c.mrc', raw)
    logger = logging.getLogger('tif_convert_test')
    with pytest.warns(UserWarning):  # its range check still warns about the overflow
        MRCConverter.convert_mrc_to_tif(str(path), 1, None, False, logger, logger)
    tif = tifffile.imread(str(tmp_path / 'images' / 'c_001.tif'))
    np.testing.assert_array_equal(tif, [[2, 3, 65535]])


# --- NaN / inf in float data ----------------------------------------------------------------

def test_nan_becomes_zero_and_is_counted():
    import warnings
    from pyautoprocess.tif_convert import convert_frame_counts
    frame = np.array([np.nan, np.inf, -np.inf, 1e30, -1e30, 0.4, np.nan], np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # no 'invalid value encountered in cast'
        out, n_high, n_nan = convert_frame_counts(frame, 1)
    np.testing.assert_array_equal(out, [0, 65535, 0, 65535, 0, 1, 0])
    assert (n_high, n_nan) == (2, 2)
    assert convert_frame(frame, 1)[1] == 2      # the 2-tuple API is unchanged


def test_integer_frames_never_report_nan():
    from pyautoprocess.tif_convert import convert_frame_counts
    assert convert_frame_counts(np.array([0, 65535], np.uint16), 1).n_nan == 0


def test_mrc_float_with_nan_warns(tmp_path):
    raw = np.array([[[np.nan, 5.0, 70000.0]], [[np.nan, np.nan, 1.0]]], dtype=np.float32)
    ok, tifs, messages = run_file_handler(tmp_path, write_mrc(tmp_path / 'n.mrc', raw), 'n')
    assert ok
    np.testing.assert_array_equal(tifs[0], [[0, 6, 65535]])
    np.testing.assert_array_equal(tifs[1], [[0, 0, 2]])
    nan_lines = [m for m in messages if 'NaN' in m]
    assert nan_lines == [
        "WARNING: 3 NaN pixel(s) in 2 frame(s) of n.mrc were set to 0 (invalid for XDS)"]
    assert any('1 pixel(s) in 1 frame(s)' in m and 'clipped to 65535' in m for m in messages)


# --- the standalone tools' range warning (NumPy 2 scalar arithmetic) ------------------------

@pytest.mark.parametrize("module, cls, pedestal", [
    ("tvips2tif", "TVIPSConverter", 1),
    ("ser2tif", "SERConverter", 200),
    ("mrc2tif", "MRCConverter", 1),
])
def test_range_warning_fires_at_the_uint16_ceiling(module, cls, pedestal):
    import importlib
    import warnings
    converter = getattr(importlib.import_module(f"pyautoprocess.{module}"), cls)
    logger = logging.getLogger("test-range")
    data = np.array([0, 65535], np.uint16)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert converter.check_data_range(data, pedestal, logger) is False
    messages = [str(w.message) for w in caught]
    assert any("Data range (" in m and f"{65535 + pedestal})" in m for m in messages)
    assert not any("overflow" in m for m in messages)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert converter.check_data_range(np.array([0, 65535 - pedestal], np.uint16),
                                          pedestal, logger) is True


def test_range_after_pedestal_is_nan_safe():
    from pyautoprocess.tif_convert import range_after_pedestal
    assert range_after_pedestal(np.array([65535], np.uint16), 1) == (65536, 65536)
    assert range_after_pedestal(np.array([np.nan, 2.0, np.inf], np.float32), 1) == (3.0, float('inf'))
    lo, hi = range_after_pedestal(np.array([np.nan], np.float32), 1)
    assert np.isnan(lo) and np.isnan(hi)


# --- XDS OVERLOAD ---------------------------------------------------------------------------

def _all_config_names():
    from pyautoprocess.config.config_manager import list_microscope_configs
    return list_microscope_configs()


@pytest.mark.parametrize("config_name", _all_config_names())
def test_xds_inp_overload_is_65534_for_every_config(config_name):
    """Pixels are clipped to 65535; OVERLOAD must be below that so XDS rejects them."""
    from pyautoprocess.config.config_manager import ConfigLoader
    from pyautoprocess.core.xds_manager import XDS_OVERLOAD, XDSManager
    params = ConfigLoader().get_config(config_name)
    text = XDSManager(params).create_xds_input('../images/sample', dict(
        distance='960', rotation='0.3', exposure='3', resolution_range=0.8,
        test_resolution_range=1.0, image_number='100',
        background_pixel=4, signal_pixel=7, min_pixel=7))
    detector = [line for line in text.splitlines() if 'OVERLOAD=' in line]
    assert detector == ['DETECTOR= ADSC MINIMUM_VALID_PIXEL_VALUE= 1 OVERLOAD= 65534']
    assert XDS_OVERLOAD == 65534 < 65535

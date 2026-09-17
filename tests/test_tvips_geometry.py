"""
TVIPS orientation and the two F30 configurations.

Raw .tvips frames are converted to TIF exactly as recorded: no flip, no rotation.
The lab's historical .img files are the same frames mirrored left-right (tvips2smv; the
archived ones are also shifted by one column, pixel = raw[2046 - c] + 1). XDS reads TIF and
SMV rows the same way, so the two inputs differ by a pure left-right mirror of the detector.

A rotation axis is an axial vector, so under the mirror x -> -x it becomes (vx, -vy, -vz),
not (-vx, vy, vz): a reflection also reverses the sense of rotation. The polar-vector version
("0.8290 -0.5592 0") still indexes, but as a wrong-handed near-solution with distorted cells
and ISa near 1 before refinement; see the 2026-09-17 HPC notes.

  F30-TVIPS-SM      raw .tvips -> TIF (autoprocess)          axis "-0.8290 0.5592 0",  ORGX 1028
  F30-TVIPS-SMV-SM  historical .img   (image_process --smv)  axis "-0.8290 -0.5592 0", ORGX 1020

Verified on HPC: from .tvips with F30-TVIPS-SM, schwartz-mov57 and Nidppf-mov5 give the same
cells as their SMV runs (7.16 9.33 13.60; 10.01 14.64 19.77) with matching ISa and Rmeas.
"""
import struct

import numpy as np
import pytest
import tifffile

from pyautoprocess.config.config_manager import ConfigLoader
from pyautoprocess.core.file_handler import FileHandler

CONFIGS = ConfigLoader().configs


def axis(name):
    return [float(v) for v in CONFIGS[name]["rotation_axis"].split()]


def test_the_two_f30_configs_are_left_right_mirrors():
    raw, smv = CONFIGS["F30-TVIPS-SM"], CONFIGS["F30-TVIPS-SMV-SM"]
    ax_raw, ax_smv = axis("F30-TVIPS-SM"), axis("F30-TVIPS-SMV-SM")
    # axial vector under the mirror x -> -x
    assert ax_raw == [ax_smv[0], -ax_smv[1], -ax_smv[2]]
    # archived SMV column c holds raw column 2046 - c, so 1-based X_raw = frame_size - X_smv
    assert raw["beam_center_x"] == raw["frame_size"] - smv["beam_center_x"]
    assert raw["beam_center_y"] == smv["beam_center_y"]
    for key in set(raw) - {"microscope_config", "rotation_axis", "beam_center_x"}:
        assert raw[key] == smv[key], key


def test_f30_values_are_the_verified_ones():
    assert CONFIGS["F30-TVIPS-SM"]["rotation_axis"] == "-0.8290 0.5592 0"
    assert CONFIGS["F30-TVIPS-SM"]["beam_center_x"] == 1028
    assert CONFIGS["F30-TVIPS-SMV-SM"]["rotation_axis"] == "-0.8290 -0.5592 0"
    assert CONFIGS["F30-TVIPS-SMV-SM"]["beam_center_x"] == 1020


def write_tvips(path, frames):
    """A minimal version-2 TVIPS series: 256-byte header, then per frame a header + uint16 data."""
    height, width = frames[0].shape
    img_header_size = 180
    header = bytearray(256)
    struct.pack_into("<5i", header, 0, 256, 2, width, height, 16)
    struct.pack_into("<i", header, 48, img_header_size)
    with open(path, "wb") as f:
        f.write(header)
        for i, frame in enumerate(frames, start=1):
            frame_header = bytearray(img_header_size)
            struct.pack_into("<3i", frame_header, 0, i, 1551832985 + 3 * i, 0)
            f.write(frame_header)
            f.write(frame.astype("<u2").tobytes())


def test_tvips_frames_are_converted_without_flip_or_rotation(tmp_path):
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 5000, size=(6, 8), dtype=np.uint16) for _ in range(3)]
    source = tmp_path / "scope-mov1_1320_0.3_3_p50ton50.tvips"
    write_tvips(source, frames)

    params = ConfigLoader().get_config("F30-TVIPS-SM")
    handler = FileHandler(params, lambda *_: None)
    data, is_multiframe = handler.read_source_file_from_path(source)
    assert is_multiframe and data.shape == (3, 6, 8)

    images = tmp_path / "images"
    images.mkdir()
    assert handler.convert_data_to_tif(data, is_multiframe, "scope-mov1", source.name,
                                       images_dir=images)
    for i, frame in enumerate(frames, start=1):
        written = tifffile.imread(images / f"scope-mov1_{i:03d}.tif")
        # the only change is the +1 pedestal shared with MRC
        np.testing.assert_array_equal(written, frame.astype(np.int32) + 1)

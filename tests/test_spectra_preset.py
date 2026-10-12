"""
Spectra-CETA-mrc-SM and camera_length_map.

On the Spectra the REyes SerialEM scripts name tilt movies with the nominal camera length
(`..._680_0.5_1_...`), while XDS needs the calibrated detector distance (1224 mm). The
filename value used to win over the config, so every Spectra movie was processed at 680 mm.
camera_length_map translates a filename value; a --detector-distance override is left alone.
"""
import sys

import pytest

from pyautoprocess.autoprocess import CrystallographyProcessor
from pyautoprocess.config.config_manager import ConfigLoader, validate_config_entry
from pyautoprocess.ui.cli_parser import parse_arguments

SPECTRA = 'Spectra-CETA-mrc-SM'
MOVIE = 'ss-jacobsen-mov6_680_0.5_1_P50toN-50_g8sp10_rt.mrc'


def make_processor(config=SPECTRA, **overrides):
    params = ConfigLoader().get_config(config)
    for key, value in overrides.items():
        setattr(params, key, value)
    processor = CrystallographyProcessor(params)
    processor.messages = []
    processor.display.log_print = processor.messages.append
    return processor


def test_spectra_preset_geometry():
    params = ConfigLoader().get_config(SPECTRA)
    assert params.wavelength == '0.0197'
    assert params.default_detector_distance == '1224'
    assert params.camera_length_map == {'680': 1224}
    assert params.file_extension == '.mrc'
    assert (params.beam_center_x, params.beam_center_y) == (1030, 1010)


def test_nominal_filename_length_becomes_calibrated_distance():
    processor = make_processor()
    assert processor.parse_filename(MOVIE) == ('ss-jacobsen-mov6', '1224', '0.5', '1')
    assert any('nominal' in m and '1224' in m for m in processor.messages)


@pytest.mark.parametrize('nominal', ['680.0', '680p0'])
def test_map_keys_compare_as_numbers(nominal):
    name = MOVIE.replace('_680_', f'_{nominal}_')
    assert make_processor().parse_filename(name)[1] == '1224'


def test_cli_distance_is_used_as_given():
    processor = make_processor(detector_distance='1210')
    assert processor.parse_filename(MOVIE)[1] == '1210'
    assert not any('nominal' in m for m in processor.messages)


def test_length_missing_from_map_is_kept_with_a_warning():
    processor = make_processor()
    assert processor.parse_filename(MOVIE.replace('_680_', '_530_'))[1] == '530'
    assert any(m.startswith('WARNING') and '530' in m for m in processor.messages)


def test_config_without_map_is_unchanged():
    processor = make_processor('Arctica-CETA-mrc-SM')
    name = 'ss-jacobsen-mov14_960_0.3_3_P50toN-50_g8sp10_c.mrc'
    assert processor.parse_filename(name)[1] == '960'
    assert processor.messages == []


@pytest.mark.parametrize('bad', [{}, [680, 1224], {'680': '1224'}, {'six-eighty': 1224}])
def test_malformed_map_is_rejected(bad):
    entry = dict(ConfigLoader().configs[SPECTRA], camera_length_map=bad)
    problems = validate_config_entry('X', entry)
    assert len(problems) == 1 and 'camera_length_map' in problems[0]


def test_map_survives_cli_parsing(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['autoprocess', '--microscope-config', SPECTRA])
    params = parse_arguments('autoprocess')
    assert params.camera_length_map == {'680': 1224}
    assert params.wavelength == '0.0197'

"""
Microscope configs must load reliably, and every failure to load them must stop the run.

History: the packaged configs were read with the deprecated open_text inside
`except Exception`, so wherever DeprecationWarning is an error the run silently fell back to
a hard-coded generic config. An unknown microscope name did the same. And --config-file was
accepted but never read, so a user's own configuration was silently ignored.
"""
import json
import logging
import os
import sys
import warnings

import pytest

from pyautoprocess.config import config_manager
from pyautoprocess.config.config_manager import ConfigLoader
from pyautoprocess.monitor_ed import _absolutize_config_file
from pyautoprocess.ui.cli_parser import parse_arguments

PACKAGED = ConfigLoader().configs


def scope(**changes):
    """A complete, valid microscope entry based on the packaged Arctica-CETA-mrc-SM."""
    entry = dict(PACKAGED['Arctica-CETA-mrc-SM'])
    entry.pop('microscope_config')
    entry.update(changes)
    return entry


def write_json(path, data):
    path.write_text(json.dumps(data))
    return path


# ------------------------------------------------------------------ packaged configs

def test_packaged_configs_load():
    assert {'default', 'F30-TVIPS-SM', 'Arctica-CETA-mrc-SM'} <= set(PACKAGED)


def test_packaged_configs_load_with_deprecation_warnings_as_errors():
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        assert 'F30-TVIPS-SM' in ConfigLoader().configs


def test_unreadable_package_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(config_manager.importlib.resources, 'files', lambda _pkg: tmp_path / 'absent')
    with pytest.raises(RuntimeError, match='Reinstall pyautoprocess'):
        ConfigLoader()


def test_corrupt_package_json_raises(monkeypatch, tmp_path):
    (tmp_path / 'microscope_configs.json').write_text('{not json')
    monkeypatch.setattr(config_manager.importlib.resources, 'files', lambda _pkg: tmp_path)
    with pytest.raises(RuntimeError, match='JSONDecodeError'):
        ConfigLoader()


def test_stray_local_config_file_is_not_picked_up(monkeypatch, tmp_path):
    """Without --config-file, a microscope_configs.json in the working directory is ignored."""
    write_json(tmp_path / 'microscope_configs.json', {'Stray': scope()})
    monkeypatch.chdir(tmp_path)
    assert 'Stray' not in ConfigLoader().configs


# ------------------------------------------------------------------ microscope names

def test_unknown_microscope_name_raises_and_lists_available():
    loader = ConfigLoader()
    with pytest.raises(ValueError, match="Unknown microscope configuration 'No-Such-Scope'") as err:
        loader.get_config('No-Such-Scope')
    for name in loader.get_available_configs():
        assert name in str(err.value)


@pytest.mark.parametrize('name', ['default', 'F30-TVIPS-SM', 'Arctica-CETA-mrc-SM'])
def test_known_microscope_name_loads(name):
    assert ConfigLoader().get_config(name).microscope_config == name


# ------------------------------------------------------------------ --config-file: loader

def test_config_file_adds_a_microscope_and_keeps_the_built_ins(tmp_path, caplog):
    path = write_json(tmp_path / 'mine.json', {'My-Scope': scope(beam_center_x=999)})
    with caplog.at_level(logging.WARNING):
        loader = ConfigLoader(str(path))
    assert loader.get_config('My-Scope').beam_center_x == 999
    assert set(PACKAGED) < set(loader.configs)
    assert 'added My-Scope' in caplog.text and str(path) in caplog.text


def test_config_file_replaces_a_built_in_entry_whole(tmp_path, caplog):
    path = write_json(tmp_path / 'mine.json', {'F30-TVIPS-SM': scope(rotation_axis='1 0 0')})
    with caplog.at_level(logging.WARNING):
        params = ConfigLoader(str(path)).get_config('F30-TVIPS-SM')
    assert params.rotation_axis == '1 0 0'
    assert params.file_extension == scope()['file_extension']   # nothing kept from the built-in
    assert 'replaced built-in F30-TVIPS-SM' in caplog.text


@pytest.mark.parametrize('content, message', [
    (None, 'Cannot read config file'),
    ('{not json', 'not valid JSON'),
    ('[]', 'must be a JSON object'),
    ('{}', 'must be a JSON object'),
    ('{"X": 5}', "entry 'X' must be a JSON object"),
    ('{"X": {"rotation_axis": "1 0 0"}}', "entry 'X' is invalid"),          # missing fields
])
def test_bad_config_file_raises(tmp_path, content, message):
    path = tmp_path / 'mine.json'
    if content is not None:
        path.write_text(content)
    with pytest.raises(ValueError, match=message):
        ConfigLoader(str(path))


def test_misspelled_field_in_an_unselected_entry_is_still_caught(tmp_path):
    entry = scope()
    entry['beam_centre_x'] = entry.pop('beam_center_x')
    path = write_json(tmp_path / 'mine.json', {'Good': scope(), 'Typo': entry})
    with pytest.raises(ValueError, match="entry 'Typo' is invalid"):
        ConfigLoader(str(path))


# ------------------------------------------------------------------ --config-file: CLI

def test_cli_uses_config_file_values_and_names(monkeypatch, tmp_path):
    path = write_json(tmp_path / 'mine.json', {'My-Scope': scope(beam_center_y=777, frame_size=4096)})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['autoprocess', '--config-file', 'mine.json',
                                      '--microscope-config', 'My-Scope'])
    params = parse_arguments('autoprocess')
    assert params.microscope_config == 'My-Scope'
    assert params.beam_center_y == 777 and params.frame_size == 4096
    assert params.config_file == str(path)


def test_cli_without_config_file_is_unchanged(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['autoprocess', '--microscope-config', 'F30-TVIPS-SM'])
    params = parse_arguments('autoprocess')
    assert params.config_file is None
    assert params.rotation_axis == PACKAGED['F30-TVIPS-SM']['rotation_axis']


def test_cli_rejects_a_name_only_defined_in_an_unused_file(monkeypatch, tmp_path):
    write_json(tmp_path / 'mine.json', {'My-Scope': scope()})
    monkeypatch.setattr(sys, 'argv', ['autoprocess', '--microscope-config', 'My-Scope'])
    with pytest.raises(SystemExit) as exc:
        parse_arguments('autoprocess')
    assert exc.value.code == 2


def test_cli_bad_config_file_is_a_usage_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, 'argv', ['image_process', '--config-file', str(tmp_path / 'nope.json')])
    with pytest.raises(SystemExit) as exc:
        parse_arguments('image_process')
    assert exc.value.code == 2
    assert 'Cannot read config file' in capsys.readouterr().err


# ------------------------------------------------------------------ monitorED

def test_monitored_forwards_an_absolute_config_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = _absolutize_config_file(['--dqa', '--config-file', 'mine.json', '--config-file=b.json'])
    assert args == ['--dqa', '--config-file', os.path.join(os.getcwd(), 'mine.json'),
                    '--config-file=' + os.path.join(os.getcwd(), 'b.json')]


def test_monitored_rejects_a_bad_config_file_at_startup(monkeypatch, tmp_path):
    from pyautoprocess import monitor_ed
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['monitorED', '--autoprocess', '--config-file', 'nope.json'])
    monkeypatch.setattr(monitor_ed, 'MonitorED', lambda **kw: pytest.fail('monitor must not start'))
    with pytest.raises(SystemExit) as exc:
        monitor_ed.main()
    assert exc.value.code == 2

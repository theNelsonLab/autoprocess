"""
Microscope configs must load reliably, and a failure to load them must stop the run.

The loader used the deprecated importlib.resources.open_text inside `except Exception`.
Wherever DeprecationWarning is raised as an error, that call failed, the failure was
logged at DEBUG, and every run silently fell back to a hard-coded generic config --
wrong geometry for every microscope, with no visible sign.
"""
import json
import logging
import warnings

import pytest

from pyautoprocess.config import config_manager
from pyautoprocess.config.config_manager import ConfigLoader


def test_packaged_configs_load():
    configs = ConfigLoader().configs
    assert 'default' in configs
    assert 'F30-TVIPS-SM' in configs
    assert len(configs) > 1


def test_packaged_configs_load_with_deprecation_warnings_as_errors():
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        configs = ConfigLoader().configs
    assert 'F30-TVIPS-SM' in configs


def _break_package_resource(monkeypatch, tmp_path):
    missing = tmp_path / 'not-a-package'
    monkeypatch.setattr(config_manager.importlib.resources, 'files', lambda _pkg: missing)


def test_unreadable_package_without_local_file_raises(monkeypatch, tmp_path):
    _break_package_resource(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match='Could not load microscope configs'):
        ConfigLoader(config_path=str(tmp_path / 'absent.json'))


def test_corrupt_package_json_raises(monkeypatch, tmp_path):
    (tmp_path / 'microscope_configs.json').write_text('{not json')
    monkeypatch.setattr(config_manager.importlib.resources, 'files', lambda _pkg: tmp_path)
    with pytest.raises(RuntimeError, match='JSONDecodeError'):
        ConfigLoader(config_path=str(tmp_path / 'absent.json'))


def test_unreadable_package_uses_local_file_and_says_so(monkeypatch, tmp_path, caplog):
    _break_package_resource(monkeypatch, tmp_path)
    local = tmp_path / 'local_configs.json'
    local.write_text(json.dumps({'MyScope': {'rotation_axis': '1 0 0'}}))
    with caplog.at_level(logging.WARNING):
        configs = ConfigLoader(config_path=str(local)).configs
    assert configs == {'MyScope': {'rotation_axis': '1 0 0'}}
    assert str(local) in caplog.text

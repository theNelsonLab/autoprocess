"""
Configuration loading and management
"""
import importlib.resources
import json
import logging
from pathlib import Path
from typing import Dict, Optional

from .parameters import ProcessingParameters


class ConfigLoader:
    """Microscope configurations: the packaged set, optionally extended by a user file.

    Entries in `config_path` are added to the packaged configs; an entry whose name matches a
    packaged one replaces it whole. The file is used only when a path is given explicitly, so
    a stray microscope_configs.json in a data directory is never picked up by accident.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.configs = self._load_packaged_configs()
        if config_path is not None:
            self._apply_config_file(Path(config_path))

    @staticmethod
    def _load_packaged_configs() -> Dict:
        # The configs ship inside the package, so failing to read them means a broken
        # install. That must stop the run: substituting a generic default would process
        # every dataset with the wrong microscope geometry while looking like success.
        try:
            resource = importlib.resources.files('pyautoprocess.data') / 'microscope_configs.json'
            with resource.open('r', encoding='utf-8') as f:
                return json.load(f)
        except (OSError, ModuleNotFoundError, json.JSONDecodeError) as pkg_error:
            raise RuntimeError(
                f"Could not load the packaged microscope configs "
                f"({type(pkg_error).__name__}: {pkg_error}). Reinstall pyautoprocess.") from pkg_error

    def _apply_config_file(self, path: Path) -> None:
        try:
            with open(path, encoding='utf-8') as f:
                user_configs = json.load(f)
        except OSError as e:
            raise ValueError(f"Cannot read config file {path}: {e.strerror or e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Config file {path} is not valid JSON: {e}") from e

        if not isinstance(user_configs, dict) or not user_configs:
            raise ValueError(
                f"Config file {path} must be a JSON object mapping microscope names to settings")

        # Check every entry now, not just the one this run selects, so a mistake in the file
        # is reported the first time it is used rather than on some later run.
        for name, entry in user_configs.items():
            if not isinstance(entry, dict):
                raise ValueError(f"Config file {path}: entry '{name}' must be a JSON object")
            try:
                self._to_parameters(name, entry)
            except TypeError as e:
                raise ValueError(f"Config file {path}: entry '{name}' is invalid: {e}") from e

        replaced = sorted(n for n in user_configs if n in self.configs)
        added = sorted(n for n in user_configs if n not in self.configs)
        self.configs.update(user_configs)
        changes = []
        if added:
            changes.append(f"added {', '.join(added)}")
        if replaced:
            changes.append(f"replaced built-in {', '.join(replaced)}")
        logging.warning(f"Using microscope configs from {path}: {'; '.join(changes)}")

    def get_config(self, microscope_name: str) -> ProcessingParameters:
        # An unknown name is an error, not a cue to substitute generic settings: those
        # would process the data with the wrong geometry while appearing to succeed.
        if microscope_name not in self.configs:
            available = ', '.join(sorted(self.configs))
            raise ValueError(
                f"Unknown microscope configuration '{microscope_name}'. "
                f"Available: {available}")
        logging.info(f"Loaded configuration for {microscope_name}")
        return self._to_parameters(microscope_name, self.configs[microscope_name])

    @staticmethod
    def _to_parameters(microscope_name: str, config: Dict) -> ProcessingParameters:
        # Remove microscope_config from parameters if present, and route
        # filename-derived fields (detector_distance/rotation/exposure) into
        # the default_* slots so params.<field> stays reserved for CLI overrides.
        config = {k: v for k, v in config.items() if k != 'microscope_config'}
        for cli_field, default_field in (
            ('detector_distance', 'default_detector_distance'),
            ('rotation', 'default_rotation'),
            ('exposure', 'default_exposure'),
        ):
            if cli_field in config:
                config[default_field] = config.pop(cli_field)

        return ProcessingParameters(**config, microscope_config=microscope_name)

    def get_available_configs(self) -> list:
        """Return list of available microscope configurations"""
        return list(self.configs.keys())
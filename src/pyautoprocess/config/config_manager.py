"""
Configuration loading and management
"""
import importlib.resources
import json
import logging
from pathlib import Path
from typing import Dict

from .parameters import ProcessingParameters


class ConfigLoader:
    def __init__(self, config_path: str = "microscope_configs.json"):
        self.config_path = config_path
        self.configs = self._load_configs()

    def _load_configs(self) -> Dict:
        # The configs ship inside the package, so failing to read them means a broken
        # install. That must stop the run: substituting a generic default would process
        # every dataset with the wrong microscope geometry while looking like success.
        try:
            resource = importlib.resources.files('pyautoprocess.data') / 'microscope_configs.json'
            with resource.open('r', encoding='utf-8') as f:
                return json.load(f)
        except (OSError, ModuleNotFoundError, json.JSONDecodeError) as pkg_error:
            pkg_problem = f"{type(pkg_error).__name__}: {pkg_error}"

        # A local file is an explicit substitute, so use it -- but say so.
        if Path(self.config_path).exists():
            logging.warning(
                f"Could not read the packaged microscope configs ({pkg_problem}); "
                f"using local file {self.config_path}")
            with open(self.config_path, encoding='utf-8') as f:
                return json.load(f)

        raise RuntimeError(
            f"Could not load microscope configs: packaged microscope_configs.json is "
            f"unreadable ({pkg_problem}) and no local {self.config_path} exists. "
            f"Reinstall pyautoprocess.")

    def get_config(self, microscope_name: str) -> ProcessingParameters:
        # An unknown name is an error, not a cue to substitute generic settings: those
        # would process the data with the wrong geometry while appearing to succeed.
        if microscope_name not in self.configs:
            available = ', '.join(sorted(self.configs))
            raise ValueError(
                f"Unknown microscope configuration '{microscope_name}'. "
                f"Available: {available}")
        config = self.configs[microscope_name]
        logging.info(f"Loaded configuration for {microscope_name}")

        # Remove microscope_config from parameters if present, and route
        # filename-derived fields (detector_distance/rotation/exposure) into
        # the default_* slots so params.<field> stays reserved for CLI overrides.
        if isinstance(config, dict):
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
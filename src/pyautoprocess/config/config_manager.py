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

    @staticmethod
    def _get_default_config() -> Dict:
        """Return default configuration"""
        return {
            "rotation_axis": "-1 0 0",
            "frame_size": 2048,
            "signal_pixel": 7,
            "min_pixel": 7,
            "background_pixel": 4,
            "pixel_size": 0.028,
            "wavelength": "0.0251",
            "beam_center_x": 1030,
            "beam_center_y": 1040,
            "file_extension": ".ser",
            "value_range_min": 6000.0,
            "value_range_max": 30000.0,
            "detector_distance": "960",
            "rotation": "0.3",
            "exposure": "3",
            "background_range_start": 1,
            "background_range_end": 10
        }

    def get_config(self, microscope_name: str) -> ProcessingParameters:
        if microscope_name not in self.configs:
            logging.warning(f"Configuration '{microscope_name}' not found, using default")
            config = self._get_default_config()
        else:
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
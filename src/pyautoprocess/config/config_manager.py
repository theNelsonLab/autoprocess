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
        try:
            # Try loading from package first
            try:
                with importlib.resources.open_text('pyautoprocess.data', 'microscope_configs.json') as f:
                    return json.load(f)
            except Exception as pkg_error:
                logging.debug(f"Could not load from package: {pkg_error}")

            # Try loading from local path
            if Path(self.config_path).exists():
                with open(self.config_path) as f:
                    return json.load(f)

            logging.warning("No config file found, using default configuration")
            return {"default": self._get_default_config()}

        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return {"default": self._get_default_config()}

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
            "file_extension": ".ser"
        }

    def get_config(self, microscope_name: str) -> ProcessingParameters:
        if microscope_name not in self.configs:
            logging.warning(f"Configuration '{microscope_name}' not found, using default")
            config = self._get_default_config()
        else:
            config = self.configs[microscope_name]
            logging.info(f"Loaded configuration for {microscope_name}")

        # Remove microscope_config from parameters if present
        if isinstance(config, dict):
            config = {k: v for k, v in config.items() if k != 'microscope_config'}

        return ProcessingParameters(**config, microscope_config=microscope_name)

    def get_available_configs(self) -> list:
        """Return list of available microscope configurations"""
        return list(self.configs.keys())
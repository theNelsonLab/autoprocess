"""
Configuration loading and management
"""
import importlib.resources
import json
import logging
from pathlib import Path
from typing import Dict, Optional

from .parameters import ProcessingParameters


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return (_is_int(value) or isinstance(value, float)) and not isinstance(value, bool)


def _is_numeric_text(value) -> bool:
    """A number, or a string holding one (the built-in configs store some values as text)."""
    if _is_number(value):
        return True
    if isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False


def _is_axis(value) -> bool:
    parts = value.split() if isinstance(value, str) else []
    return len(parts) == 3 and all(_is_numeric_text(p) for p in parts)


SUPPORTED_EXTENSIONS = ('.mrc', '.ser', '.tvips')

# The --config-file schema: field -> (required, check, description). Documented in the README
# under "Custom microscope configurations"; keep the two in step.
CONFIG_SCHEMA = {
    'rotation_axis':    (True,  _is_axis, 'three numbers in a string, e.g. "-1 0 0"'),
    'frame_size':       (True,  _is_int, 'an integer (pixels)'),
    'signal_pixel':     (True,  _is_int, 'an integer'),
    'min_pixel':        (True,  _is_int, 'an integer'),
    'background_pixel': (True,  _is_int, 'an integer'),
    'pixel_size':       (True,  _is_number, 'a number (mm)'),
    'wavelength':       (True,  _is_numeric_text, 'a number or numeric string (A)'),
    'beam_center_x':    (True,  _is_number, 'a number (pixels)'),
    'beam_center_y':    (True,  _is_number, 'a number (pixels)'),
    'file_extension':   (True,  lambda v: v in SUPPORTED_EXTENSIONS, f'one of {", ".join(SUPPORTED_EXTENSIONS)}'),
    'value_range_min':  (False, _is_number, 'a number'),
    'value_range_max':  (False, _is_number, 'a number'),
    'detector_distance': (False, _is_numeric_text, 'a number or numeric string (mm)'),
    'rotation':         (False, _is_numeric_text, 'a number or numeric string (deg/s)'),
    'exposure':         (False, _is_numeric_text, 'a number or numeric string (s)'),
    'background_range_start': (False, _is_int, 'an integer (frame)'),
    'background_range_end':   (False, _is_int, 'an integer (frame)'),
    'microscope_config': (False, lambda v: isinstance(v, str), 'a string (ignored; the key is the name)'),
}


def validate_config_entry(name: str, entry) -> list:
    """Problems with one configuration entry, each naming the entry and the field."""
    if not isinstance(entry, dict):
        return [f"entry '{name}': must be a JSON object of settings, got {type(entry).__name__}"]
    problems = []
    for field, (required, check, description) in CONFIG_SCHEMA.items():
        if field not in entry:
            if required:
                problems.append(f"entry '{name}', field '{field}': missing (required; {description})")
        elif not check(entry[field]):
            problems.append(f"entry '{name}', field '{field}': must be {description}, "
                            f"got {entry[field]!r}")
    for field in entry:
        if field not in CONFIG_SCHEMA:
            problems.append(f"entry '{name}', field '{field}': unknown field")
    return problems


def list_microscope_configs(config_file: Optional[str] = None) -> list:
    """Names of the available microscope configurations, including any added by config_file.

    The supported way for other tools to discover names; do not read the package's data
    files directly. Raises ValueError for an invalid config_file.
    """
    return ConfigLoader(config_file).get_available_configs()


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
        problems = [p for name, entry in user_configs.items()
                    for p in validate_config_entry(name, entry)]
        if problems:
            raise ValueError(f"Config file {path} is invalid:\n  " + "\n  ".join(problems))

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
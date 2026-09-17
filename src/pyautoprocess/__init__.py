"""
PyAutoProcess - Automated processing and analysis of MicroED data
"""
# Import key classes for backward compatibility
from .autoprocess import CrystallographyProcessor
from .config.parameters import ProcessingParameters
from .config.config_manager import ConfigLoader, list_microscope_configs

__version__ = "0.5.2"
__all__ = [
    'CrystallographyProcessor',
    'ProcessingParameters',
    'ConfigLoader',
    'list_microscope_configs',
]
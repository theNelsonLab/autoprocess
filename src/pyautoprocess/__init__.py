"""
PyAutoProcess - Automated processing and analysis of MicroED data
"""
# Import key classes for backward compatibility
from .autoprocess import CrystallographyProcessor
from .config.parameters import ProcessingParameters
from .config.config_manager import ConfigLoader

__version__ = "0.4.1"
__all__ = [
    'CrystallographyProcessor',
    'ProcessingParameters',
    'ConfigLoader'
]
"""
Processing parameters data structures
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProcessingParameters:
    rotation_axis: str
    frame_size: int
    signal_pixel: int
    min_pixel: int
    background_pixel: int
    pixel_size: float
    wavelength: str
    beam_center_x: int
    beam_center_y: int
    file_extension: str
    detector_distance: Optional[str] = None
    exposure: Optional[str] = None
    rotation: Optional[str] = None
    microscope_config: str = "default"
    pointless: bool = False
    parallel: bool = False
    quality_analysis: bool = False
    paths: list = None
    reprocess: bool = False
    verbose: bool = False
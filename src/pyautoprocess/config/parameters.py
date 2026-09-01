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
    value_range_min: float = 6000.0
    value_range_max: float = 30000.0
    detector_distance: Optional[str] = None
    exposure: Optional[str] = None
    rotation: Optional[str] = None
    default_detector_distance: Optional[str] = None
    default_rotation: Optional[str] = None
    default_exposure: Optional[str] = None
    microscope_config: str = "default"
    pointless: bool = False
    parallel: bool = False
    quality_analysis: bool = False
    friedel: bool = True
    paths: list = None
    reprocess: bool = False
    verbose: bool = False
    res_range: Optional[float] = None
    min_res: Optional[float] = None
    # Opt-in: detect the beam centre from the frames instead of trusting the config.
    beam_center: bool = False
    # Opt-in: derive the rotation-axis sign from the tilt-direction token in the filename.
    auto_rotation_axis: bool = False
    # Was the value supplied explicitly on the command line, as opposed to inherited from the
    # microscope config? Needed so opt-in auto-detection knows not to override a deliberate choice.
    rotation_axis_explicit: bool = False
    beam_center_x_explicit: bool = False
    beam_center_y_explicit: bool = False
    background_range_start: Optional[int] = None
    background_range_end: Optional[int] = None
    sample_id: Optional[str] = None
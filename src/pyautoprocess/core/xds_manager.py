"""
XDS processing operations for crystallography data
"""
from pathlib import Path
from typing import Tuple


class XDSManager:
    """Handles XDS processing operations for crystallography data"""

    def __init__(self, params, log_print_func=None):
        self.params = params
        self.log_print = log_print_func or self._default_log_print
        self.current_path = Path.cwd()

    def _default_log_print(self, message: str) -> None:
        """Default logging function"""
        print(message)

    def create_xds_input(self, data_path: str, params: dict) -> str:
        """Generate XDS.INP configuration with provided parameters"""
        data_range, spot_range, background_range = self._get_frame_ranges(params)

        # Use parsed oscillation range if available, otherwise calculate from exposure * rotation
        oscillation_range = (params.get('oscillation_range') or
                            (float(params['exposure']) * float(params['rotation'])))

        template = f"""JOB= XYCORR INIT COLSPOT IDXREF DEFPIX INTEGRATE CORRECT
!JOB=DEFPIX INTEGRATE CORRECT
!JOB= CORRECT
ORGX= {self.params.beam_center_x} ORGY= {self.params.beam_center_y}
DETECTOR_DISTANCE= {float(params['distance'])}
OSCILLATION_RANGE= {oscillation_range}
X-RAY_WAVELENGTH= {self.params.wavelength}

NAME_TEMPLATE_OF_DATA_FRAMES= {data_path}_???.tif

BACKGROUND_RANGE={background_range}

!DELPHI=15
!SPACE_GROUP_NUMBER=0
!UNIT_CELL_CONSTANTS= 1 1 1 90 90 90
!REIDX=
INCLUDE_RESOLUTION_RANGE= 40 {params['resolution_range']}
TEST_RESOLUTION_RANGE= 40 {params['test_resolution_range']}
TRUSTED_REGION=0.0 1.2
VALUE_RANGE_FOR_TRUSTED_DETECTOR_PIXELS=6000. 30000.
DETECTOR= ADSC MINIMUM_VALID_PIXEL_VALUE= 1 OVERLOAD= 65000
SENSOR_THICKNESS= 0.01
NX= {self.params.frame_size} NY= {self.params.frame_size} QX= {self.params.pixel_size} QY= {self.params.pixel_size}
ROTATION_AXIS={self.params.rotation_axis}
DIRECTION_OF_DETECTOR_X-AXIS=1 0 0
DIRECTION_OF_DETECTOR_Y-AXIS=0 1 0
INCIDENT_BEAM_DIRECTION=0 0 1
FRACTION_OF_POLARIZATION=0.98
POLARIZATION_PLANE_NORMAL=0 1 0
REFINE(IDXREF)=CELL BEAM ORIENTATION AXIS
REFINE(INTEGRATE)=DISTANCE BEAM ORIENTATION
REFINE(CORRECT)=CELL BEAM ORIENTATION AXIS

DATA_RANGE= {data_range}
SPOT_RANGE= {spot_range}

BACKGROUND_PIXEL= {params['background_pixel']}
SIGNAL_PIXEL= {params['signal_pixel']}
MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT= {params['min_pixel']}
"""
        return template

    def _get_frame_ranges(self, params: dict) -> Tuple[str, str, str]:
        """Get frame ranges in the correct format for XDS.INP."""
        if 'data_start' in params and 'data_end' in params:
            # New quality-based format
            data_range = f"{params['data_start']} {params['data_end']}"
            spot_range = f"{params['data_start']} {params['data_end']}"
            background_range = f"{params['background_start']} {params['background_end']}"
        else:
            # Legacy format for compatibility
            data_range = f"1 {params['image_number']}"
            spot_range = f"1 {params['image_number']}"
            background_range = "1 10"
        return data_range, spot_range, background_range
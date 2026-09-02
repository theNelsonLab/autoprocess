"""
Unified command line argument parsing for autoprocess and image_process
Provides selective argument imports for each tool
"""
import argparse
from typing import Set, Optional
from ..config.config_manager import ConfigLoader
from ..config.parameters import ProcessingParameters


# Argument groups for selective inclusion
COMMON_ARGS = {
    'microscope_config', 'config_file', 'rotation_axis', 'frame_size',
    'signal_pixel', 'min_pixel', 'background_pixel', 'pixel_size',
    'wavelength', 'beam_center_x', 'beam_center_y', 'file_extension',
    'detector_distance', 'exposure', 'rotation', 'pointless', 'parallel',
    'dqa', 'verbose', 'paths', 'res_range', 'min_res', 'friedel', 'background_range',
    'auto_rotation_axis', 'beam_center', 'seed'
}

AUTOPROCESS_ONLY_ARGS = {'reprocess', 'sample_id'}
IMAGE_PROCESS_ONLY_ARGS = {'smv', 'trim_front', 'trim_end'}

ALL_ARGS = COMMON_ARGS | AUTOPROCESS_ONLY_ARGS | IMAGE_PROCESS_ONLY_ARGS


def parse_arguments(tool: str = 'autoprocess', include_args: Optional[Set[str]] = None) -> ProcessingParameters:
    """
    Parse command line arguments with selective inclusion for different tools

    Args:
        tool: Target tool ('autoprocess' or 'image_process')
        include_args: Optional set of specific arguments to include (overrides tool defaults)

    Returns:
        ProcessingParameters with appropriate fields
    """
    # Determine which arguments to include
    if include_args is not None:
        args_to_include = include_args
    elif tool == 'autoprocess':
        args_to_include = COMMON_ARGS | AUTOPROCESS_ONLY_ARGS
    elif tool == 'image_process':
        args_to_include = COMMON_ARGS | IMAGE_PROCESS_ONLY_ARGS
    else:
        raise ValueError(f"Unknown tool: {tool}")

    # Create initial parser for microscope config
    pre_parser = argparse.ArgumentParser(add_help=False)
    config_loader = ConfigLoader()
    available_configs = config_loader.get_available_configs()

    if 'microscope_config' in args_to_include:
        pre_parser.add_argument('--microscope-config',
                               type=str,
                               default='default',
                               choices=available_configs)

    # Get microscope config
    known_args, _ = pre_parser.parse_known_args()
    config = config_loader.get_config(known_args.microscope_config if hasattr(known_args, 'microscope_config') else 'default')

    # Create main parser
    tool_description = {
        'autoprocess': 'Process crystallography data files with conversion and analysis.',
        'image_process': 'Process pre-converted crystallography images with reprocessing capabilities.'
    }

    parser = argparse.ArgumentParser(
        description=tool_description.get(tool, 'Process crystallography data files.'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Add conditional arguments based on tool requirements
    def add_argument_if_needed(arg_name: str, *args, **kwargs):
        """Helper to conditionally add arguments"""
        if arg_name in args_to_include:
            parser.add_argument(*args, **kwargs)

    # Positional arguments (always included)
    if 'paths' in args_to_include:
        help_text = {
            'autoprocess': 'Path(s) to process: single .mrc/.ser/.tvips file, folder containing files, or multiple files/folders. If not specified, processes all files in current directory.',
            'image_process': 'Path(s) to process: folders containing pre-converted images. If not specified, processes all suitable folders in current directory.'
        }
        parser.add_argument('paths',
                           nargs='*',
                           help=help_text.get(tool, help_text['autoprocess']))

    # Common arguments
    add_argument_if_needed('microscope_config', '--microscope-config',
                          type=str, default='default', choices=available_configs,
                          help='Choose instrument configuration')

    add_argument_if_needed('config_file', '--config-file',
                          type=str, default='microscope_configs.json',
                          help='Path to microscope configuration file')

    add_argument_if_needed('rotation_axis', '--rotation-axis',
                          type=str, default=None,
                          help=f'Override rotation axis (microscope config: {config.rotation_axis})')

    add_argument_if_needed('frame_size', '--frame-size',
                          type=int, default=config.frame_size,
                          help='Override frame size')

    add_argument_if_needed('signal_pixel', '--signal-pixel',
                          type=int, default=config.signal_pixel,
                          help='Override signal pixel value')

    add_argument_if_needed('min_pixel', '--min-pixel',
                          type=int, default=config.min_pixel,
                          help='Override minimum pixel value')

    add_argument_if_needed('background_pixel', '--background-pixel',
                          type=int, default=config.background_pixel,
                          help='Override background pixel value')

    add_argument_if_needed('pixel_size', '--pixel-size',
                          type=float, default=config.pixel_size,
                          help='Override pixel size value')

    add_argument_if_needed('wavelength', '--wavelength',
                          type=str, default=config.wavelength,
                          help='Override wavelength value')

    add_argument_if_needed('beam_center_x', '--beam-center-x',
                          type=int, default=None,
                          help=f'Override beam center X coordinate (microscope config: {config.beam_center_x})')

    add_argument_if_needed('beam_center_y', '--beam-center-y',
                          type=int, default=None,
                          help=f'Override beam center Y coordinate (microscope config: {config.beam_center_y})')

    add_argument_if_needed('file_extension', '--file-extension',
                          type=str, default=config.file_extension,
                          help='Override input file extension')

    add_argument_if_needed('detector_distance', '--detector-distance',
                          type=str, default=None,
                          help='Override detector distance (in mm); takes precedence over filename and microscope config')

    add_argument_if_needed('exposure', '--exposure',
                          type=str, default=None,
                          help='Override exposure time; takes precedence over filename and microscope config')

    add_argument_if_needed('rotation', '--rotation',
                          type=str, default=None,
                          help='Override rotation value; takes precedence over filename and microscope config')

    add_argument_if_needed('pointless', '--pointless',
                          action='store_true',
                          help='Run pointless for space group analysis')

    add_argument_if_needed('parallel', '--parallel',
                          action='store_true',
                          help='Use parallel XDS (xds_par) instead of serial XDS')

    add_argument_if_needed('dqa', '--dqa',
                          action='store_true',
                          help='Enable diffraction quality analysis and frame selection')

    add_argument_if_needed('seed', '--seed',
                          type=int, default=None,
                          help='Seed the indexing-retry search so a failed first-pass indexing '
                               'reproduces exactly. Without it those retries use random parameters, '
                               'so a dataset that fails first-pass indexing can give a DIFFERENT '
                               'space group and unit cell on every run. Seeded per dataset, so the '
                               'result does not depend on how many movies preceded it. Note that a '
                               'seed makes failure reproducible too: if a seeded run fails to index, '
                               're-running with the SAME seed repeats it exactly - try a different '
                               'seed, or drop the flag, to explore other retry parameters.')

    add_argument_if_needed('beam_center', '--beam-center',
                          action='store_true',
                          help='EXPERIMENTAL: detect the beam centre from the diffraction frames '
                               'and use it for ORGX/ORGY instead of the microscope-config value. '
                               'Probes the first, middle and last frame; falls back to the config '
                               'value whenever detection is not trustworthy. Off by default.')

    add_argument_if_needed('auto_rotation_axis', '--auto-rotation-axis',
                          action='store_true',
                          help='EXPERIMENTAL: derive the rotation-axis sign from the tilt-direction '
                               'token in the filename (e.g. P50toN-50). A PtoN sweep keeps the axis '
                               'as configured; an NtoP sweep negates it. Falls back to the configured '
                               'axis when no token is present. Off by default.')

    add_argument_if_needed('verbose', '--verbose',
                          action='store_true',
                          help='Enable verbose logging for detailed conversion validation')

    add_argument_if_needed('res_range', '--res-range',
                          type=float, default=None,
                          help='Manual resolution range in Angstroms (overrides calculated values)')

    add_argument_if_needed('min_res', '--min-res',
                          type=float, default=None,
                          help='Minimum resolution for XSCALE in Angstroms (overrides INCLUDE_RESOLUTION_RANGE for scaling)')

    add_argument_if_needed('friedel', '--friedel',
                          type=lambda x: x.lower() == 'true', default=True,
                          help="Set Friedel's law for XDS (true or false, default: true)")

    config_bg_default = None
    if config.background_range_start is not None and config.background_range_end is not None:
        config_bg_default = [config.background_range_start, config.background_range_end]
    add_argument_if_needed('background_range', '--background-range',
                          type=int, nargs=2, metavar=('START', 'END'), default=config_bg_default,
                          help='Custom background range as two integers (start end). Overrides microscope config default.')

    # Tool-specific arguments
    add_argument_if_needed('reprocess', '--reprocess',
                          action='store_true',
                          help='Reprocess files even if they have been processed before')

    add_argument_if_needed('sample_id', '--id',
                          dest='sample_id', type=str, default=None,
                          help='Override (or supply, if the filename has none) the sample name. '
                               'autoprocess only; image_process derives the sample name from the folder.')

    add_argument_if_needed('smv', '--smv',
                          action='store_true',
                          help='Process SMV (.img) files instead of TIF files')

    add_argument_if_needed('trim_front', '--trim-front',
                          type=int, default=0,
                          help='Number of frames to trim from the start of the range')

    add_argument_if_needed('trim_end', '--trim-end',
                          type=int, default=0,
                          help='Number of frames to trim from the end of the range')

    args = parser.parse_args()

    # Build parameter dictionary based on available arguments
    params = {}

    # Helper function to safely get argument values
    def get_arg_value(arg_name: str, default=None):
        if arg_name in args_to_include and hasattr(args, arg_name):
            return getattr(args, arg_name)
        return default

    # Populate parameters conditionally
    # These three may be auto-detected per dataset by opt-in features, so record whether the
    # user asked for a specific value (which auto-detection must not override) or merely
    # inherited the microscope-config default (which it may).
    _cli_rotation_axis = get_arg_value('rotation_axis', None)
    params['rotation_axis'] = config.rotation_axis if _cli_rotation_axis is None else _cli_rotation_axis
    params['rotation_axis_explicit'] = _cli_rotation_axis is not None
    params['frame_size'] = get_arg_value('frame_size', config.frame_size)
    params['signal_pixel'] = get_arg_value('signal_pixel', config.signal_pixel)
    params['min_pixel'] = get_arg_value('min_pixel', config.min_pixel)
    params['background_pixel'] = get_arg_value('background_pixel', config.background_pixel)
    params['pixel_size'] = get_arg_value('pixel_size', config.pixel_size)
    params['wavelength'] = get_arg_value('wavelength', config.wavelength)
    _cli_beam_center_x = get_arg_value('beam_center_x', None)
    _cli_beam_center_y = get_arg_value('beam_center_y', None)
    params['beam_center_x'] = config.beam_center_x if _cli_beam_center_x is None else _cli_beam_center_x
    params['beam_center_y'] = config.beam_center_y if _cli_beam_center_y is None else _cli_beam_center_y
    params['beam_center_x_explicit'] = _cli_beam_center_x is not None
    params['beam_center_y_explicit'] = _cli_beam_center_y is not None
    params['file_extension'] = get_arg_value('file_extension', config.file_extension)
    params['value_range_min'] = get_arg_value('value_range_min', config.value_range_min)
    params['value_range_max'] = get_arg_value('value_range_max', config.value_range_max)
    params['detector_distance'] = get_arg_value('detector_distance', None)
    params['exposure'] = get_arg_value('exposure', None)
    params['rotation'] = get_arg_value('rotation', None)
    params['default_detector_distance'] = config.default_detector_distance
    params['default_exposure'] = config.default_exposure
    params['default_rotation'] = config.default_rotation
    params['microscope_config'] = get_arg_value('microscope_config', 'default')
    params['pointless'] = get_arg_value('pointless', False)
    params['parallel'] = get_arg_value('parallel', False)
    params['quality_analysis'] = get_arg_value('dqa', False)
    params['auto_rotation_axis'] = get_arg_value('auto_rotation_axis', False)
    params['beam_center'] = get_arg_value('beam_center', False)
    params['seed'] = get_arg_value('seed', None)
    params['paths'] = get_arg_value('paths', [])
    params['reprocess'] = get_arg_value('reprocess', False)
    params['verbose'] = get_arg_value('verbose', False)
    params['res_range'] = get_arg_value('res_range', None)
    params['friedel'] = get_arg_value('friedel', True)
    params['min_res'] = get_arg_value('min_res', None)

    # Handle background_range (convert from [start, end] to two separate parameters)
    background_range = get_arg_value('background_range', None)
    if background_range is not None and len(background_range) == 2:
        params['background_range_start'] = background_range[0]
        params['background_range_end'] = background_range[1]
    else:
        params['background_range_start'] = None
        params['background_range_end'] = None

    # autoprocess-only: sample id override (image_process derives sample name from folder)
    if tool == 'autoprocess':
        params['sample_id'] = get_arg_value('sample_id', None)

    # Handle image_process specific parameters
    if tool == 'image_process':
        # Import the extended parameters class from image_process module
        from ..image_process import ExtendedProcessingParameters
        params['smv'] = get_arg_value('smv', False)
        params['trim_front'] = get_arg_value('trim_front', 0)
        params['trim_end'] = get_arg_value('trim_end', 0)
        return ExtendedProcessingParameters(**params)
    else:
        return ProcessingParameters(**params)


# Convenience functions for each tool
def parse_autoprocess_arguments() -> ProcessingParameters:
    """Parse arguments specifically for autoprocess"""
    return parse_arguments('autoprocess')


def parse_image_process_arguments():
    """Parse arguments specifically for image_process"""
    return parse_arguments('image_process')
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
    'dqa', 'verbose', 'paths'
}

AUTOPROCESS_ONLY_ARGS = {'reprocess'}
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
            'autoprocess': 'Path(s) to process: single .mrc/.ser file, folder containing files, or multiple files/folders. If not specified, processes all files in current directory.',
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
                          type=str, default=config.rotation_axis,
                          help='Override rotation axis')

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
                          type=int, default=config.beam_center_x,
                          help='Override beam center X coordinate')

    add_argument_if_needed('beam_center_y', '--beam-center-y',
                          type=int, default=config.beam_center_y,
                          help='Override beam center Y coordinate')

    add_argument_if_needed('file_extension', '--file-extension',
                          type=str, default=config.file_extension,
                          help='Override input file extension')

    add_argument_if_needed('detector_distance', '--detector-distance',
                          type=str, default=config.detector_distance,
                          help='Override detector distance (in mm)')

    add_argument_if_needed('exposure', '--exposure',
                          type=str, default=config.exposure,
                          help='Override exposure time')

    add_argument_if_needed('rotation', '--rotation',
                          type=str, default=config.rotation,
                          help='Override rotation value')

    add_argument_if_needed('pointless', '--pointless',
                          action='store_true',
                          help='Run pointless for space group analysis')

    add_argument_if_needed('parallel', '--parallel',
                          action='store_true',
                          help='Use parallel XDS (xds_par) instead of serial XDS')

    add_argument_if_needed('dqa', '--dqa',
                          action='store_true',
                          help='Enable diffraction quality analysis and frame selection')

    add_argument_if_needed('verbose', '--verbose',
                          action='store_true',
                          help='Enable verbose logging for detailed conversion validation')

    # Tool-specific arguments
    add_argument_if_needed('reprocess', '--reprocess',
                          action='store_true',
                          help='Reprocess files even if they have been processed before')

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
    params['rotation_axis'] = get_arg_value('rotation_axis', config.rotation_axis)
    params['frame_size'] = get_arg_value('frame_size', config.frame_size)
    params['signal_pixel'] = get_arg_value('signal_pixel', config.signal_pixel)
    params['min_pixel'] = get_arg_value('min_pixel', config.min_pixel)
    params['background_pixel'] = get_arg_value('background_pixel', config.background_pixel)
    params['pixel_size'] = get_arg_value('pixel_size', config.pixel_size)
    params['wavelength'] = get_arg_value('wavelength', config.wavelength)
    params['beam_center_x'] = get_arg_value('beam_center_x', config.beam_center_x)
    params['beam_center_y'] = get_arg_value('beam_center_y', config.beam_center_y)
    params['file_extension'] = get_arg_value('file_extension', config.file_extension)
    params['detector_distance'] = get_arg_value('detector_distance', config.detector_distance)
    params['exposure'] = get_arg_value('exposure', config.exposure)
    params['rotation'] = get_arg_value('rotation', config.rotation)
    params['microscope_config'] = get_arg_value('microscope_config', 'default')
    params['pointless'] = get_arg_value('pointless', False)
    params['parallel'] = get_arg_value('parallel', False)
    params['quality_analysis'] = get_arg_value('dqa', False)
    params['paths'] = get_arg_value('paths', [])
    params['reprocess'] = get_arg_value('reprocess', False)
    params['verbose'] = get_arg_value('verbose', False)

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
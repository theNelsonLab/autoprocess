import numpy as np
import logging
import tifffile
import argparse
import warnings
from pathlib import Path
import mrcfile

class MRCConverter:
    @staticmethod
    def setup_logging(log_file: str, dir_name: str) -> tuple[logging.Logger, logging.Logger]:
        """
        Configure two loggers: one for detailed file logging and one for concise console output.
        Returns tuple of (file_logger, console_logger)
        """
        log_dir = Path.cwd() / dir_name
        log_dir.mkdir(exist_ok=True)
        
        # Create and configure file logger (detailed output)
        file_logger = logging.getLogger('file_logger')
        file_logger.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(message)s')
        log_path = log_dir / log_file
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setFormatter(file_formatter)
        file_logger.addHandler(file_handler)
        
        # Create and configure console logger (concise output)
        console_logger = logging.getLogger('console_logger')
        console_logger.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        console_logger.addHandler(console_handler)
        
        return file_logger, console_logger

    @staticmethod
    def log_to_file(message: str, file_logger: logging.Logger) -> None:
        """Log detailed message to file only."""
        file_logger.info(message)

    @staticmethod
    def log_to_console(message: str, console_logger: logging.Logger) -> None:
        """Log concise message to console only."""
        console_logger.info(message)

    @staticmethod
    def log_to_both(message: str, file_logger: logging.Logger, console_logger: logging.Logger) -> None:
        """Log message to both file and console."""
        file_logger.info(message)
        console_logger.info(message)

    @staticmethod
    def convert_mrc_to_tif(mrc_file_path: str, pedestal: int, tif_name: str | None, 
                          raw_conversion: bool,
                          file_logger: logging.Logger, console_logger: logging.Logger) -> bool:
        """Convert .mrc movie to .tif frames. Returns True if all verifications pass."""
        try:
            # Get output directory path
            mrc_dir = Path(mrc_file_path).parent
            images_dir = mrc_dir / 'images'
            images_dir.mkdir(exist_ok=True)
            
            # Load .mrc movie using mrcfile package
            with mrcfile.open(str(mrc_file_path)) as mrc:
                # Handle single vs multi-frame data
                if len(mrc.data.shape) == 2:
                    frames = [mrc.data]
                    overall_data = mrc.data
                else:
                    frames = [mrc.data[i] for i in range(mrc.data.shape[0])]
                    overall_data = mrc.data

            base_name = tif_name if tif_name is not None else Path(mrc_file_path).stem

            # Log data type information
            MRCConverter.log_to_file(f"Original data type: {overall_data.dtype}", file_logger)
            
            overall_stats_msg = (
                f"\nOverall statistics for {base_name}:\n"
                f"Raw data - min: {float(np.min(overall_data)):.2f}, "
                f"max: {float(np.max(overall_data)):.2f}, "
                f"mean: {float(np.mean(overall_data)):.2f}"
            )
            MRCConverter.log_to_both(overall_stats_msg, file_logger, console_logger)
            
            all_frames_verified = True
            
            # Process each frame
            for idx, frame_data in enumerate(frames):
                if raw_conversion:
                    output_data = frame_data
                else:
                    output_data = frame_data.astype(np.uint16)
                    if not MRCConverter.check_data_range(frame_data, pedestal, file_logger):
                        all_frames_verified = False
                    if pedestal:
                        output_data = output_data.astype(np.int32) + pedestal
                        output_data = output_data.astype(np.uint16)

                MRCConverter.log_to_file(
                    f"Frame {idx+1} pre-save details:\n"
                    f"Data type: {output_data.dtype}\n"
                    f"Data range: [{np.min(output_data)}, {np.max(output_data)}]",
                    file_logger
                )
                
                output_tif_path = images_dir / f'{base_name}_{idx+1:03d}.tif'
                
                # Save with explicit data type
                tifffile.imwrite(
                    str(output_tif_path),
                    output_data,
                    dtype=output_data.dtype  # Preserve original dtype
                )
                
                # Read back and verify
                saved_tif = tifffile.imread(str(output_tif_path))
                
                MRCConverter.log_to_file(
                    f"Frame {idx+1} post-save details:\n"
                    f"Saved TIF type: {saved_tif.dtype}\n"
                    f"Saved TIF range: [{np.min(saved_tif)}, {np.max(saved_tif)}]",
                    file_logger
                )
                
                MRCConverter.log_frame_statistics(frame_data, saved_tif, idx, base_name, file_logger)
                MRCConverter.log_to_file(f"Frame {idx+1} saved as {output_tif_path}", file_logger)
                
                if not MRCConverter.verify_conversion(frame_data, saved_tif, raw_conversion, pedestal, output_tif_path, file_logger):
                    all_frames_verified = False
            
            if all_frames_verified:
                MRCConverter.log_to_both("\nVerification passed for all frames", file_logger, console_logger)
            else:
                MRCConverter.log_to_both("\nVerification failed for one or more frames", file_logger, console_logger)
            
            return all_frames_verified
                    
        except Exception as e:
            error_msg = f"Error processing {mrc_file_path}: {e}"
            MRCConverter.log_to_both(error_msg, file_logger, console_logger)
            return False

    

    @staticmethod
    def check_data_range(data: np.ndarray, pedestal: int, file_logger: logging.Logger) -> bool:
        """Check if data will fit within uint16 range."""
        min_val = np.min(data) + pedestal
        max_val = np.max(data) + pedestal
        
        if min_val < 0 or max_val > 65535:
            warning_msg = (
                f"Data range ({min_val}, {max_val}) exceeds uint16 range (0, 65535). "
                f"Original data range: ({np.min(data)}, {np.max(data)}), "
                f"Pedestal: {pedestal}"
            )
            warnings.warn(warning_msg)
            MRCConverter.log_to_file(warning_msg, file_logger)
            return False
        return True

    @staticmethod
    def log_frame_statistics(frame_data: np.ndarray, tif_data: np.ndarray, 
                           frame_idx: int, mrc_name: str, file_logger: logging.Logger) -> None:
        """Log frame statistics to file only."""
        stats_msg = (
            f"\nFrame {frame_idx+1} statistics for {mrc_name}:\n"
            f"Original data - min: {float(np.min(frame_data)):.2f}, "
            f"max: {float(np.max(frame_data)):.2f}, "
            f"mean: {float(np.mean(frame_data)):.2f}\n"
            f"TIF data      - min: {float(np.min(tif_data)):.2f}, "
            f"max: {float(np.max(tif_data)):.2f}, "
            f"mean: {float(np.mean(tif_data)):.2f}"
        )
        MRCConverter.log_to_file(stats_msg, file_logger)

    @staticmethod
    def verify_conversion(original_data: np.ndarray, tif_data: np.ndarray,
                         raw_conversion: bool, pedestal: int,
                         tif_file_path: Path, file_logger: logging.Logger) -> bool:
        """Verify conversion and log details to file only."""
        try:
            if raw_conversion:
                compare_original = original_data
                compare_tif = tif_data
            else:
                compare_original = original_data.astype(np.uint16)
                if pedestal:
                    compare_original = compare_original.astype(np.int32) + pedestal
                    compare_original = compare_original.astype(np.uint16)
                compare_tif = tif_data

            # Log detailed comparison info
            MRCConverter.log_to_file(
                f"\nDetailed comparison for {tif_file_path}:\n"
                f"Original dtype: {compare_original.dtype}, TIF dtype: {compare_tif.dtype}\n"
                f"Original range: [{np.min(compare_original)}, {np.max(compare_original)}]\n"
                f"TIF range: [{np.min(compare_tif)}, {np.max(compare_tif)}]",
                file_logger
            )
            
            if np.array_equal(compare_original, compare_tif):
                MRCConverter.log_to_file(f'Verification passed for {tif_file_path}', file_logger)
                return True
            else:
                diff = np.abs(compare_original.astype(np.float32) - compare_tif.astype(np.float32))
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                num_diff = np.sum(diff > 0)
                MRCConverter.log_to_file(
                    f'Verification failed for {tif_file_path}:\n'
                    f'Max difference: {max_diff}\n'
                    f'Mean difference: {mean_diff}\n'
                    f'Number of differing pixels: {num_diff}',
                    file_logger
                )
                return False
                
        except Exception as e:
            MRCConverter.log_to_file(f'Error during verification for {tif_file_path}: {e}', file_logger)
            return False


def main():
    """Main function to handle command line arguments and execute the script."""
    try:
        parser = argparse.ArgumentParser(description='Convert MRC files to TIF format with optional pedestal value.')
        parser.add_argument('--folder', type=str, 
                          help='Path to folder containing MRC files (default: current directory)',
                          default=str(Path.cwd()))
        parser.add_argument('--ped', type=int, 
                          help='Pedestal value to add to each pixel (default: 0)',
                          default=0)
        parser.add_argument('--tif-name', type=str,
                          help='Base name for output TIF files (default: same as MRC filename)',
                          default=None)
        parser.add_argument('--recursive', action='store_true',
                          help='Search for MRC files recursively in subdirectories',
                          default=False)
        parser.add_argument('--raw', action='store_true',
                          help='Convert data without any type conversion or modifications',
                          default=False)
        
        args = parser.parse_args()
        
        # Setup split logging
        file_logger, console_logger = MRCConverter.setup_logging('mrc_conversion.log', 'logs')
        
        folder_path = Path(args.folder)
        if not folder_path.exists():
            raise ValueError(f"The specified path does not exist: {folder_path}")
        
        MRCConverter.log_to_both(f"Processing folder: {folder_path}", file_logger, console_logger)
        if not args.raw:
            MRCConverter.log_to_both(f"Using pedestal value: {args.ped}", file_logger, console_logger)
        if args.tif_name:
            MRCConverter.log_to_both(f"Using custom TIF base name: {args.tif_name}", file_logger, console_logger)
        MRCConverter.log_to_both(f"Recursive search: {'enabled' if args.recursive else 'disabled'}", file_logger, console_logger)
        MRCConverter.log_to_both(f"Conversion mode: {'raw' if args.raw else 'processed'}", file_logger, console_logger)
        
        # Process MRC files
        pattern = '**/*.mrc' if args.recursive else '*.mrc'
        mrc_files = list(folder_path.glob(pattern))
        if not mrc_files:
            MRCConverter.log_to_both("No .mrc files were found in the specified folder", 
                                   file_logger, console_logger)
            return 1
        
        all_files_success = True
        for mrc_file in mrc_files:
            try:
                success = MRCConverter.convert_mrc_to_tif(str(mrc_file), args.ped,
                                                        args.tif_name, args.raw,
                                                        file_logger, console_logger)
                if not success:
                    all_files_success = False
            except Exception as e:
                MRCConverter.log_to_both(f"Failed to convert {mrc_file.name}: {e}", 
                                       file_logger, console_logger)
                all_files_success = False
                continue
        
        return 0 if all_files_success else 1
            
    except Exception as e:
        if 'file_logger' in locals() and 'console_logger' in locals():
            MRCConverter.log_to_both(f"An error occurred: {e}", file_logger, console_logger)
        else:
            print(f"An error occurred before logging was initialized: {e}")
        return 1

if __name__ == "__main__":
    main()
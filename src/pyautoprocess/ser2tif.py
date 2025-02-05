import numpy as np
import logging
import tifffile
import argparse
import warnings
import seremi
from pathlib import Path

class SERConverter:
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
    def convert_ser_to_tif(ser_file_path: str, pedestal: int, tif_name: str | None, 
                          raw_conversion: bool,
                          file_logger: logging.Logger, console_logger: logging.Logger) -> bool:
        """Convert .ser movie to .tif frames. Returns True if all verifications pass."""
        try:
            # Get output directory path
            ser_dir = Path(ser_file_path).parent
            images_dir = ser_dir / 'images'
            images_dir.mkdir(exist_ok=True)
            
            # Load .ser movie using Hyperspy
            with seremi.SERFile(ser_file_path) as ser:
                frames = ser.read_all_frames()

            base_name = tif_name if tif_name is not None else Path(ser_file_path).stem

            # Log data type information
            SERConverter.log_to_file(f"Original data type: {frames[0].dtype}", file_logger)
            
            overall_stats_msg = (
                f"\nOverall statistics for {base_name}:\n"
                f"Raw data - min: {float(np.min(frames)):.2f}, "
                f"max: {float(np.max(frames)):.2f}, "
                f"mean: {float(np.mean(frames)):.2f}"
            )
            SERConverter.log_to_both(overall_stats_msg, file_logger, console_logger)
            
            all_frames_verified = True
            
            # Process each frame
            for idx, frame_data in enumerate(frames):
                # flip vertically to match previous version of autoprocess
                frame_data = frame_data[::-1]

                if raw_conversion:
                    output_data = frame_data
                else:
                    # Apply pedestal and ensure non-negative values
                    output_data = np.maximum(frame_data + pedestal, 0)
                    output_data = output_data.astype(np.uint16)
                    if not SERConverter.check_data_range(frame_data, pedestal, file_logger):
                        all_frames_verified = False

                SERConverter.log_to_file(
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
                
                SERConverter.log_to_file(
                    f"Frame {idx+1} post-save details:\n"
                    f"Saved TIF type: {saved_tif.dtype}\n"
                    f"Saved TIF range: [{np.min(saved_tif)}, {np.max(saved_tif)}]",
                    file_logger
                )
                
                SERConverter.log_frame_statistics(frame_data, saved_tif, idx, base_name, file_logger)
                SERConverter.log_to_file(f"Frame {idx+1} saved as {output_tif_path}", file_logger)
                
                if not SERConverter.verify_conversion(frame_data, saved_tif, raw_conversion, pedestal, output_tif_path, file_logger):
                    all_frames_verified = False
            
            if all_frames_verified:
                SERConverter.log_to_both("\nVerification passed for all frames", file_logger, console_logger)
            else:
                SERConverter.log_to_both("\nVerification failed for one or more frames", file_logger, console_logger)
            
            return all_frames_verified
                    
        except Exception as e:
            error_msg = f"Error processing {ser_file_path}: {e}"
            SERConverter.log_to_both(error_msg, file_logger, console_logger)
            return False

    @staticmethod
    def check_data_range(data: np.ndarray, pedestal: int, file_logger: logging.Logger) -> bool:
        """Check if data will fit within uint16 range."""
        min_val = np.min(data) + pedestal
        max_val = np.max(data) + pedestal
        
        if max_val > 65535:
            warning_msg = (
                f"Data range ({min_val}, {max_val}) exceeds uint16 range (0, 65535). "
                f"Original data range: ({np.min(data)}, {np.max(data)}), "
                f"Pedestal: {pedestal}"
            )
            warnings.warn(warning_msg)
            SERConverter.log_to_file(warning_msg, file_logger)
            return False
        return True

    @staticmethod
    def log_frame_statistics(frame_data: np.ndarray, tif_data: np.ndarray, 
                           frame_idx: int, ser_name: str, file_logger: logging.Logger) -> None:
        """Log frame statistics to file only."""
        stats_msg = (
            f"\nFrame {frame_idx+1} statistics for {ser_name}:\n"
            f"Original data - min: {float(np.min(frame_data)):.2f}, "
            f"max: {float(np.max(frame_data)):.2f}, "
            f"mean: {float(np.mean(frame_data)):.2f}\n"
            f"TIF data      - min: {float(np.min(tif_data)):.2f}, "
            f"max: {float(np.max(tif_data)):.2f}, "
            f"mean: {float(np.mean(tif_data)):.2f}"
        )
        SERConverter.log_to_file(stats_msg, file_logger)

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
                # Apply the same transformation as during conversion
                compare_original = np.maximum(original_data + pedestal, 0).astype(np.uint16)
                compare_tif = tif_data

            # Log detailed comparison info
            SERConverter.log_to_file(
                f"\nDetailed comparison for {tif_file_path}:\n"
                f"Original dtype: {compare_original.dtype}, TIF dtype: {compare_tif.dtype}\n"
                f"Original range: [{np.min(compare_original)}, {np.max(compare_original)}]\n"
                f"TIF range: [{np.min(compare_tif)}, {np.max(compare_tif)}]",
                file_logger
            )
            
            if np.array_equal(compare_original, compare_tif):
                SERConverter.log_to_file(f'Verification passed for {tif_file_path}', file_logger)
                return True
            else:
                diff = np.abs(compare_original.astype(np.float32) - compare_tif.astype(np.float32))
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                num_diff = np.sum(diff > 0)
                SERConverter.log_to_file(
                    f'Verification failed for {tif_file_path}:\n'
                    f'Max difference: {max_diff}\n'
                    f'Mean difference: {mean_diff}\n'
                    f'Number of differing pixels: {num_diff}',
                    file_logger
                )
                return False
                
        except Exception as e:
            SERConverter.log_to_file(f'Error during verification for {tif_file_path}: {e}', file_logger)
            return False

def main():
    """Main function to handle command line arguments and execute the script."""
    try:
        parser = argparse.ArgumentParser(description='Convert SER files to TIF format with optional pedestal value.')
        parser.add_argument('--folder', type=str, 
                          help='Path to folder containing SER files (default: current directory)',
                          default=str(Path.cwd()))
        parser.add_argument('--ped', type=int, 
                          help='Pedestal value to add to each pixel (default: 200)',
                          default=200)
        parser.add_argument('--tif-name', type=str,
                          help='Base name for output TIF files (default: same as SER filename)',
                          default=None)
        parser.add_argument('--recursive', action='store_true',
                          help='Search for SER files recursively in subdirectories',
                          default=False)
        parser.add_argument('--raw', action='store_true',
                          help='Convert data without any type conversion or modifications',
                          default=False)
        
        args = parser.parse_args()
        
        # Setup split logging
        file_logger, console_logger = SERConverter.setup_logging('ser_conversion.log', 'logs')
        
        folder_path = Path(args.folder)
        if not folder_path.exists():
            raise ValueError(f"The specified path does not exist: {folder_path}")
        
        SERConverter.log_to_both(f"Processing folder: {folder_path}", file_logger, console_logger)
        if not args.raw:
            SERConverter.log_to_both(f"Using pedestal value: {args.ped}", file_logger, console_logger)
        if args.tif_name:
            SERConverter.log_to_both(f"Using custom TIF base name: {args.tif_name}", file_logger, console_logger)
        SERConverter.log_to_both(f"Recursive search: {'enabled' if args.recursive else 'disabled'}", file_logger, console_logger)
        SERConverter.log_to_both(f"Conversion mode: {'raw' if args.raw else 'processed'}", file_logger, console_logger)
        
        # Process SER files
        pattern = '**/*.ser' if args.recursive else '*.ser'
        ser_files = list(folder_path.glob(pattern))
        if not ser_files:
            SERConverter.log_to_both("No .ser files were found in the specified folder", 
                                   file_logger, console_logger)
            return 1
        
        all_files_success = True
        for ser_file in ser_files:
            try:
                success = SERConverter.convert_ser_to_tif(str(ser_file), args.ped,
                                                        args.tif_name, args.raw,
                                                        file_logger, console_logger)
                if not success:
                    all_files_success = False
            except Exception as e:
                SERConverter.log_to_both(f"Failed to convert {ser_file.name}: {e}", 
                                       file_logger, console_logger)
                all_files_success = False
                continue
        
        return 0 if all_files_success else 1
            
    except Exception as e:
        if 'file_logger' in locals() and 'console_logger' in locals():
            SERConverter.log_to_both(f"An error occurred: {e}", file_logger, console_logger)
        else:
            print(f"An error occurred before logging was initialized: {e}")
        return 1

if __name__ == "__main__":
    main()
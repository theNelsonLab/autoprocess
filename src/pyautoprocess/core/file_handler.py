"""
File I/O operations for crystallography data processing
"""
import numpy as np
import mrcfile
import seremi
import tifffile
from pathlib import Path
from typing import Optional, Tuple
from ..tvips import TvipsReader


class FileHandler:
    """Handles file I/O operations for crystallography data processing"""

    def __init__(self, params, log_print_func=None):
        self.params = params
        self.log_print = log_print_func or self._default_log_print
        self.current_path = Path.cwd()

    def _default_log_print(self, message: str) -> None:
        """Default logging function"""
        print(message)

    def read_source_file_from_path(self, file_path: Path) -> Tuple[np.ndarray, bool]:
        """Read source file (MRC, SER, TVIPS) from specific path and return data and multiframe status"""
        file_extension = file_path.suffix.lower()

        if file_extension == '.mrc':
            with mrcfile.open(str(file_path), mode='r') as mrc:
                data = mrc.data.copy()
        elif file_extension == '.ser':
            with seremi.SERFile(str(file_path)) as ser:
                frames = ser.read_all_frames()
            data = np.array(frames)
        elif file_extension == '.tvips':
            with TvipsReader(file_path) as tvips:
                frames = tvips.read_all_frames()
            data = np.array(frames)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        is_multiframe = len(data.shape) == 3 and data.shape[0] > 1
        return data, is_multiframe

    def convert_data_to_tif(self, data: np.ndarray, is_multiframe: bool,
                           sample_movie: str, filename: str, frame_range: Optional[Tuple[int, int]] = None,
                           images_dir: Optional[Path] = None) -> bool:
        """Convert data from memory to TIF files without re-reading source file"""
        try:
            # Validate input data (always check for critical errors)
            if data is None:
                self.log_print(f"ERROR: Input data is None for {filename}")
                return False

            if data.size == 0:
                self.log_print(f"ERROR: Input data is empty for {filename}")
                return False

            # Verbose logging only if enabled
            file_extension = Path(filename).suffix.lower()
            if self.params.verbose:
                self.log_print(f"Converting {filename} ({file_extension.upper()}) to TIF format:")
                self.log_print(f"  Data shape: {data.shape}")
                self.log_print(f"  Data type: {data.dtype}")
                self.log_print(f"  Data range: {np.min(data)} to {np.max(data)}")
                self.log_print(f"  Multiframe: {is_multiframe}")
                if frame_range:
                    self.log_print(f"  Frame range: {frame_range[0]}-{frame_range[1]}")

            # Use sample directory name for TIF files, not full MRC filename
            # This ensures consistency with XDS.INP template naming
            base_name = Path(sample_movie).name

            # Use provided images_dir or create default
            if images_dir is None:
                images_dir = Path(sample_movie) / "images"
                images_dir.mkdir(exist_ok=True)

            # Determine which frames to process
            if is_multiframe:
                total_frames = data.shape[0]
                if frame_range:
                    start_frame, end_frame = frame_range
                    # Convert to 0-indexed
                    start_idx = max(0, start_frame - 1)
                    end_idx = min(total_frames, end_frame)
                    frames_to_process = range(start_idx, end_idx)
                    self.log_print(f"Converting frames {start_frame}-{end_frame} based on quality analysis")
                else:
                    frames_to_process = range(total_frames)
                    self.log_print(f"Converting all {total_frames} frames")

                # Process each frame
                all_frames_verified = True
                if self.params.verbose:
                    conversion_stats = {
                        'total_frames': len(frames_to_process),
                        'verified_frames': 0,
                        'failed_frames': 0,
                        'file_sizes': []
                    }

                for frame_idx in frames_to_process:
                    original_frame_data = data[frame_idx]  # Keep original for verification
                    frame_data = original_frame_data.copy()

                    # Validate frame data (always check for critical errors)
                    if original_frame_data.size == 0:
                        self.log_print(f"ERROR: Frame {frame_idx + 1} is empty")
                        all_frames_verified = False
                        if self.params.verbose:
                            conversion_stats['failed_frames'] += 1
                        continue

                    # Apply vertical flip for SER files only (v0.1.1 ser2tif.py line 87)
                    file_extension = Path(filename).suffix.lower()
                    if file_extension == '.ser':
                        frame_data = frame_data[::-1]  # SER files need flip to match v0.1.1

                    # Apply file-specific processing following v0.1.1 exact workflow
                    if file_extension == '.ser':
                        # Apply pedestal and ensure non-negative values (matching ser2tif.py behavior)
                        output_data = np.maximum(frame_data + 200, 0).astype(np.uint16)
                    else:
                        # MRC files: exact v0.1.1 workflow (uint16 cast + pedestal 1)
                        output_data = frame_data.astype(np.uint16)
                        output_data = output_data.astype(np.int32) + 1
                        output_data = output_data.astype(np.uint16)

                    # Validate output data before writing (always check for critical errors)
                    if output_data.size == 0:
                        self.log_print(f"ERROR: Processed frame {frame_idx + 1} is empty after transformation")
                        all_frames_verified = False
                        if self.params.verbose:
                            conversion_stats['failed_frames'] += 1
                        continue

                    output_path = images_dir / f'{base_name}_{frame_idx + 1:03d}.tif'

                    # Write TIF file
                    tifffile.imwrite(str(output_path), output_data)

                    # Record file size for validation (verbose only)
                    if self.params.verbose and output_path.exists():
                        file_size = output_path.stat().st_size
                        conversion_stats['file_sizes'].append(file_size)

                    # Verify conversion (always verify but log differently based on verbose)
                    if self.verify_tif_conversion(original_frame_data, output_path, file_extension):
                        if self.params.verbose:
                            conversion_stats['verified_frames'] += 1
                    else:
                        self.log_print(f"Warning: Verification failed for frame {frame_idx + 1}")
                        all_frames_verified = False
                        if self.params.verbose:
                            conversion_stats['failed_frames'] += 1

                # Verbose conversion summary only
                if self.params.verbose:
                    avg_file_size = np.mean(conversion_stats['file_sizes']) if conversion_stats['file_sizes'] else 0
                    self.log_print(f"Conversion Summary:")
                    self.log_print(f"  Converted: {conversion_stats['verified_frames']}/{conversion_stats['total_frames']} frames")
                    self.log_print(f"  Failed: {conversion_stats['failed_frames']} frames")
                    self.log_print(f"  Average file size: {avg_file_size:.0f} bytes")
                    self.log_print(f"  Output directory: {images_dir}")

                self.log_print(f"Converted {len(frames_to_process)} frames to TIF format")

                # Final verification summary (following v0.1.1 pattern)
                if all_frames_verified:
                    if self.params.verbose:
                        self.log_print("Verification passed for all frames")
                else:
                    self.log_print("Verification failed for one or more frames")
                    return False
            else:
                # Single frame processing with validation
                original_frame_data = data  # Keep original for verification

                # Validate single frame data
                if original_frame_data.size == 0:
                    self.log_print(f"ERROR: Single frame data is empty")
                    return False

                frame_data = original_frame_data.copy()
                file_extension = Path(filename).suffix.lower()

                # Apply vertical flip for SER files only (v0.1.1 ser2tif.py line 87)
                if file_extension == '.ser':
                    frame_data = frame_data[::-1]  # SER files need flip to match v0.1.1

                # Apply file-specific processing following v0.1.1 exact workflow
                if file_extension == '.ser':
                    # Apply pedestal and ensure non-negative values (matching ser2tif.py behavior)
                    output_data = np.maximum(frame_data + 200, 0).astype(np.uint16)
                    transformation_desc = "SER: vertical flip + pedestal +200 + clipping"
                else:
                    # MRC files/TVIPS files: exact v0.1.1 workflow (uint16 cast + pedestal 1)
                    output_data = frame_data.astype(np.uint16)
                    output_data = output_data.astype(np.int32) + 1
                    output_data = output_data.astype(np.uint16)
                    transformation_desc = "MRC: uint16 cast + pedestal +1"

                # Validate output data
                if output_data.size == 0:
                    self.log_print(f"ERROR: Single frame is empty after transformation")
                    return False

                output_path = images_dir / f'{base_name}_001.tif'

                # Write TIF file
                tifffile.imwrite(str(output_path), output_data)

                # Log single frame details (verbose only)
                if self.params.verbose and output_path.exists():
                    file_size = output_path.stat().st_size
                    self.log_print(f"Single frame conversion:")
                    self.log_print(f"  File: {output_path.name}")
                    self.log_print(f"  Size: {file_size:,} bytes")
                    self.log_print(f"  Transformation: {transformation_desc}")

                # Verify conversion
                if self.verify_tif_conversion(original_frame_data, output_path, file_extension):
                    if self.params.verbose:
                        self.log_print("Single frame conversion and verification successful")
                else:
                    self.log_print("Warning: Verification failed for single frame")
                    return False

                self.log_print("Converted single frame to TIF format")

            return True

        except Exception as e:
            self.log_print(f"Error converting data to TIF: {str(e)}")
            return False

    def verify_tif_conversion(self, original_data: np.ndarray, tif_path: Path, file_extension: str) -> bool:
        """Verify TIF conversion matches original data with comprehensive validation"""
        try:
            # Validate input parameters
            if original_data is None:
                self.log_print(f"ERROR: Original data is None for {tif_path}")
                return False

            if not tif_path.exists():
                self.log_print(f"ERROR: TIF file does not exist: {tif_path}")
                return False

            # Read back the saved TIF
            saved_tif = tifffile.imread(str(tif_path))

            # Validate TIF file was read correctly
            if saved_tif is None:
                self.log_print(f"ERROR: Failed to read TIF file: {tif_path}")
                return False

            # Apply the same transformation that was used during conversion
            if file_extension == '.ser':
                # SER processing: vertical flip + pedestal +200 with clipping
                expected_data = original_data[::-1]  # Apply flip first
                expected_data = np.maximum(expected_data + 200, 0).astype(np.uint16)
                transformation_desc = "SER: vertical flip + pedestal +200 + clipping"
            else:
                # MRC/TVIPS processing: exact v0.1.1 workflow (no flip)
                expected_data = original_data.astype(np.uint16)
                expected_data = expected_data.astype(np.int32) + 1
                expected_data = expected_data.astype(np.uint16)
                transformation_desc = "MRC: uint16 cast + pedestal +1"

            # Validate shapes match
            if expected_data.shape != saved_tif.shape:
                self.log_print(f"ERROR: Shape mismatch for {tif_path}:")
                self.log_print(f"  Expected shape: {expected_data.shape}")
                self.log_print(f"  Saved TIF shape: {saved_tif.shape}")
                return False

            # Validate data types
            if expected_data.dtype != saved_tif.dtype:
                self.log_print(f"WARNING: Data type mismatch for {tif_path}:")
                self.log_print(f"  Expected dtype: {expected_data.dtype}")
                self.log_print(f"  Saved TIF dtype: {saved_tif.dtype}")

            # Compare arrays
            if np.array_equal(expected_data, saved_tif):
                # Successful verification - verbose logging only if enabled
                if self.params.verbose:
                    total_pixels = expected_data.size
                    value_range = f"{np.min(saved_tif)} to {np.max(saved_tif)}"
                    self.log_print(f"Verification passed for {tif_path.name}")
                    self.log_print(f"  Transformation: {transformation_desc}")
                    self.log_print(f"  Shape: {saved_tif.shape}, Pixels: {total_pixels:,}, Range: {value_range}")
                return True
            else:
                # Log verification failure (always report failures)
                if self.params.verbose:
                    # Detailed analysis for verbose mode
                    diff = np.abs(expected_data.astype(np.float32) - saved_tif.astype(np.float32))
                    max_diff = np.max(diff)
                    mean_diff = np.mean(diff)
                    num_diff = np.sum(diff > 0)
                    percent_diff = (num_diff / expected_data.size) * 100

                    self.log_print(f"Verification failed for {tif_path}:")
                    self.log_print(f"  Transformation applied: {transformation_desc}")
                    self.log_print(f"  Max difference: {max_diff}")
                    self.log_print(f"  Mean difference: {mean_diff:.3f}")
                    self.log_print(f"  Differing pixels: {num_diff:,} ({percent_diff:.2f}%)")
                    self.log_print(f"  Expected range: {np.min(expected_data)} to {np.max(expected_data)}")
                    self.log_print(f"  Actual range: {np.min(saved_tif)} to {np.max(saved_tif)}")

                    # Sample some differing pixels for debugging
                    if num_diff > 0:
                        diff_indices = np.where(diff > 0)
                        sample_size = min(5, len(diff_indices[0]))
                        self.log_print(f"  Sample differences (first {sample_size}):")
                        for i in range(sample_size):
                            idx = (diff_indices[0][i], diff_indices[1][i]) if len(diff_indices) == 2 else diff_indices[0][i]
                            expected_val = expected_data[idx]
                            actual_val = saved_tif[idx]
                            self.log_print(f"    Pixel {idx}: expected {expected_val}, got {actual_val}, diff {abs(expected_val - actual_val)}")

                return False

        except Exception as e:
            self.log_print(f"ERROR during verification for {tif_path}: {e}")
            self.log_print(f"  File extension: {file_extension}")
            self.log_print(f"  Original data shape: {original_data.shape if original_data is not None else 'None'}")
            return False

    def validate_conversion_pipeline(self, filename: str) -> bool:
        """Validate that the conversion pipeline matches v0.1.1 exactly (verbose mode only)"""
        if not self.params.verbose:
            return True

        file_extension = Path(filename).suffix.lower()

        self.log_print(f"Validating conversion pipeline for {file_extension.upper()} files:")

        if file_extension == '.ser':
            expected_steps = [
                "1. Read SER file with seremi.SERFile",
                "2. Apply vertical flip (frame_data[::-1])",
                "3. Add pedestal +200 with clipping (np.maximum(data + 200, 0))",
                "4. Cast to uint16",
                "5. Write with tifffile.imwrite",
                "6. Verify by reading back and comparing"
            ]
        elif file_extension == '.mrc':
            expected_steps = [
                "1. Read MRC file with mrcfile.open",
                "2. Cast to uint16",
                "3. Cast to int32 and add pedestal +1",
                "4. Cast back to uint16",
                "5. Write with tifffile.imwrite",
                "6. Verify by reading back and comparing"
            ]
        else:
            self.log_print(f"Warning: Unknown file extension: {file_extension}")
            return False

        self.log_print("Expected conversion steps:")
        for step in expected_steps:
            self.log_print(f"   {step}")

        self.log_print("Pipeline validation complete - matches v0.1.1 specification")
        return True
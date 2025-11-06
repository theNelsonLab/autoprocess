"""
Process pre-converted crystallography images
Based on autoprocess.py and batch_reprocess.py
Handles pre-converted TIF images in existing directory structure
"""
import os
import re
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from .autoprocess import CrystallographyProcessor
from .config.parameters import ProcessingParameters
from .quality_analyzer import DiffractionQualityAnalyzer
from .ui.cli_parser import parse_image_process_arguments

@dataclass
class ExtendedProcessingParameters(ProcessingParameters):
    """Extended processing parameters to include SMV flag and frame trimming"""
    smv: bool = False
    trim_front: int = 0
    trim_end: int = 0
    quality_analysis: bool = False
    paths: list = None

class PreConvertedProcessor:
    """Process pre-converted crystallography images with backup functionality"""

    OUTPUT_FOLDER = "auto_process_direct"
    BACKUP_FOLDER = "processing_backups"

    def __init__(self, params: ProcessingParameters):
        self.params = params
        self.processor = CrystallographyProcessor(params)
        self.current_path = Path.cwd()
        self.file_extension = '.img' if hasattr(params, 'smv') and params.smv else '.tif'

    @contextmanager
    def _working_directory(self, path: Path):
        """Context manager for temporarily changing working directory"""
        original_dir = os.getcwd()
        try:
            os.chdir(str(path))
            yield path
        finally:
            os.chdir(original_dir)

    def _create_backup(self, process_dir: Path, sample_name: str) -> Optional[Path]:
        """
        Create a timestamped backup of existing processing directory.
        Preserves all files since crystallography processing requires complete history.

        Args:
            process_dir: Path to the processing directory
            sample_name: Name of the sample being processed

        Returns:
            Path to backup directory if backup was created, None otherwise
        """
        try:
            # Create backup folder inside the sample folder
            backup_root = process_dir.parent / self.BACKUP_FOLDER
            backup_root.mkdir(exist_ok=True)

            # Create timestamped folder name with processing parameters
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            param_suffix = f"d{self.params.detector_distance}_e{self.params.exposure}_r{self.params.rotation}"
            backup_dir = backup_root / f"{sample_name}_{param_suffix}_{timestamp}"

            # Copy entire processing directory to backup
            shutil.copytree(process_dir, backup_dir)

            # Create a summary file with processing parameters
            summary_file = backup_dir / "processing_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Processing Summary for {sample_name}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Detector Distance: {self.params.detector_distance} mm\n")
                f.write(f"Exposure Time: {self.params.exposure} s\n")
                f.write(f"Rotation Rate: {self.params.rotation} deg/s\n")
                f.write(f"Wavelength: {self.params.wavelength}\n")
                f.write(f"Beam Center: ({self.params.beam_center_x}, {self.params.beam_center_y})\n")

            self.processor.log_print(f"Created backup in: {backup_dir}")
            return backup_dir

        except Exception as e:
            self.processor.log_print(f"Warning: Failed to create backup: {str(e)}")
            return None

    def print_image_banner(self):
        """Print the complete banner for image processing mode."""
        self.processor.display.print_full_banner("image_process")

    def validate_image_folder(self, image_path: Path) -> bool:
        """
        Validate that the images folder contains image files (TIF or IMG)
        matching expected pattern
        """
        # Determine file extension based on SMV flag
        pattern = '*.img' if self.params.smv else '*.tif'
        image_files = list(image_path.glob(pattern))
        if not image_files:
            return False

        # We just need to verify that we have a consistent set of numbered images
        # Extract the base name from the first image
        if image_files:
            first_image = image_files[0].name
            # Get the base name by removing the numbering and extension
            base_name = first_image.rsplit('_', 1)[0]

            # Check if all files follow the same pattern
            for img in image_files:
                if not img.name.startswith(base_name):
                    return False

            return True

        return False

    def find_valid_folders(self) -> List[Path]:
        """Find folders containing valid pre-converted images"""
        valid_folders = []

        for item in self.current_path.iterdir():
            if not item.is_dir():
                continue

            images_dir = item / 'images'
            if not images_dir.exists() or not images_dir.is_dir():
                continue

            if self.validate_image_folder(images_dir):
                valid_folders.append(item)

        return valid_folders

    def _count_images(self, image_dir: Path) -> Optional[Tuple[int, str]]:
        """Count images and get base name from image directory"""
        ext = '*.img' if self.params.smv else '*.tif'

        with self._working_directory(image_dir):
            image_files = sorted(list(Path().glob(ext)))
            if not image_files:
                self.processor.log_print(f"No images found in {image_dir}")
                return None

            image_count = len(image_files)
            base_name = image_files[0].name.rsplit('_', 1)[0]
            self.processor.log_print(f"Found {image_count} images")

            return image_count, base_name

    def process_data(self, sample_path: Path) -> None:
        """Process data using the same logic as autoprocess"""
        try:
            image_dir = sample_path / "images"
            result = self._count_images(image_dir)
            if not result:
                return

            image_count, base_name = result
            image_number = str(image_count)

            # Process in auto_process_direct directory
            process_dir = sample_path / self.OUTPUT_FOLDER
            with self._working_directory(process_dir):
                self._process_with_parameters(sample_path, image_number, base_name)

        except Exception as e:
            self.processor.log_print(f"Error processing data: {str(e)}")

    def _process_with_parameters(self, sample_path: Path, image_number: str, base_name: str) -> None:
        """Process data with parameter resolution and XDS execution"""
        try:
            # 1. Command line overrides (highest priority)
            # 2. Source file metadata (if available)
            # 3. Default values (lowest priority)

            # Parse metadata from source file if needed
            metadata = None
            metadata_source = None
            if (self.params.detector_distance is None or
                self.params.rotation is None or
                self.params.exposure is None):
                metadata = self._parse_source_file_metadata(sample_path)
                if metadata:
                    metadata_source = "source file"
                else:
                    # If source file parsing failed, try XDS.INP fallback
                    metadata = self._parse_xds_inp_metadata(sample_path)
                    if metadata:
                        metadata_source = "XDS.INP"

            # Apply priority order for each parameter
            actual_distance = (self.params.detector_distance or
                             (metadata[1] if metadata else None) or
                             "960")

            actual_rotation = (self.params.rotation or
                             (metadata[2] if metadata else None) or
                             "0.3")

            actual_exposure = (self.params.exposure or
                             (metadata[3] if metadata else None) or
                             "3.0")

            # Check if we have oscillation range from XDS.INP
            oscillation_range = metadata[4] if metadata and len(metadata) > 4 else None

            # Calculate resolution ranges using parsed or existing detector distance
            ranges = self.processor.calculate_resolution_ranges(actual_distance)
            if ranges is None:
                self.processor.log_print("Could not calculate resolution ranges")
                return

            resolution_range, test_resolution_range = ranges

            # Log processing parameters
            # Extract sample name from the path
            sample_name = sample_path.name

            # Log processing parameters with correct source attribution
            self.processor.log_print(f"\nProcessing parameters for {sample_name}:")

            # Determine and log source for each parameter
            if self.params.detector_distance:
                self.processor.log_print(f"Detector Distance: {actual_distance} mm (command line override)")
            elif metadata and metadata[1]:
                self.processor.log_print(f"Detector Distance: {actual_distance} mm (from {metadata_source})")
            else:
                self.processor.log_print(f"Detector Distance: {actual_distance} mm (default)")

            # Only log exposure/rotation if we have meaningful values (not when using XDS.INP fallback)
            if not oscillation_range:  # Only show these when NOT using XDS.INP fallback
                if self.params.exposure:
                    self.processor.log_print(f"Exposure Time: {actual_exposure} s (command line override)")
                elif metadata and metadata[3] and metadata_source == "source file":
                    self.processor.log_print(f"Exposure Time: {actual_exposure} s (from {metadata_source})")
                else:
                    self.processor.log_print(f"Exposure Time: {actual_exposure} s (default)")

                if self.params.rotation:
                    self.processor.log_print(f"Rotation Rate: {actual_rotation} deg/s (command line override)")
                elif metadata and metadata[2] and metadata_source == "source file":
                    self.processor.log_print(f"Rotation Rate: {actual_rotation} deg/s (from {metadata_source})")
                else:
                    self.processor.log_print(f"Rotation Rate: {actual_rotation} deg/s (default)")

            # Log oscillation range if parsed from XDS.INP
            if oscillation_range:
                self.processor.log_print(f"Oscillation Range: {oscillation_range} deg (from XDS.INP)")
            self.processor.log_print(f"Resolution Range: {resolution_range} Å")
            self.processor.log_print(f"Test Resolution Range: {test_resolution_range} Å\n")

            # Create initial XDS.INP with .tif extension - we'll modify it later
            params = {
                'distance': actual_distance,
                'rotation': actual_rotation,
                'exposure': actual_exposure,
                'resolution_range': resolution_range,
                'test_resolution_range': test_resolution_range,
                'image_number': image_number,
                'background_pixel': self.params.background_pixel,
                'signal_pixel': self.params.signal_pixel,
                'min_pixel': self.params.min_pixel,
                'oscillation_range': oscillation_range,  # Use parsed oscillation range if available
            }

            data_path = os.path.join(str(Path("..") / "images"), base_name)
            xds_content = self.processor.create_xds_input(data_path, params)

            # Write initial XDS.INP
            with open('XDS.INP', 'w') as xds_inp:
                xds_inp.write(xds_content)

            # If we're in SMV mode, modify the template line
            if self.params.smv:
                self._modify_xds_template_extension()

            # Run diffraction quality analysis if enabled and data available
            quality_start_frame = None
            quality_end_frame = None

            if self.params.quality_analysis:
                # Check DQA data availability in priority order:
                # 1. Existing CSV file (highest priority)
                # 2. Raw data file (MRC/SER) for fresh analysis
                # 3. Only TIF/SMV data available (cannot do DQA)

                existing_csv = self._check_existing_quality_csv(base_name)
                raw_data_available = self._check_raw_data_available(sample_path, base_name)

                analyzer = None

                if existing_csv:
                    # Priority 1: Use existing CSV data
                    self.processor.log_print("Using existing DQA data")
                    analyzer = DiffractionQualityAnalyzer()
                    if not analyzer.load_quality_from_csv(existing_csv):
                        # CSV failed to load, fallback to raw data
                        if raw_data_available:
                            analyzer = self._run_quality_analysis(sample_path, base_name)
                        else:
                            analyzer = None

                elif raw_data_available:
                    # Priority 2: Run analysis on raw data
                    analyzer = self._run_quality_analysis(sample_path, base_name)

                else:
                    # Priority 3: Only converted data available, cannot do DQA
                    self.processor.log_print("DQA is not currently supported for this data type")
                    analyzer = None

                if analyzer:
                    quality_start_frame, quality_end_frame = analyzer.find_good_frame_range()
                    # Final frame range will be logged once below after manual trim consideration

            # Determine final frame range (manual trim overrides quality analysis)
            final_start_frame = 1
            final_end_frame = int(image_number)

            # Apply quality analysis range if available and no manual trim specified
            if quality_start_frame and quality_end_frame and self.params.trim_front == 0 and self.params.trim_end == 0:
                final_start_frame = quality_start_frame
                final_end_frame = quality_end_frame

            # Apply manual trim (overrides quality analysis)
            if self.params.trim_front > 0 or self.params.trim_end > 0:
                final_start_frame = 1 + self.params.trim_front
                final_end_frame = int(image_number) - self.params.trim_end
                # Manual trim logging will be consolidated below

            # Log the final selected frame range
            self.processor.log_print(f"Selected frame range: {final_start_frame}-{final_end_frame}")

            # Apply the frame range modifications if different from full range
            if final_start_frame != 1 or final_end_frame != int(image_number):
                self._modify_frame_ranges_with_range(final_start_frame, final_end_frame, int(image_number))
            else:
                # Even if using full range, still log background range for consistency
                background_start = 1
                background_end = min(10, int(image_number))
                self.processor.log_print(f"Background range: {background_start}-{background_end}")

            # Log initial XDS processing parameters
            self._log_xds_processing_parameters("Initial XDS.INP")

            # Run full autoprocess workflow (XDS + mosaicity + space group + scaling)
            self.processor.log_print(f"\nProcessing {sample_path.name}...\n")
            self.processor.process_check(sample_path.name)

        except Exception as e:
            self.processor.log_print(f"Error processing data: {str(e)}")
        finally:
            os.chdir(str(self.current_path))

    def _modify_xds_template_extension(self) -> None:
        """Modify NAME_TEMPLATE_OF_DATA_FRAMES line in XDS.INP to use .img extension"""
        try:
            with open('XDS.INP', 'r') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                if line.strip().startswith('NAME_TEMPLATE_OF_DATA_FRAMES='):
                    # Replace .tif with .img in the line
                    lines[i] = line.replace('.tif', '.img')
                    break

            with open('XDS.INP', 'w') as f:
                f.writelines(lines)

        except Exception as e:
            self.processor.log_print(f"Error modifying XDS.INP template: {str(e)}")
            raise

    def _check_raw_data_available(self, sample_path: Path, base_name: str) -> bool:
        """Check if raw data files (MRC/SER) are available for fresh DQA analysis"""
        try:
            # Look for MRC or SER files in the parent directory (same level as sample folder)
            parent_dir = sample_path.parent

            # Check for MRC files (exact sample name match with underscore)
            mrc_files = list(parent_dir.glob(f"{base_name}_*.mrc"))
            if mrc_files:
                return True

            # Check for SER files (exact sample name match with underscore)
            ser_files = list(parent_dir.glob(f"{base_name}_*.ser"))
            if ser_files:
                return True

            return False

        except Exception as e:
            self.processor.log_print(f"Error checking for raw data files: {str(e)}")
            return False

    def _check_existing_quality_csv(self, base_name: str) -> Optional[Path]:
        """Check if DQA analysis CSV file already exists in autoprocess logs location"""
        try:
            # Check only in the standard autoprocess location
            logs_dir = Path("../images/logs")
            csv_path = logs_dir / f"{base_name}_quality_analysis.csv"

            if csv_path.exists() and csv_path.stat().st_size > 0:
                return csv_path

            return None
        except Exception as e:
            self.processor.log_print(f"Error checking for existing DQA CSV: {str(e)}")
            return None

    def _run_quality_analysis(self, sample_path: Path, base_name: str) -> Optional[DiffractionQualityAnalyzer]:
        """Run diffraction quality analysis on the original MRC, SER, or TVIPS file"""
        try:
            # Look for the original MRC or SER file in the parent directory (same level as sample folder)
            parent_dir = sample_path.parent
            mrc_files = list(parent_dir.glob(f"{base_name}_*.mrc"))
            ser_files = list(parent_dir.glob(f"{base_name}_*.ser"))
            tvips_files = list(parent_dir.glob(f"{base_name}_*.tvips"))

            source_files = mrc_files + ser_files + tvips_files
            if not source_files:
                self.processor.log_print("No MRC, SER, or TVIPS file found for quality analysis")
                return None

            source_file = source_files[0]

            # Create analyzer and run analysis
            analyzer = DiffractionQualityAnalyzer()
            quality_results = analyzer.analyze_file(str(source_file))

            if quality_results:
                # Log quality distribution like autoprocess
                summary = analyzer.get_quality_summary()
                self.processor.log_print(f"Quality analysis complete:")
                for quality, count in summary['quality_distribution'].items():
                    percentage = (count / summary['total_frames']) * 100
                    self.processor.log_print(f"  {quality}: {count} frames ({percentage:.1f}%)")

                # Save CSV to logs directory
                output_dir = sample_path / "images"
                csv_path = analyzer.save_quality_csv(output_dir, base_name)
                if csv_path:
                    self.processor.log_print(f"Quality analysis CSV saved")

                return analyzer
            else:
                self.processor.log_print("Quality analysis failed")
                return None

        except Exception as e:
            self.processor.log_print(f"Error running quality analysis: {str(e)}")
            return None

    def _parse_source_file_metadata(self, sample_path: Path) -> Optional[tuple]:
        """Parse metadata from MRC/SER file at parent level (same level as sample folder)"""
        try:
            # Look for MRC or SER files in the parent directory (same level as sample folder)
            parent_dir = sample_path.parent
            sample_name = sample_path.name

            # Look for files matching the sample name (exact match with underscore)
            mrc_files = list(parent_dir.glob(f"{sample_name}_*.mrc"))
            ser_files = list(parent_dir.glob(f"{sample_name}_*.ser"))
            tvips_files = list(parent_dir.glob(f"{sample_name}_*.tvips"))

            source_files = mrc_files + ser_files + tvips_files
            if not source_files:
                return None

            source_file = source_files[0]
            filename = source_file.name

            # Parse filename using same format as autoprocess: sample_distance_rotation_exposure.ext
            split = filename.split("_")
            if len(split) < 4:
                self.processor.log_print(f"Filename {filename} doesn't follow expected format: sample_distance_rotation_exposure.ext")
                return None

            sample_movie = split[0]
            distance = split[1]
            rotation = split[2]
            exposure = split[3]

            # Metadata parsed successfully from source file

            return (sample_movie, distance, rotation, exposure)

        except Exception as e:
            self.processor.log_print(f"Error parsing source file metadata: {str(e)}")
            return None

    def _parse_xds_inp_metadata(self, sample_path: Path):
        """Parse detector distance and rotation parameters from existing XDS.INP files as fallback"""
        try:
            # Look for XDS.INP in auto_process or auto_process_direct directories
            xds_paths = [
                sample_path / "auto_process" / "XDS.INP",
                sample_path / "auto_process_direct" / "XDS.INP"
            ]

            for xds_path in xds_paths:
                if xds_path.exists():
                    with open(xds_path, 'r') as f:
                        content = f.read()

                    # Parse DETECTOR_DISTANCE
                    distance_match = re.search(r'DETECTOR_DISTANCE=\s*(\d+(?:\.\d+)?)', content)
                    distance = distance_match.group(1) if distance_match else None

                    # Parse OSCILLATION_RANGE
                    oscillation_match = re.search(r'OSCILLATION_RANGE=\s*(\d+(?:\.\d+)?)', content)
                    oscillation_range = float(oscillation_match.group(1)) if oscillation_match else None

                    if distance and oscillation_range:
                        sample_movie = sample_path.name  # Use directory name as sample name
                        # Return only distance, oscillation range goes as a special marker
                        # rotation and exposure will use defaults since XDS.INP doesn't contain them
                        return (sample_movie, distance, None, None, oscillation_range)

            return None

        except Exception as e:
            self.processor.log_print(f"Error parsing XDS.INP metadata: {str(e)}")
            return None

    def _modify_frame_ranges_with_range(self, start_frame: int, end_frame: int, total_images: int) -> None:
        """Modify DATA_RANGE, SPOT_RANGE, and BACKGROUND_RANGE in XDS.INP with specific frame range"""
        try:
            # Validate the range
            if start_frame >= end_frame or start_frame < 1 or end_frame > total_images:
                self.processor.log_print(
                    f"Warning: Invalid frame range {start_frame}-{end_frame}, using full range"
                )
                return

            # Calculate background range (start + 10 frames, but at least frame 1)
            background_start = max(1, start_frame)
            background_end = min(start_frame + 9, end_frame)

            with open('XDS.INP', 'r') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                if line.strip().startswith('DATA_RANGE='):
                    lines[i] = f'DATA_RANGE= {start_frame} {end_frame}\n'
                elif line.strip().startswith('SPOT_RANGE='):
                    lines[i] = f'SPOT_RANGE= {start_frame} {end_frame}\n'
                elif line.strip().startswith('BACKGROUND_RANGE='):
                    lines[i] = f'BACKGROUND_RANGE= {background_start} {background_end}\n'

            with open('XDS.INP', 'w') as f:
                f.writelines(lines)

            # Log background range only (frame range already logged above)
            background_range = f"{background_start}-{background_end}"
            self.processor.log_print(f"Background range: {background_range}")

        except Exception as e:
            self.processor.log_print(f"Error modifying frame ranges: {str(e)}")

    def _modify_frame_ranges(self, total_images: int) -> None:
        """Modify DATA_RANGE, SPOT_RANGE, and BACKGROUND_RANGE in XDS.INP based on trim parameters"""
        start_frame = 1 + self.params.trim_front
        end_frame = total_images - self.params.trim_end
        self._modify_frame_ranges_with_range(start_frame, end_frame, total_images)

    def _log_xds_processing_parameters(self, context: str = "") -> None:
        """Log current XDS processing parameters from XDS.INP"""
        try:
            if not Path('XDS.INP').exists():
                return

            with open('XDS.INP', 'r') as f:
                content = f.read()

            # Extract the three key parameters
            background_match = re.search(r'BACKGROUND_PIXEL=\s*(\d+)', content)
            signal_match = re.search(r'SIGNAL_PIXEL=\s*(\d+)', content)
            min_spot_match = re.search(r'MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT=\s*(\d+)', content)

            background_val = background_match.group(1) if background_match else "N/A"
            signal_val = signal_match.group(1) if signal_match else "N/A"
            min_spot_val = min_spot_match.group(1) if min_spot_match else "N/A"

            prefix = f"[{context}] " if context else ""
            self.processor.log_print(f"{prefix}XDS Processing Parameters:")
            self.processor.log_print(f"  XDS Background Pixel: {background_val}")
            self.processor.log_print(f"  XDS Signal Pixel: {signal_val}")
            self.processor.log_print(f"  XDS Min Spot Pixels: {min_spot_val}")

        except Exception as e:
            self.processor.log_print(f"Warning: Could not log XDS parameters: {str(e)}")

    def process_folder(self, sample_path: Path) -> bool:
        """Process a single folder containing pre-converted images"""
        try:
            # Setup processing directory
            process_dir = sample_path / self.OUTPUT_FOLDER

            # Handle existing directory - always create backup and proceed (this is a reprocessing tool)
            if process_dir.exists():
                self.processor.log_print(f"\nFound existing processing for {sample_path.name}")
                # Create backup before proceeding
                backup_dir = self._create_backup(process_dir, sample_path.name)

                # Verify backup was successful by checking for XDS.INP
                if backup_dir and (backup_dir / "XDS.INP").exists():
                    # Remove existing directory after successful backup
                    shutil.rmtree(process_dir)
                    self.processor.log_print(f"Reprocessing {sample_path.name}")
                else:
                    self.processor.log_print("Failed to create backup, skipping processing")
                    return False
            else:
                self.processor.log_print(f"\nProcessing {sample_path.name}")

            # Create fresh processing directory
            process_dir.mkdir(parents=True)

            # Process the data
            self.process_data(sample_path)

            return True

        except Exception as e:
            self.processor.log_print(f"Error processing {sample_path.name}: {str(e)}")
            return False

    def _get_folders_to_process(self) -> List[Path]:
        """
        Smart folder detection based on paths argument:
        1. If paths provided: each path should contain /images subdirectory or be a parent containing multiple folders with /images
        2. If no paths provided: find all folders with /images in current directory
        """
        folders_to_process = []

        if hasattr(self.params, 'paths') and self.params.paths:
            # Process provided paths
            for path_str in self.params.paths:
                path = Path(path_str).resolve()

                if not path.exists():
                    self.processor.log_print(f"Warning: Path does not exist: {path}")
                    continue

                if not path.is_dir():
                    self.processor.log_print(f"Warning: Path is not a directory: {path}")
                    continue

                # Check if this folder directly contains /images subdirectory
                images_dir = path / "images"
                if images_dir.exists() and images_dir.is_dir():
                    # This folder has /images subdirectory, process it
                    folders_to_process.append(path)
                    self.processor.log_print(f"Found folder with /images: {path}")
                else:
                    # This might be a parent folder containing multiple folders with /images
                    subfolders_with_images = []
                    for subfolder in path.iterdir():
                        if subfolder.is_dir():
                            subfolder_images = subfolder / "images"
                            if subfolder_images.exists() and subfolder_images.is_dir():
                                subfolders_with_images.append(subfolder)

                    if subfolders_with_images:
                        folders_to_process.extend(subfolders_with_images)
                        self.processor.log_print(f"Found {len(subfolders_with_images)} subfolders with /images in: {path}")
                        for subfolder in subfolders_with_images:
                            self.processor.log_print(f"  - {subfolder}")
                    else:
                        self.processor.log_print(f"Warning: No /images subdirectories found in: {path}")
        else:
            # No paths provided, find all folders with /images in current directory
            current_dir = Path.cwd()
            for item in current_dir.iterdir():
                if item.is_dir():
                    images_dir = item / "images"
                    if images_dir.exists() and images_dir.is_dir():
                        folders_to_process.append(item)

            if folders_to_process:
                self.processor.log_print(f"Found {len(folders_to_process)} folders with /images in current directory")
            else:
                self.processor.log_print("No folders with /images subdirectories found in current directory")

        return folders_to_process

    def process_all(self) -> None:
        """Process folders based on paths argument or smart folder detection"""
        valid_folders = self._get_folders_to_process()

        if not valid_folders:
            self.processor.log_print("No valid folders with pre-converted images found")
            return

        self.processor.log_print(f"Found {len(valid_folders)} folders to process")
        self.processor.log_print(f"Backups will be stored in individual {self.BACKUP_FOLDER} folders")

        processed = 0
        failed = 0

        for folder in valid_folders:
            if self.process_folder(folder):
                processed += 1
            else:
                failed += 1

        self.processor.log_print("\nProcessing Summary:")
        self.processor.log_print(f"Successfully processed: {processed}")
        self.processor.log_print(f"Failed: {failed}")
        self.processor.log_print(f"Backups stored in respective {self.BACKUP_FOLDER} folders")

def parse_arguments() -> ExtendedProcessingParameters:
    """Parse arguments for image_process using unified CLI parser (without --reprocess)"""
    return parse_image_process_arguments()

def main():
    # Parse arguments
    params = parse_arguments()

    # Initialize processor
    processor = PreConvertedProcessor(params)

    # Setup logging
    processor.processor.setup_logging("process_converted.log", "autoprocess_logs")

    # Print banner and configuration
    processor.print_image_banner()
    processor.processor.log_print("\nProcessing with parameters:")
    processor.processor.log_print(f"Microscope: {params.microscope_config}")
    processor.processor.log_print(f"Output folder: {processor.OUTPUT_FOLDER}")
    processor.processor.log_print(f"Rotation Axis: {params.rotation_axis}")
    processor.processor.log_print(f"Frame Size: {params.frame_size}")
    processor.processor.log_print(f"Image format: {'SMV (.img)' if params.smv else 'TIF (.tif)'}")
    processor.processor.log_print(f"Signal Pixel: {params.signal_pixel}")
    processor.processor.log_print(f"Min Pixel: {params.min_pixel}")
    processor.processor.log_print(f"Background Pixel: {params.background_pixel}")
    processor.processor.log_print(f"Pixel Size: {params.pixel_size}")
    processor.processor.log_print(f"Wavelength: {params.wavelength}")
    processor.processor.log_print(f"Beam Center X: {params.beam_center_x}")
    processor.processor.log_print(f"Beam Center Y: {params.beam_center_y}")
    processor.processor.log_print(f"XDS Background Pixel: {params.background_pixel}")
    processor.processor.log_print(f"XDS Signal Pixel: {params.signal_pixel}")
    processor.processor.log_print(f"XDS Min Spot Pixels: {params.min_pixel}")
    # Log parameter source info
    if params.detector_distance:
        processor.processor.log_print(f"Detector Distance: {params.detector_distance} (override)")
    else:
        processor.processor.log_print(f"Detector Distance: will parse from source file or use default 960")

    if params.exposure:
        processor.processor.log_print(f"Exposure Time: {params.exposure} (override)")
    else:
        processor.processor.log_print(f"Exposure Time: will parse from source file or use default 3.0")

    if params.rotation:
        processor.processor.log_print(f"Rotation Rate: {params.rotation} (override)")
    else:
        processor.processor.log_print(f"Rotation Rate: will parse from source file or use default 0.3")
    if params.trim_front > 0 or params.trim_end > 0:
        processor.processor.log_print(
            f"Frame trimming: {params.trim_front} frames from start, {params.trim_end} frames from end"
        )

    # Log quality analysis status
    if params.quality_analysis:
        processor.processor.log_print("Diffraction quality analysis: ENABLED")
    else:
        processor.processor.log_print("Diffraction quality analysis: DISABLED")

    processor.processor.log_print("")

    processor.process_all()

if __name__ == "__main__":
    main()
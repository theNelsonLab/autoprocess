"""
Crystallography data processing script
Originally by Jessica Burch, modified by Dmitry Eremin
Refactored version with modular architecture
"""
import os
import random
import re
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from subprocess import run
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import from new modular structure
from .core.file_handler import FileHandler
from .core.xds_manager import XDSManager
from .core.process_tracker import ProcessTracker
from .config.parameters import ProcessingParameters
from .ui.display_manager import DisplayManager
from .ui.cli_parser import parse_autoprocess_arguments
from .quality_analyzer import DiffractionQualityAnalyzer


@dataclass
class LatticeCandidate:
    """Represents a lattice candidate from CORRECT.LP starred table"""
    line_order: int
    bravais_lattice: str
    quality_of_fit: float
    unit_cell: List[float]  # [a, b, c, alpha, beta, gamma]
    reindexing_matrix: List[int]  # 12 values


@dataclass
class SpaceGroupAnalysis:
    """Results of space group analysis for a dataset"""
    current_space_group: int
    current_bravais: str
    lattice_candidates: Dict[int, LatticeCandidate]
    best_candidate: LatticeCandidate
    needs_testing: bool
    test_space_group: int
    archive_directory: str


class CrystallographyProcessor:
    PROCESSED_FILES_LOG = "autoprocess_tracking.log"

    def __init__(self, params: ProcessingParameters):
        self.params = params
        self.current_path = Path.cwd()

        # Initialize display manager
        self.display = DisplayManager()

        # Initialize handlers with log_print delegation
        self.file_handler = FileHandler(params, self.display.log_print)
        self.xds_manager = XDSManager(params, self.display.log_print)
        self.process_tracker = ProcessTracker(self.display.log_print)

        # Load bravais lattice data for space group analysis
        self.bravais_data = self._load_bravais_data()

        # Flag to disable space group optimization (used in batch processing)
        self.disable_space_group_optimization = False

        # Flag to track if space group was corrected (need to read from CORRECT.LP)
        self._space_group_corrected = False

    def _get_processed_files_log_path(self) -> Path:
        """Get path to the processed files tracking log - delegates to ProcessTracker"""
        return self.process_tracker.get_processed_files_log_path()

    def _is_file_already_processed(self, file_path: Path) -> Optional[str]:
        """Check if file already processed - delegates to ProcessTracker"""
        return self.process_tracker.is_file_already_processed(file_path)

    def _add_to_processed_files_log(self, file_path: Path, output_folder: Path) -> None:
        """Add processed file entry - delegates to ProcessTracker"""
        self.process_tracker.add_to_processed_files_log(file_path, output_folder)
        self.display.log_print(f"Added to processing log: {file_path.name}")

    def _read_source_file_from_path(self, file_path: Path) -> Tuple[np.ndarray, bool]:
        """Read source file (MRC, SER, TVIPS) from specific path - delegates to FileHandler"""
        return self.file_handler.read_source_file_from_path(file_path)

    def _convert_data_to_tif(self, data: np.ndarray, is_multiframe: bool,
                           sample_movie: str, filename: str, frame_range: Optional[Tuple[int, int]] = None,
                           images_dir: Optional[Path] = None) -> bool:
        """Convert data from memory to TIF files - delegates to FileHandler"""
        return self.file_handler.convert_data_to_tif(data, is_multiframe, sample_movie, filename, frame_range, images_dir)

    def create_xds_input(self, data_path: str, params: dict) -> str:
        """Generate XDS.INP content - delegates to XDSManager"""
        return self.xds_manager.create_xds_input(data_path, params)

    @contextmanager
    def _working_directory(self, path: Path):
        """Context manager for temporarily changing working directory"""
        original_dir = os.getcwd()
        try:
            os.chdir(str(path))
            yield path
        finally:
            os.chdir(original_dir)


    def log_print(self, message: str) -> None:
        """Helper function to log messages as plain text - delegates to DisplayManager"""
        self.display.log_print(message)

    def print_banner(self):
        """Display the program banner - delegates to DisplayManager"""
        self.display.print_banner()

    def print_sub_banner(self):
        """Print subbanner - delegates to DisplayManager"""
        self.display.print_sub_banner()

    def setup_logging(self, log_file: str, dir_name: str) -> None:
        """Configure logging - delegates to DisplayManager"""
        self.display.setup_logging(log_file, dir_name)

    def _setup_movie_directories(self, sample_movie: str, distance: str, source_file_path: Path) -> Optional[Path]:
        """Set up directory structure for movie processing without moving source files."""
        # Create directories in the same location as the source file
        source_dir = source_file_path.parent
        movie_path = source_dir / sample_movie

        if not movie_path.exists():
            movie_path.mkdir(exist_ok=True)

            # Copy associated .emi files if they exist in the same directory as source
            for emi_file in source_dir.glob(f"{sample_movie}_{distance}*.emi"):
                # Copy instead of move to preserve original structure
                shutil.copy2(str(emi_file), str(movie_path / emi_file.name))

            # Create images directory
            image_path = movie_path / "images"
            image_path.mkdir(exist_ok=True)

            # Create auto_process directory
            auto_process_path = movie_path / "auto_process"
            auto_process_path.mkdir(exist_ok=True)

            return movie_path
        return movie_path  # Return existing path

    def _read_source_file(self, filename: str) -> Tuple[np.ndarray, bool]:
        """Read source file (MRC or SER) from current path - backward compatibility wrapper"""
        file_path = self.current_path / filename
        return self._read_source_file_from_path(file_path)

    def _verify_tif_conversion(self, original_data: np.ndarray, tif_path: Path, file_extension: str) -> bool:
        """Verify TIF conversion - delegates to FileHandler"""
        return self.file_handler.verify_tif_conversion(original_data, tif_path, file_extension)

    def _process_movie_data(self, sample_movie: str, distance: str,
                        rotation: str, exposure: str, resolution_range: float,
                        test_resolution_range: float, filename: str,
                        source_file_path: Path) -> None:
        """Process the movie data after directories are set up."""
        try:
            # Get absolute paths for all processing directories
            source_dir = source_file_path.parent
            movie_path = source_dir / sample_movie
            image_dir = movie_path / "images"
            auto_process_dir = movie_path / "auto_process"

            # Initialize frame range variables
            start_frame = None
            end_frame = None
            background_start = 1
            background_end = 10

            # OPTIMIZED WORKFLOW: Read file once for both quality analysis and conversion
            self.log_print(f"\nStep 1: Reading source file: {filename}")

            # Read source file once from the correct location
            data, is_multiframe = self._read_source_file_from_path(source_file_path)
            file_extension = Path(filename).suffix.upper()
            self.log_print(f"Successfully read {file_extension} file into memory")

            if self.params.quality_analysis:
                total_frames = data.shape[0] if is_multiframe else 1
                self.log_print(f"\nStep 2: Processing {total_frames} frames for quality analysis")

                quality_analyzer = DiffractionQualityAnalyzer()
                quality_results = quality_analyzer.analyze_data(data, filename)

                if not quality_results:
                    self.log_print(f"Quality analysis failed for {filename}. Skipping processing.")
                    return

                # Get quality summary and log only quality distribution
                summary = quality_analyzer.get_quality_summary()
                self.log_print(f"Quality analysis complete:")
                for quality, count in summary['quality_distribution'].items():
                    percentage = (count / summary['total_frames']) * 100
                    self.log_print(f"  {quality}: {count} frames ({percentage:.1f}%)")

                # Save quality analysis to CSV in images directory
                # Extract just the sample name (part before first underscore) for consistent CSV naming
                full_filename = Path(filename).stem  # Remove .mrc extension
                base_filename = full_filename.split('_')[0]  # Get sample name only
                csv_path = quality_analyzer.save_quality_csv(image_dir, base_filename)
                if csv_path:
                    self.log_print(f"Quality analysis CSV saved")

                # Find optimal frame range
                start_frame, end_frame = quality_analyzer.find_good_frame_range()
                if start_frame is None or end_frame is None:
                    self.log_print("No good quality frames found. Skipping processing.")
                    return

                # Calculate background range (start + 10 frames, but at least frame 1)
                background_start = max(1, start_frame)
                background_end = min(start_frame + 9, end_frame)  # 10 frames starting from start_frame

                self.log_print(f"Selected frame range: {start_frame}-{end_frame}")
                self.log_print(f"Background range: {background_start}-{background_end}")
            else:
                self.log_print(f"\nStep 2: Quality analysis disabled, processing all frames")
                # Set default frame range when quality analysis is disabled
                start_frame = 1
                end_frame = data.shape[0] if is_multiframe else 1

            # STEP 3: Convert from memory to TIF (no second file read)
            self.log_print(f"\nStep 3: Converting data to TIF images")

            # Always convert all frames to allow manual curation later
            # DQA selection will be applied during XDS processing, not file conversion
            success = self._convert_data_to_tif(data, is_multiframe, sample_movie, filename, None, image_dir)

            if not success:
                self.log_print(f"Failed to convert {filename}. Skipping processing.")
                return

            # Count converted images using absolute path
            image_files = list(image_dir.glob("*.tif"))
            if not image_files:
                self.log_print(f"No converted images found in {image_dir}")
                return

            total_images = len(image_files)
            self.log_print(f"Found {total_images} converted images")

            # Set end_frame if not set by quality analysis
            if end_frame is None:
                end_frame = total_images

            # Validate that our selected range doesn't exceed available images
            if end_frame > total_images:
                self.log_print(f"Warning: Selected end frame {end_frame} exceeds available images {total_images}")
                end_frame = total_images

            if start_frame > total_images:
                self.log_print(f"Warning: Selected start frame {start_frame} exceeds available images {total_images}")
                start_frame = 1
                end_frame = total_images

            # Update background range if using full range
            if not self.params.quality_analysis:
                background_end = min(10, end_frame)
                self.log_print(f"Using full frame range: {start_frame}-{end_frame}")
                self.log_print(f"Background range: {background_start}-{background_end}")

            # STEP 3: Create XDS.INP with frame ranges in auto_process directory
            step3_msg = "quality-based frame selection" if self.params.quality_analysis else "standard frame processing"
            self.log_print(f"\nStep 3: Creating XDS.INP with {step3_msg}")
            params = {
                'distance': distance,
                'rotation': rotation,
                'exposure': exposure,
                'resolution_range': resolution_range,
                'test_resolution_range': test_resolution_range,
                'total_images': total_images,
                'data_start': start_frame,
                'data_end': end_frame,
                'background_start': background_start,
                'background_end': background_end,
                'background_pixel': self.params.background_pixel,
                'signal_pixel': self.params.signal_pixel,
                'min_pixel': self.params.min_pixel,
            }

            # Create relative path from auto_process to images directory
            data_path = os.path.join("..", "images", sample_movie)
            xds_content = self.create_xds_input(data_path, params)

            # Write XDS.INP in auto_process directory using absolute path
            xds_inp_path = auto_process_dir / "XDS.INP"
            with open(xds_inp_path, 'w') as xds_inp:
                xds_inp.write(xds_content)

            # STEP 4: Run XDS processing in auto_process directory
            self.log_print(f"\nStep 4: Processing {sample_movie} with XDS...\n")
            with self._working_directory(auto_process_dir):
                with open("XDS.LP", "w+") as xds_out:
                    self._run_xds_command(xds_out)
                self.process_check(sample_movie)

        except Exception as e:
            self.log_print(f"Error processing movie data: {str(e)}")

    def _get_crystal_parameters(self) -> Tuple[Optional[str], Optional[str]]:
        """Extract final space group and unit cell parameters.

        If space group has been corrected (via _space_group_corrected flag),
        extracts from CORRECT.LP since XDS.LP may contain stale values after JOB=CORRECT.
        Otherwise extracts from XDS.LP using regex.

        Returns:
            Tuple containing (space_group, unit_cell_string) or (None, None) if not found
        """
        if getattr(self, '_space_group_corrected', False):
            return self._get_crystal_parameters_from_correct_lp()
        else:
            return self._get_crystal_parameters_from_xds_lp()

    def _get_crystal_parameters_from_xds_lp(self) -> Tuple[Optional[str], Optional[str]]:
        """Extract crystal parameters from XDS.LP (original method)."""
        space_group = None
        unit_cell = None

        with open('XDS.LP', 'r') as f:
            lines = f.readlines()
            # Read file in reverse to find the last occurrence first
            for line in reversed(lines):
                if space_group is None and "SPACE_GROUP_NUMBER=" in line:
                    space_group = line.split()[1]
                elif unit_cell is None and "UNIT_CELL_CONSTANTS=" in line:
                    # Refine regex to match exactly six floating-point numbers after UNIT_CELL_CONSTANTS=
                    match = re.search(r'UNIT_CELL_CONSTANTS=\s*([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)', line)
                    if match:
                        # Join the six captured groups into a single string
                        unit_cell = " ".join(match.groups())

                # Break once we've found both parameters
                if space_group is not None and unit_cell is not None:
                    break

        return space_group, unit_cell

    def _get_crystal_parameters_from_correct_lp(self) -> Tuple[Optional[str], Optional[str]]:
        """Extract crystal parameters from xds_ap.LP after space group correction."""
        space_group = None
        unit_cell = None

        try:
            with open('xds_ap.LP', 'r') as f:
                content = f.read()

            # Find space group and unit cell from "THE DATA COLLECTION STATISTICS REPORTED BELOW ASSUMES:" section
            # This section contains the final values after CORRECT step
            stats_section = re.search(r'THE DATA COLLECTION STATISTICS REPORTED BELOW ASSUMES:.*?SPACE_GROUP_NUMBER=\s*(\d+).*?UNIT_CELL_CONSTANTS=\s*([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)', content, re.DOTALL)

            if stats_section:
                space_group = stats_section.group(1)
                # Join the six unit cell parameters
                unit_cell = " ".join(stats_section.groups()[1:7])
            else:
                # Fallback: try to find them separately
                sg_match = re.search(r'SPACE_GROUP_NUMBER=\s*(\d+)', content)
                if sg_match:
                    space_group = sg_match.group(1)

                uc_match = re.search(r'UNIT_CELL_CONSTANTS=\s*([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)', content)
                if uc_match:
                    unit_cell = " ".join(uc_match.groups())

        except FileNotFoundError:
            self.log_print("Warning: xds_ap.LP not found for parameter extraction")
            return None, None
        except Exception as e:
            self.log_print(f"Warning: Error reading xds_ap.LP: {e}")
            return None, None

        return space_group, unit_cell

    def _save_crystal_parameters(self, space_group: str, unit_cell: str) -> None:
        """Save crystal parameters to stats.LP file.

        Args:
            space_group: Space group number
            unit_cell: Unit cell parameters string
        """
        with open('stats.LP', 'w') as f:
            f.write(f"{space_group}\n{unit_cell}")

    def _print_crystal_parameters(self, space_group: str, unit_cell: str) -> None:
        """Print crystal parameters to log.

        Args:
            space_group: Space group number
            unit_cell: Unit cell parameters string
        """
        self.log_print(f"\nI found space group {space_group} and a unit cell of")
        self.log_print(f"{unit_cell}")
        self.log_print(f"\nProceeding with scaling")

    def parse_filename(self, filename: str) -> Optional[Tuple[str, str, str, str]]:
        """Parse the input filename to extract parameters."""
        if not filename.endswith(self.params.file_extension):
            return None

        split = filename.split("_")
        if len(split) < 4:
            self.log_print(f"Skipping {filename}: unexpected filename format.")
            return None

        sample_movie = split[0]
        # Use command line parameters if provided, otherwise use filename values
        distance = self.params.detector_distance or split[1]
        rotation = self.params.rotation or split[2]
        exposure = self.params.exposure or split[3]

        return (sample_movie, distance, rotation, exposure)  # Return as a tuple

    def calculate_resolution_ranges(self, distance: str) -> Optional[Tuple[float, float]]:
        """Calculate resolution ranges based on perpendicular distance from beam center to frame edges.

        Args:
            distance: Detector distance in mm

        Returns:
            Tuple of (resolution_range, test_resolution_range) or None if calculation fails
        """
        try:
            # Convert detector distance to float
            detector_distance = float(distance)

            # Calculate perpendicular distances from beam center to edges
            edge_distances = []

            # Distance to left/right edges
            dx_left = abs(0 - self.params.beam_center_x)
            dx_right = abs(self.params.frame_size - self.params.beam_center_x)
            edge_distances.append(min(dx_left, dx_right) * self.params.pixel_size)

            # Distance to top/bottom edges
            dy_top = abs(0 - self.params.beam_center_y)
            dy_bottom = abs(self.params.frame_size - self.params.beam_center_y)
            edge_distances.append(min(dy_top, dy_bottom) * self.params.pixel_size)

            # Find minimum perpendicular distance to any edge
            min_distance = min(edge_distances)

            # Calculate resolution using Bragg's law
            # resolution = wavelength / (2 * sin(theta))
            # where theta = arctan(radius / detector_distance) / 2
            wavelength = float(self.params.wavelength)
            theta = np.arctan(min_distance / detector_distance) / 2
            max_resolution = wavelength / (2 * np.sin(theta))

            # Set resolution range slightly inside the edge
            resolution_range = max_resolution * 0.9  # 10% buffer from edge
            test_resolution_range = max_resolution * 1.1  # Less demanding for testing

            if resolution_range <= 0 or test_resolution_range <= 0:
                self.log_print(f"Invalid resolution ranges calculated: {resolution_range}, {test_resolution_range}")
                return None

            return round(resolution_range, 2), round(test_resolution_range, 2)

        except Exception as e:
            self.log_print(f"Error calculating resolution ranges: {str(e)}")
            return None

    def _get_frame_ranges(self, params: dict) -> Tuple[str, str, str]:
        """Get frame ranges - delegates to XDSManager"""
        return self.xds_manager._get_frame_ranges(params)

    def process_check(self, sample_movie: str) -> Optional[bool]:
        """Check and handle processing status."""
        if not Path('XDS.INP').exists():
            return None

        if not Path('X-CORRECTIONS.cbf').exists():
            self._run_xds("XDS is running...")

        if not Path('XPARM.XDS').exists():
            return self._handle_missing_xparm(sample_movie)

        if not Path('DEFPIX.LP').exists():
            return self._handle_missing_defpix(sample_movie)

        if not Path("INTEGRATE.HKL").exists():
            return self._handle_missing_integrate(sample_movie)

        if Path("CORRECT.LP").exists():
            self.log_print("Successful indexing!\n")
            return self.mosaicity(sample_movie)

        return None

    # =====================================
    # Space Group Analysis Methods
    # =====================================

    def _load_bravais_data(self) -> Dict:
        """Load bravais lattice to space group mapping"""
        try:
            bravais_path = Path(__file__).parent / "data" / "bravais_lattices.json"
            with open(bravais_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.log_print(f"Warning: Could not load bravais_lattices.json: {e}")
            return {}

    def parse_current_space_group(self, content: str) -> Optional[int]:
        """Parse current space group from CORRECT.LP content"""
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "THE DATA COLLECTION STATISTICS REPORTED BELOW ASSUMES:" in line:
                # Check next line for SPACE_GROUP_NUMBER
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if "SPACE_GROUP_NUMBER=" in next_line:
                        # Extract space group number
                        match = re.search(r'SPACE_GROUP_NUMBER=\s*(\d+)', next_line)
                        if match:
                            return int(match.group(1))
        return None

    def parse_starred_lattice_candidates(self, content: str) -> Dict[int, LatticeCandidate]:
        """Parse starred lattice candidates from CORRECT.LP content"""
        candidates = {}
        lines = content.split('\n')

        # Find lattice table
        table_start = -1
        for i, line in enumerate(lines):
            if "LATTICE-" in line and "BRAVAIS-" in line and "QUALITY" in line:
                table_start = i + 2  # Skip header and character/lattice line
                break

        if table_start == -1:
            return candidates

        # Parse starred lines
        line_order = 0
        for i in range(table_start, len(lines)):
            line = lines[i].strip()
            if not line:
                continue

            if not line.startswith('*'):
                # Continue reading until we hit a non-starred line that's not empty
                if line and not line.startswith(' '):
                    break
                continue

            try:
                # Parse starred line format: *  44        aP          0.0       5.2   10.4   21.0  90.3  90.5  90.0   -1  0  0...
                # Split by whitespace and filter out empty strings
                parts = [part for part in line.split() if part]

                if len(parts) < 19:  # Need: *, number, lattice, quality, 6 unit cell, 12 reindexing
                    continue

                line_order += 1
                # parts[0] = '*', parts[1] = number, parts[2] = lattice type, parts[3] = quality
                bravais_lattice = parts[2]  # aP, mP, etc.
                quality_of_fit = float(parts[3])

                # Unit cell parameters: a, b, c, alpha, beta, gamma (parts[4] through parts[9])
                unit_cell = [float(parts[i]) for i in range(4, 10)]

                # Reindexing matrix: 12 values (parts[10] through parts[21])
                reindexing_matrix = [int(parts[i]) for i in range(10, 22)]

                candidate = LatticeCandidate(
                    line_order=line_order,
                    bravais_lattice=bravais_lattice,
                    quality_of_fit=quality_of_fit,
                    unit_cell=unit_cell,
                    reindexing_matrix=reindexing_matrix
                )

                candidates[line_order] = candidate

            except (ValueError, IndexError):
                # Skip malformed lines
                continue

        return candidates

    def get_bravais_for_space_group(self, space_group: int) -> Optional[str]:
        """Get Bravais lattice type for given space group number"""
        if "crystal_systems" not in self.bravais_data:
            # Force reload the data if it wasn't loaded during initialization
            try:
                import json
                bravais_path = Path(__file__).parent / "data" / "bravais_lattices.json"
                if bravais_path.exists():
                    with open(bravais_path, 'r') as f:
                        self.bravais_data = json.load(f)
            except Exception:
                pass

            if "crystal_systems" not in self.bravais_data:
                return None

        for crystal_system, cs_data in self.bravais_data["crystal_systems"].items():
            if "bravais_types" in cs_data:
                for bravais_type, bt_data in cs_data["bravais_types"].items():
                    if "space_groups" in bt_data and space_group in bt_data["space_groups"]:
                        return bt_data["pearson_symbol"]  # Return pearson symbol like "mP", "oP"
        return None

    def get_minimal_space_group_for_bravais(self, bravais: str) -> Optional[int]:
        """Get minimal (first) space group number for given Bravais lattice"""
        if "crystal_systems" not in self.bravais_data:
            # Force reload the data if it wasn't loaded during initialization
            try:
                import json
                bravais_path = Path(__file__).parent / "data" / "bravais_lattices.json"
                if bravais_path.exists():
                    with open(bravais_path, 'r') as f:
                        self.bravais_data = json.load(f)
            except Exception:
                pass

            if "crystal_systems" not in self.bravais_data:
                return None

        for crystal_system, cs_data in self.bravais_data["crystal_systems"].items():
            if "bravais_types" in cs_data:
                for bravais_type, bt_data in cs_data["bravais_types"].items():
                    if bt_data.get("pearson_symbol") == bravais:
                        if "space_groups" in bt_data:
                            return min(bt_data["space_groups"])
        return None

    def analyze_space_group_optimality(self, current_sg: int, candidates: Dict[int, LatticeCandidate]) -> SpaceGroupAnalysis:
        """Compare current space group against best quality lattice"""
        if not candidates:
            # No alternatives found, keep current
            current_bravais = self.get_bravais_for_space_group(current_sg)
            return SpaceGroupAnalysis(
                current_space_group=current_sg,
                current_bravais=current_bravais or "unknown",
                lattice_candidates=candidates,
                best_candidate=None,
                needs_testing=False,
                test_space_group=current_sg,
                archive_directory=""
            )

        # Find best quality candidate (last starred line = highest line order)
        best_candidate = max(candidates.values(), key=lambda x: x.line_order)

        # Get current space group's Bravais lattice
        current_bravais = self.get_bravais_for_space_group(current_sg)
        if not current_bravais:
            current_bravais = "unknown"

        # Compare Bravais lattices
        if current_bravais == best_candidate.bravais_lattice:
            # Same Bravais lattice, keep current
            return SpaceGroupAnalysis(
                current_space_group=current_sg,
                current_bravais=current_bravais,
                lattice_candidates=candidates,
                best_candidate=best_candidate,
                needs_testing=False,
                test_space_group=current_sg,
                archive_directory=""
            )
        else:
            # Different Bravais lattice is better, test it
            test_sg = self.get_minimal_space_group_for_bravais(best_candidate.bravais_lattice)
            if test_sg:
                archive_name = f"{current_bravais}-{current_sg}"
                return SpaceGroupAnalysis(
                    current_space_group=current_sg,
                    current_bravais=current_bravais,
                    lattice_candidates=candidates,
                    best_candidate=best_candidate,
                    needs_testing=True,
                    test_space_group=test_sg,
                    archive_directory=archive_name
                )
            else:
                # Cannot find test space group, keep current
                return SpaceGroupAnalysis(
                    current_space_group=current_sg,
                    current_bravais=current_bravais,
                    lattice_candidates=candidates,
                    best_candidate=best_candidate,
                    needs_testing=False,
                    test_space_group=current_sg,
                    archive_directory=""
                )

    def check_space_group_before_scaling(self, sample_movie: str) -> bool:
        """Main entry point for space group validation before scaling"""
        try:
            # Skip space group optimization if disabled (e.g., in batch processing mode)
            if self.disable_space_group_optimization:
                self.log_print("Best space group has been identified")
                return True

            # Check if CORRECT.LP exists
            if not Path("CORRECT.LP").exists():
                self.log_print("Warning: CORRECT.LP not found, proceeding with current space group")
                return True

            # Read and parse CORRECT.LP
            with open("CORRECT.LP", 'r') as f:
                content = f.read()

            # Parse current space group
            current_sg = self.parse_current_space_group(content)
            if current_sg is None:
                self.log_print("Warning: Could not parse current space group, proceeding with original")
                return True

            # Parse lattice candidates
            candidates = self.parse_starred_lattice_candidates(content)

            # Analyze space group optimality
            analysis = self.analyze_space_group_optimality(current_sg, candidates)

            if not analysis.needs_testing:
                self.log_print("Best space group has been identified")
                return True

            # Log decision to test alternative
            self.log_print(f"Assumed {analysis.current_bravais} space group {analysis.current_space_group} might not be optimal")
            self.log_print(f"Testing {analysis.best_candidate.bravais_lattice} space group {analysis.test_space_group} before scaling")

            # Test alternative space group
            success = self.archive_current_processing_and_run_correct(analysis)
            if success:
                self.log_print("Space group corrected")
            else:
                self.log_print("Space group evaluation failed, proceeding with original space group option")

            return True

        except Exception as e:
            self.log_print(f"Warning: Space group analysis failed ({e}), proceeding with original")
            return True

    def copy_auto_process_for_archive(self, archive_name: str) -> str:
        """Copy auto_process folder to processing_backups with current space group name"""
        current_dir = Path.cwd()
        parent_dir = current_dir.parent

        # Create processing_backups directory if it doesn't exist
        backups_dir = parent_dir / "processing_backups"
        backups_dir.mkdir(exist_ok=True)

        # Create archive directory name inside processing_backups
        base_archive_path = backups_dir / archive_name

        # Handle conflicts by adding a counter if directory exists
        archive_path = base_archive_path
        counter = 1
        while archive_path.exists():
            archive_path = backups_dir / f"{archive_name}-{counter}"
            counter += 1
            if counter > 100:  # Safety limit
                break

        try:
            # Copy current auto_process directory to archive location
            shutil.copytree(current_dir, archive_path)
            # Show relative path from parent for cleaner logging
            relative_path = archive_path.relative_to(parent_dir)
            self.log_print(f"Archived current processing to {relative_path}")
            return str(archive_path)
        except Exception as e:
            raise Exception(f"Failed to archive current processing: {e}")

    def modify_xds_inp_for_correct_only(self, sg_number: int, unit_cell: List[float], reidx: List[int]) -> None:
        """Modify XDS.INP in auto_process directory for CORRECT-only run"""
        xds_inp_path = Path("XDS.INP")

        if not xds_inp_path.exists():
            raise Exception("XDS.INP not found")

        # Read current XDS.INP
        with open(xds_inp_path, 'r') as f:
            lines = f.readlines()

        # Modify lines
        modified_lines = []
        for line in lines:
            stripped = line.strip()

            # Handle JOB lines - comment out all active JOB lines
            if stripped.startswith("JOB=") and not stripped.startswith("!"):
                modified_lines.append(f"!{line}")
            # Activate JOB= CORRECT line
            elif stripped == "!JOB= CORRECT":
                modified_lines.append("JOB= CORRECT\n")
            # Handle space group parameters
            elif stripped.startswith("!SPACE_GROUP_NUMBER="):
                modified_lines.append(f"SPACE_GROUP_NUMBER= {sg_number}\n")
            elif stripped.startswith("!UNIT_CELL_CONSTANTS="):
                unit_cell_str = " ".join(f"{val:.3f}" for val in unit_cell)
                modified_lines.append(f"UNIT_CELL_CONSTANTS= {unit_cell_str}\n")
            elif stripped.startswith("!REIDX="):
                reidx_str = " ".join(str(val) for val in reidx)
                modified_lines.append(f"REIDX= {reidx_str}\n")
            else:
                modified_lines.append(line)

        # Write modified XDS.INP
        with open(xds_inp_path, 'w') as f:
            f.writelines(modified_lines)

    def archive_current_processing_and_run_correct(self, analysis: SpaceGroupAnalysis) -> bool:
        """Archive current auto_process, modify XDS.INP, and run CORRECT"""
        archive_path = None

        try:
            # Step 1: Archive current processing
            archive_path = self.copy_auto_process_for_archive(analysis.archive_directory)

            # Step 2: Modify XDS.INP for CORRECT-only run
            self.modify_xds_inp_for_correct_only(
                analysis.test_space_group,
                analysis.best_candidate.unit_cell,
                analysis.best_candidate.reindexing_matrix
            )

            # Step 3: Verify prerequisites for XDS CORRECT
            if not Path("INTEGRATE.HKL").exists():
                raise Exception("INTEGRATE.HKL not found - cannot run CORRECT")

            # Get timestamp of existing CORRECT.LP for comparison
            old_correct_time = Path("CORRECT.LP").stat().st_mtime if Path("CORRECT.LP").exists() else 0

            # Step 4: Run XDS CORRECT
            self.log_print("Running XDS CORRECT with new space group parameters...")

            # Ensure XDS.INP is flushed to disk
            import time
            time.sleep(0.1)

            # Verify XDS.INP has correct content before running
            if Path("XDS.INP").exists():
                with open("XDS.INP", 'r') as f:
                    inp_content = f.read()
                    if f"SPACE_GROUP_NUMBER= {analysis.test_space_group}" not in inp_content:
                        raise Exception("XDS.INP was not properly modified")
                    if "JOB= CORRECT" not in inp_content:
                        raise Exception("XDS.INP JOB line not set to CORRECT")
            else:
                raise Exception("XDS.INP not found")

            try:
                # Use the same method as the existing _run_xds_command
                with open("xds_ap.LP", "w") as output_file:
                    self._run_xds_command(output_file)

            except Exception as xds_error:
                raise Exception(f"XDS execution failed: {xds_error}")

            # Step 5: Verify CORRECT succeeded by checking xds_ap.LP output
            if Path("xds_ap.LP").exists():
                with open("xds_ap.LP", 'r') as f:
                    xds_output = f.read()

                # Parse space group from the XDS terminal output in xds_ap.LP
                # Look for "THE DATA COLLECTION STATISTICS REPORTED BELOW ASSUMES:" section
                sg_pattern = rf'THE DATA COLLECTION STATISTICS REPORTED BELOW ASSUMES:.*?SPACE_GROUP_NUMBER=\s*{analysis.test_space_group}\s'
                if re.search(sg_pattern, xds_output, re.DOTALL):
                    # Mark that we need to extract parameters from the XDS output instead of XDS.LP
                    self._space_group_corrected = True
                    return True
                else:
                    # Debug: show what space group was actually found
                    sg_match = re.search(r'THE DATA COLLECTION STATISTICS REPORTED BELOW ASSUMES:.*?SPACE_GROUP_NUMBER=\s*(\d+)', xds_output, re.DOTALL)
                    found_sg = sg_match.group(1) if sg_match else "none"
                    raise Exception(f"XDS CORRECT completed but wrong space group found: {found_sg}, expected: {analysis.test_space_group}")
            else:
                raise Exception("xds_ap.LP not found after XDS run")

        except Exception as e:
            self.log_print(f"Space group testing failed: {e}")

            # Revert by renaming directories
            try:
                current_dir = Path.cwd()
                parent_dir = current_dir.parent
                failed_dir = parent_dir / f"{analysis.best_candidate.bravais_lattice}-{analysis.test_space_group}-FAILED"

                # Rename current auto_process to FAILED
                if current_dir.exists():
                    current_dir.rename(failed_dir)

                # Rename archive back to auto_process
                if archive_path and Path(archive_path).exists():
                    Path(archive_path).rename(current_dir)
                    os.chdir(str(current_dir))  # Change back to working directory

                self.log_print(f"Reverted to original processing in {current_dir}")

            except Exception as revert_error:
                self.log_print(f"Warning: Failed to revert directories: {revert_error}")

            return False

    def _run_xds_command(self, output_file) -> None:
        """Run XDS command, using parallel version if specified."""
        command = "xds_par" if self.params.parallel else "xds"
        run(command, stdout=output_file)

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
            self.log_print(f"{prefix}XDS Processing Parameters:")
            self.log_print(f"  BACKGROUND_PIXEL= {background_val}")
            self.log_print(f"  SIGNAL_PIXEL= {signal_val}")
            self.log_print(f"  MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT= {min_spot_val}")

        except Exception as e:
            self.log_print(f"Warning: Could not log XDS parameters: {str(e)}")

    def _run_xds(self, message: str) -> None:
        """Run XDS with logging."""
        self.log_print(message)
        with open('XDS.LP', "w+") as xds_out:
            self._run_xds_command(xds_out)

    def _handle_missing_xparm(self, sample_movie: str) -> Optional[bool]:
        """Handle missing XPARM.XDS file."""
        for _ in range(10):  # Try 10 times
            # Generate new random parameters
            with open('XDS.INP', 'r+') as f:
                lines = f.readlines()

                bkgrnd_pix = random.randrange(3, 5, 1)
                s_pix = random.randrange(4, 9, 1)
                min_pix = random.randrange(5, 9, 1)

                for index, line in enumerate(lines):
                    if "BACKGROUND_PIXEL=" in line:
                        lines[index] = f"BACKGROUND_PIXEL= {bkgrnd_pix}\n"
                    if "SIGNAL_PIXEL=" in line:
                        lines[index] = f"SIGNAL_PIXEL= {s_pix}\n"
                    if "MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT=" in line:
                        lines[index] = f"MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT= {min_pix}\n"

                f.seek(0)
                f.writelines(lines)
                f.truncate()

            self.log_print(f"Screening new indexing values (attempt {_ + 1}/10):")
            self.log_print("Updated XDS processing parameters:")
            self.log_print(f"  XDS Background Pixel: {bkgrnd_pix}")
            self.log_print(f"  XDS Signal Pixel: {s_pix}")
            self.log_print(f"  XDS Min Spot Pixels: {min_pix}")
            self._run_xds("XDS is running...")

            if Path('XPARM.XDS').exists():
                self.log_print("Successful indexing with new parameters!")
                return self.process_check(sample_movie)

        self.log_print(f"Unable to autoprocess {sample_movie}!")
        return None

    def _handle_missing_defpix(self, sample_movie: str) -> Optional[bool]:
        """Handle missing DEFPIX.LP file."""
        self._modify_xds_job()
        self.log_print("Less than 50% of spots went through:")
        self.log_print("Running with JOB= DEFPIX INTEGRATE CORRECT...")
        self._run_xds("XDS is running...")

        # Add check here
        if not Path('XPARM.XDS').exists():
            self.log_print(f"Unable to autoprocess {sample_movie}!")
            return None
        return self.process_check(sample_movie)

    def _handle_missing_integrate(self, sample_movie: str) -> Optional[bool]:
        """Handle missing INTEGRATE.HKL file."""
        # Add beam divergence parameters
        with open('XDS.INP', 'r+') as f:
            content = f.read()
            f.seek(0)
            f.write(content)
            f.write("""BEAM_DIVERGENCE= 0.03 BEAM_DIVERGENCE_E.S.D.= 0.003
    REFLECTING_RANGE=1.0 REFLECTING_RANGE_E.S.D.= 0.2""")

        self.log_print("Adding beam divergence values to correct a common error.")
        self._run_xds("XDS is running...")

        if not Path("INTEGRATE.HKL").exists():
            self.log_print(f"Unable to autoprocess {sample_movie}!")
            return None

        return self.process_check(sample_movie)

    def _modify_xds_job(self) -> None:
        """Modify XDS job parameters."""
        with open('XDS.INP', 'r+') as f:
            lines = f.readlines()
            f.seek(0)
            f.write("!JOB=XYCORR INIT COLSPOT IDXREF DEFPIX INTEGRATE CORRECT\n")
            f.write("JOB=DEFPIX INTEGRATE CORRECT\n")
            f.writelines(lines[2:])

    def mosaicity(self, sample_movie: str) -> Optional[bool]:
        """Process mosaicity parameters."""
        self._modify_xds_job()

        beam_divergence = None
        reflecting_range = None

        # Read parameters from INTEGRATE.LP
        with open('INTEGRATE.LP', 'r') as f:
            for line in f:
                if "BEAM_DIVERGENCE=" in line:
                    beam_divergence = line.strip()
                if "REFLECTING_RANGE=" in line:
                    reflecting_range = line.strip()

        # Update XDS.INP with new parameters
        with open('XDS.INP', 'r+') as f:
            lines = f.readlines()

            beam_divergence_found = False
            reflecting_range_found = False

            for index, line in enumerate(lines):
                if "BEAM_DIVERGENCE=" in line:
                    lines[index] = f"{beam_divergence}\n"
                    beam_divergence_found = True
                if "REFLECTING_RANGE=" in line:
                    lines[index] = f"{reflecting_range}\n"
                    reflecting_range_found = True

            if not beam_divergence_found and beam_divergence:
                lines.append(f"{beam_divergence}\n")
            if not reflecting_range_found and reflecting_range:
                lines.append(f"{reflecting_range}\n")

            with open('XDS.INP', 'w') as f_write:
                f_write.writelines(lines)

        return self.iterate_opt(sample_movie)

    def iterate_opt(self, sample_movie: str, previous_isa: float = None) -> Optional[bool]:
        """Optimize processing parameters."""
        # Get first ISa value
        Isa1 = None
        with open('XDS.LP', 'r+') as f:
            lines = f.readlines()
            f.seek(0)
            f.writelines(lines[-26:])

        for line in lines:
            if ["a", "b", "ISa"] == line.split():
                next_line = lines[lines.index(line) + 1]
                stats = next_line.split()
                Isa1 = float(stats[2])
                if previous_isa is not None and abs(Isa1 - previous_isa) < 0.01:
                    self.log_print(f"Same ISa value of {Isa1} was obtained")
                else:
                    self.log_print(f"ISa: {Isa1}. Testing new values now.")
                break

        # Run XDS again
        self._run_xds("XDS is running...")

        # Get second set of ISa values
        Isa2_values = []
        with open('XDS.LP', 'r+') as f:
            lines = f.readlines()
            f.seek(0)
            f.writelines(lines[-26:])

            for line in lines:
                if ["a", "b", "ISa"] == line.split():
                    new_next_line = lines[lines.index(line) + 1]
                    new_stats = new_next_line.split()
                    Isa2_values.append(float(new_stats[2]))

        # Check if we found any Isa2 values
        if not Isa2_values:
            self.log_print(f"Unable to process {sample_movie} - No ISa values found in second XDS run")
            return None

        # Check if all Isa2 values are the same
        if len(set(Isa2_values)) > 1:
            self.log_print(f"Unable to process {sample_movie} - Inconsistent ISa values: {Isa2_values}")
            return None

        # Use the Isa2 value (they're all the same if we got here)
        Isa2 = Isa2_values[0]

        # Check if we have valid ISa values before doing math
        if Isa1 is None:
            self.log_print(f"Warning: Could not parse initial ISa value, using Isa2: {Isa2}")
            Isa1 = Isa2  # Set to same value to avoid math errors

        if abs(Isa2 - Isa1) < 0.01:
            self.log_print(f"Same ISa value of {Isa2} was obtained")
        else:
            self.log_print(f"ISa: {Isa2}")

        if abs(Isa2 - Isa1) > 0.5:
            self.log_print("I'm trying to optimize beam divergence values.")
            return self.iterate_opt(sample_movie, Isa2)  # Pass current ISa as previous_isa
        else:
            self.log_print("Optimized beam divergence values.")

            # Check space group before extracting crystal parameters
            if not self.check_space_group_before_scaling(sample_movie):
                return None

            # Get crystal parameters using the dedicated method (after potential space group correction)
            space_group, unit_cell = self._get_crystal_parameters()

            if space_group and unit_cell:
                self._save_crystal_parameters(space_group, unit_cell)
                self._print_crystal_parameters(space_group, unit_cell)
                return self.scale_conv(sample_movie)
            else:
                self.log_print(f"Space group or unit cell not found. Cannot finish autoprocess for {sample_movie}.")
                return None

    def scale_conv(self, sample_movie: str) -> bool:
        """Perform scaling and conversion."""
        if not Path("CORRECT.LP").exists():
            return False

        # XSCALE processing
        xscale_content = f"""OUTPUT_FILE= {sample_movie}.ahkl
INPUT_FILE= XDS_ASCII.HKL
RESOLUTION_SHELLS= 10 8 5 3 2.3 2.0 1.7 1.5 1.3 1.2 1.1 1.0 0.90 0.80
"""
        with open('XSCALE.INP', 'w') as xscale:
            xscale.write(xscale_content)

        with open("xscale_ap.LP", "w+") as xscale_out:
            run("xscale", stdout=xscale_out)
        self.log_print("I scaled the data in XSCALE.")

        # XDSCONV processing
        xdsconv_content = f"""INPUT_FILE= {sample_movie}.ahkl
OUTPUT_FILE= {sample_movie}.hkl SHELX
GENERATE_FRACTION_OF_TEST_REFLECTIONS=0.10
FRIEDEL'S_LAW=FALSE
"""
        with open('XDSCONV.INP', 'w') as xdsconv:
            xdsconv.write(xdsconv_content)

        return self.check_space_group(sample_movie)

    def check_space_group(self, sample_movie: str) -> bool:
        """Check space group using CCP4's pointless (only if --pointless flag is set)."""
        with open("xdsconv_ap.LP", "w+") as xdsconv_out:
            run("xdsconv", stdout=xdsconv_out)
        self.log_print("I converted it for use in shelx!")

        # Only run pointless if the flag is set
        if self.params.pointless:
            # Run pointless
            result = run("pointless XDS_ASCII.HKL > pointless.LP",
                shell=True, capture_output=True)

            # Check if pointless ran successfully
            if result.returncode != 0:
                self.log_print("Warning: Could not run pointless, but processing completed")
                return False  # Stop further processing

            self._process_pointless_output()
        else:
            self.log_print("Pointless analysis skipped (--pointless flag not set)")

        return True

    def _process_pointless_output(self) -> None:
        """Process the output from pointless."""
        with open('pointless.LP', 'r') as p1:
            content = p1.read()

        # Find the spacegroup section and parse each line
        pattern = r'\s*(\S+(?:\s+\S+)*?)\s*\(\s*(\d+)\s*\)\s+(\d+\.\d+)\s+(\d+\.\d+)'

        with open("pointless_group.LP", 'w') as pg:
            for line in content.split('\n'):
                match = re.match(pattern, line.strip())
                if match:
                    name, number, tot_prob, sys_prob = match.groups()
                    # Write all info to file
                    pg.write(f"{name},{number},{tot_prob},{sys_prob}\n")
                    # Print to log as before
                    self.log_print(f"Possible space group: {number} - {name}")

    def process_movie(self) -> None:
        files_to_process = self._get_files_to_process()

        if not files_to_process:
            self.log_print("No .mrc or .ser files found to process.")
            return

        processed_movie = False

        for file_path in files_to_process:
            filename = os.path.basename(file_path)
            file_path_obj = Path(file_path)

            # Check if file has already been processed
            if not self.params.reprocess:
                existing_output = self._is_file_already_processed(file_path_obj)
                if existing_output:
                    self.log_print(f"Already processed {filename} (output in: {existing_output})")
                    continue

            file_info = self.parse_filename(filename)
            if not file_info:
                self.log_print(f"Skipping {filename}: Could not parse filename")
                continue

            sample_movie, distance, rotation, exposure = file_info
            ranges = self.calculate_resolution_ranges(distance)
            if ranges is None:
                self.log_print(f"Skipping {filename}: Could not calculate resolution ranges")
                continue

            resolution_range, test_resolution_range = ranges

            # Process without changing directories - use absolute paths
            processed_movie = True
            self._process_single_movie(
                sample_movie, distance, rotation, exposure,
                resolution_range, test_resolution_range,
                filename, file_path_obj
            )

        if not processed_movie:
            self.log_print('Found no movies to process.')

    def _get_files_to_process(self) -> list:
        """Get list of files to process based on paths argument or current directory"""
        files_to_process = []

        if not self.params.paths:
            # No paths specified, use current directory (legacy behavior)
            files = os.listdir('.')
            for filename in files:
                if filename.endswith(('.mrc', '.ser')):
                    files_to_process.append(os.path.abspath(filename))
            return files_to_process

        # Process each specified path
        for path_str in self.params.paths:
            path = Path(path_str).resolve()

            if path.is_file():
                # Single file specified
                if path.suffix.lower() in ['.mrc', '.ser']:
                    files_to_process.append(str(path))
                    self.log_print(f"Added file: {path}")
                else:
                    self.log_print(f"Warning: {path} is not a .mrc or .ser file")

            elif path.is_dir():
                # Directory specified, find all .mrc/.ser files
                found_files = []
                for ext in ['*.mrc', '*.ser']:
                    found_files.extend(path.glob(ext))

                if found_files:
                    for file_path in found_files:
                        files_to_process.append(str(file_path))
                    self.log_print(f"Found {len(found_files)} files in {path}")
                else:
                    self.log_print(f"Warning: No .mrc or .ser files found in {path}")

            else:
                self.log_print(f"Warning: Path does not exist: {path}")

        self.log_print(f"Total files to process: {len(files_to_process)}")
        return files_to_process

    def _process_single_movie(self, sample_movie: str, distance: str,
                            rotation: str, exposure: str, resolution_range: float,
                            test_resolution_range: float, filename: str,
                            source_file_path: Path) -> None:
        """Process a single movie file."""
        # Check if output directory already exists (old behavior for directory structure)
        if Path(sample_movie).exists():
            self.log_print(f"Output directory {sample_movie} already exists - reprocessing")

        # Log processing parameters for this movie
        self.log_print(f"\nProcessing parameters for {filename}:")
        self.log_print(f"Detector Distance: {distance} mm")
        self.log_print(f"Exposure Time: {exposure} s")
        self.log_print(f"Rotation Rate: {rotation} deg/s")
        self.log_print(f"Resolution Range: {resolution_range} Å")
        self.log_print(f"Test Resolution Range: {test_resolution_range} Å\n")

        movie_dir = self._setup_movie_directories(
            sample_movie, distance, source_file_path
        )
        if not movie_dir:
            return

        self._process_movie_data(
            sample_movie, distance, rotation, exposure,
            resolution_range, test_resolution_range,
            filename, source_file_path
        )

        # Add file to processed tracking log
        # Use absolute path for output folder based on source file location
        source_dir = source_file_path.parent
        output_folder = source_dir / sample_movie
        self._add_to_processed_files_log(source_file_path, output_folder)


def main():
    # Parse arguments and create ProcessingParameters instance
    params = parse_autoprocess_arguments()

    # Initialize processor with parsed parameters
    processor = CrystallographyProcessor(params)

    # Setup logging and start processing
    processor.setup_logging(log_file="autoprocess.log", dir_name="autoprocess_logs")
    processor.display.print_full_banner()

    # print command line arguments
    processor.log_print(' '.join(sys.argv))

    # Log the current parameters being used
    processor.log_print("\nUsing processing parameters:")
    processor.log_print(f"Microscope: {params.microscope_config}")
    processor.log_print(f"Rotation Axis: {params.rotation_axis}")
    processor.log_print(f"Frame Size: {params.frame_size}")
    processor.log_print(f"File Extension: {params.file_extension}")
    processor.log_print(f"Signal Pixel: {params.signal_pixel}")
    processor.log_print(f"Min Pixel: {params.min_pixel}")
    processor.log_print(f"Background Pixel: {params.background_pixel}")
    processor.log_print(f"Pixel Size: {params.pixel_size}")
    processor.log_print(f"Wavelength: {params.wavelength} (fixed)")
    processor.log_print(f"Beam Center X: {params.beam_center_x}")
    processor.log_print(f"Beam Center Y: {params.beam_center_y}")
    processor.log_print(f"Pointless Analysis: {'Enabled' if params.pointless else 'Disabled'}")
    processor.log_print(f"Parallel XDS: {'Enabled (xds_par)' if params.parallel else 'Disabled (xds)'}")
    processor.log_print(f"Quality Analysis: {'Enabled' if params.quality_analysis else 'Disabled'}")
    processor.log_print(f"XDS Background Pixel: {params.background_pixel}")
    processor.log_print(f"XDS Signal Pixel: {params.signal_pixel}")
    processor.log_print(f"XDS Min Spot Pixels: {params.min_pixel}\n")

    # Log command-line overrides if provided
    if params.detector_distance:
        processor.log_print(f"Detector Distance: {params.detector_distance} (override)")
    if params.exposure:
        processor.log_print(f"Exposure Time: {params.exposure} (override)")
    if params.rotation:
        processor.log_print(f"Rotation Rate: {params.rotation} (override)")
    processor.log_print("")

    processor.process_movie()

if __name__ == "__main__":
    main()
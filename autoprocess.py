"""
Crystallography data processing script
Originally by Jessica Burch, modified by Dmitry Eremin
Refactored version with improved structure and error handling
"""
import os
import sys
import re
import random
import argparse
from subprocess import run, PIPE
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple
from pathlib import Path
import numpy as np

@dataclass
class ProcessingParameters:
    """Data class to hold processing parameters"""
    microscope: str
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
    output_extension: str
    # Add new optional parameters
    detector_distance: Optional[str] = None
    exposure: Optional[str] = None
    rotation: Optional[str] = None

MICROSCOPE_CONFIGS = {
    "Arctica-CETA": ProcessingParameters(
        microscope="Arctica-CETA",
        rotation_axis="0 -1 0",
        frame_size=2048,
        signal_pixel=7,
        min_pixel=7,
        background_pixel=4,
        pixel_size=0.028,
        wavelength="0.0251",
        beam_center_x=1018,
        beam_center_y=1008,
        file_extension=".ser",
        output_extension=".img"
    ),
    "Arctica-EM-core": ProcessingParameters(
        microscope="Arctica-EM-core",
        rotation_axis="1 0 0",
        frame_size=2048,
        signal_pixel=7,
        min_pixel=7,
        background_pixel=4,
        pixel_size=0.028,
        wavelength="0.0251",
        beam_center_x=1018,
        beam_center_y=1008,
        file_extension=".ser",
        output_extension=".img"
    ),
    "Talos-Apollo": ProcessingParameters(
        microscope="Talos-Apollo",
        rotation_axis="1 0 0",
        frame_size=4096,
        signal_pixel=7,
        min_pixel=7,
        background_pixel=4,
        pixel_size=0.008,
        wavelength="0.0251",
        beam_center_x=2040,
        beam_center_y=2020,
        file_extension=".mrc",
        output_extension=".tif"
    )
}

class FileConverter:
    """Handles conversion between different file formats."""
    
    def __init__(self, params: ProcessingParameters):
        self.params = params
        self.current_path = Path.cwd()
        
    def convert_file(self, sample_movie: str, filename: str, distance: str = None, 
                rotation: str = None, exposure: str = None) -> bool:
        """Convert files based on extension type."""
        if self.params.file_extension == '.mrc' and self.params.output_extension == '.tif':
            return self._convert_mrc_to_tif(sample_movie, filename)
        elif self.params.file_extension == '.ser' and self.params.output_extension == '.img':
            return self._convert_ser_to_img(sample_movie, filename, distance, rotation, exposure)
        return True  # Return True for no conversion needed
        
    def _convert_mrc_to_tif(self, sample_movie: str, filename: str) -> bool:
        """Convert MRC to TIF using mrc2tif.py script."""
        try:
            # Check for mrc2tif.py in both current and script directories
            possible_locations = [
                Path().absolute() / "mrc2tif.py",  # Current working directory
                Path(__file__).resolve().parent / "mrc2tif.py"  # Script's directory
            ]
            
            script_path = next(
                (path for path in possible_locations if path.exists()),
                None
            )
            
            if script_path is None:
                logging.error(
                    "Could not find mrc2tif.py in either:\n"
                    f"Current dir: {possible_locations[0].parent}\n"
                    f"Script dir: {possible_locations[1].parent}"
                )
                return False
                
            # Get the parent directory where the MRC file is located
            mrc_folder = Path.cwd().parent  # Go up one level from 'images' directory
                
            # Construct the conversion command with absolute paths
            conversion_cmd = [
                sys.executable,  # Current Python interpreter
                str(script_path),  # Use absolute path to mrc2tif.py
                "--tif-name", sample_movie,
                "--folder", str(mrc_folder),  # Point to the directory containing the MRC file
                "--ped", "1"  # Add pedestal value
            ]
            
            # Run the conversion process
            result = run(conversion_cmd, stdout=PIPE, stderr=PIPE, text=True)
            
            if result.returncode != 0:
                logging.error(f"MRC conversion failed for {filename}: {result.stderr}")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error during MRC conversion for {filename}: {str(e)}")
            return False
        
    def _convert_ser_to_img(self, sample_movie: str, filename: str, distance: str = None, 
                        rotation: str = None, exposure: str = None) -> bool:
        """Convert .ser file to .img format."""
        try:
            ser2smv = "/groups/NelsonLab/programs/ser2smv"
            if not Path(ser2smv).exists():
                logging.error(f"Could not find ser2smv converter at {ser2smv}")
                return False
            
            # Only extract from filename if parameters weren't provided
            if distance is None or rotation is None or exposure is None:
                parts = filename.split('_')
                if len(parts) >= 4:
                    _, distance, rotation, exposure = parts[:4]
                else:
                    logging.error(f"Invalid filename format for {filename}")
                    return False
            
            scaled_pixel_size = self.params.pixel_size / 2
            # Construct and run the conversion command
            conversion_cmd = [
                ser2smv,
                "-P", str(scaled_pixel_size),
                "-B", "2",
                "-r", rotation,
                "-w", self.params.wavelength,
                "-d", distance,
                "-E", exposure,
                "-M", "200",
                "-v",
                "-o", f"{sample_movie}_###.img",
                os.path.join("..", filename)
            ]
            
            result = run(conversion_cmd, stdout=PIPE, stderr=PIPE, text=True)
            
            if result.returncode != 0:
                logging.error(f"SER conversion failed for {filename}: {result.stderr}")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error during SER conversion for {filename}: {str(e)}")
            return False

class CrystallographyProcessor:
    def __init__(self, params: ProcessingParameters):
        self.params = params
        self.file_converter = FileConverter(params)
        self.current_path = Path.cwd()

    @staticmethod
    def setup_logging(log_file: str, dir_name: str) -> None:
        """Configure logging with plain message output to both console and file."""
        log_dir = Path.cwd() / dir_name
        log_dir.mkdir(exist_ok=True)

        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        root_logger.setLevel(logging.INFO)

        plain_formatter = logging.Formatter('%(message)s')

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(plain_formatter)
        root_logger.addHandler(console_handler)

        # File handler
        log_path = log_dir / log_file
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setFormatter(plain_formatter)
        root_logger.addHandler(file_handler)

    @staticmethod
    def log_print(message: str) -> None:
        """Helper function to log messages as plain text."""
        logging.info(message)

    def print_banner(self):
        """Display the program banner."""
        banner_lines = [
            r"",
            r"                     THANK YOU FOR USING                    ",
            r"",
            r"    ___         __       ____                               ",
            r"   /   | __  __/ /_____ / __ \___________________________   ",
            r"  / /| |/ / / / __/ __ / /_/ / __/__ / __/ _  / ___/ ___/hmn",
            r" / ___ / /_/ / /_/ /_// /\__/ // /_// /_/  __(__  (__  )jeb ",
            r"/_/  |_\____/\__/\___/_/   /_/ \___/\___/\__/\___/\___/dbe  ",
            ""
        ]
        for line in banner_lines:
            self.log_print(line)

    def _setup_movie_directories(self, sample_movie: str, distance: str, filename: str) -> Optional[Path]:
        """Set up directory structure for movie processing."""
        movie_dir = Path(sample_movie)
        if not movie_dir.exists():
            movie_dir.mkdir(exist_ok=True)
            movie_path = self.current_path / sample_movie

            # Move .ser/.mrc file
            (self.current_path / filename).rename(movie_path / filename)

            # Move corresponding .emi file if exists
            for emi_file in self.current_path.glob(f"{sample_movie}_{distance}*.emi"):
                emi_file.rename(movie_path / emi_file.name)

            # Create images directory
            image_path = movie_path / "images"
            image_path.mkdir(exist_ok=True)
            
            # Create auto_process directory
            auto_process_path = movie_path / "auto_process"
            auto_process_path.mkdir(exist_ok=True)

            return movie_dir
        return None

    def _process_movie_data(self, sample_movie: str, distance: str, 
                        rotation: str, exposure: str, resolution_range: float,
                        test_resolution_range: float, filename: str) -> None:
        """Process the movie data after directories are set up."""
        try:
            # Change to images directory first
            image_dir = Path(sample_movie) / "images"
            os.chdir(str(image_dir))
            
            # Handle file conversion
            if not self.file_converter.convert_file(sample_movie, filename, distance, rotation, exposure):
                self.log_print(f"Failed to convert {filename}. Skipping processing.")
                os.chdir(str(self.current_path))
                return
                
            # Count converted images
            image_files = list(Path().glob(f"*{self.params.output_extension}"))
            if not image_files:
                self.log_print(f"No converted images found in {image_dir}")
                os.chdir(str(self.current_path))
                return
                
            image_number = str(len(image_files))
            self.log_print(f"Found {image_number} converted images")
            
            # Change to auto_process directory for XDS processing
            os.chdir(str(Path("..") / "auto_process"))
            
            # Create and process XDS.INP
            params = {
                'distance': distance,
                'rotation': rotation,
                'exposure': exposure,
                'resolution_range': resolution_range,
                'test_resolution_range': test_resolution_range,
                'image_number': image_number,
                'background_pixel': self.params.background_pixel,
                'signal_pixel': self.params.signal_pixel,
                'min_pixel': self.params.min_pixel,
            }

            data_path = os.path.join(str(Path("..") / "images"), sample_movie)
            xds_content = self.create_xds_input(data_path, params)
            
            # Write XDS.INP in current directory (auto_process)
            with open('XDS.INP', 'w') as xds_inp:
                xds_inp.write(xds_content)
                
            # Run XDS
            with open('XDS.LP', "w+") as xds_out:
                self.log_print(f"\nProcessing {sample_movie}...\n")
                run("xds", stdout=xds_out)
                
            self.process_check(sample_movie)
            
        except Exception as e:
            self.log_print(f"Error processing movie data: {str(e)}")
        finally:
            os.chdir(str(self.current_path))

    def _get_crystal_parameters(self) -> Tuple[Optional[str], Optional[str]]:
        """Extract final space group and unit cell parameters from XDS.LP using regex.
        Takes only the last occurrence of each parameter, as these represent
        the final refined values.

        Returns:
            Tuple containing (space_group, unit_cell_string) or (None, None) if not found
        """
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
        self.log_print(f"{unit_cell}\n")

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

    def create_xds_input(self, data_path: str, params: dict) -> str:
        """Generate XDS.INP content with support for different file extensions."""
        template = f"""JOB= XYCORR INIT COLSPOT IDXREF DEFPIX INTEGRATE CORRECT
!JOB=DEFPIX INTEGRATE CORRECT
ORGX= {self.params.beam_center_x} ORGY= {self.params.beam_center_y}
DETECTOR_DISTANCE= {float(params['distance'])}
OSCILLATION_RANGE= {float(params['exposure']) * float(params['rotation'])}
X-RAY_WAVELENGTH= {self.params.wavelength}

NAME_TEMPLATE_OF_DATA_FRAMES= {data_path}_???{self.params.output_extension}

BACKGROUND_RANGE=1 10

!DELPHI=15
!SPACE_GROUP_NUMBER=0
!UNIT_CELL_CONSTANTS= 1 1 1 90 90 90
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

DATA_RANGE= 1 {params['image_number']}
SPOT_RANGE= 1 {params['image_number']}

BACKGROUND_PIXEL= {params['background_pixel']}
SIGNAL_PIXEL= {params['signal_pixel']}
MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT= {params['min_pixel']}
"""
        return template

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

    def _run_xds(self, message: str) -> None:
        """Run XDS with logging."""
        self.log_print(message)
        with open('XDS.LP', "w+") as xds_out:
            run("xds", stdout=xds_out)

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

            self.log_print("Screening new indexing values.")
            self._run_xds("XDS is running...")

            if Path('XPARM.XDS').exists():
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

        # Check if all Isa2 values are the same
        if len(set(Isa2_values)) > 1:
            self.log_print(f"Unable to process {sample_movie} - Inconsistent ISa values: {Isa2_values}")
            return None

        # Use the Isa2 value (they're all the same if we got here)
        Isa2 = Isa2_values[0]
        
        if abs(Isa2 - Isa1) < 0.01:
            self.log_print(f"Same ISa value of {Isa2} was obtained")
        else:
            self.log_print(f"ISa: {Isa2}")

        if abs(Isa2 - Isa1) > 0.5:
            self.log_print("I'm trying to optimize beam divergence values.")
            return self.iterate_opt(sample_movie, Isa2)  # Pass current ISa as previous_isa
        else:
            # Get crystal parameters using the dedicated method
            space_group, unit_cell = self._get_crystal_parameters()
            
            if space_group and unit_cell:
                self.log_print("Optimized beam divergence values.")
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
        """Check space group using CCP4's pointless."""
        with open("xdsconv_ap.LP", "w+") as xdsconv_out:
            run("xdsconv", stdout=xdsconv_out)
        self.log_print("I converted it for use in shelx!")

        # Run pointless
        result = run("/central/groups/NelsonLab/programs/ccp4-8.0/bin/pointless XDS_ASCII.HKL > pointless.LP",
            shell=True, capture_output=True)
        
        # Check if pointless ran successfully
        if result.returncode != 0:
            self.log_print("Warning: Could not run pointless, but processing completed")
            return False  # Stop further processing
            
        self._process_pointless_output()
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
        files = os.listdir()
        
        for filename in files:
            file_info = self.parse_filename(filename)
            if not file_info:
                continue
                
            sample_movie, distance, rotation, exposure = file_info
            ranges = self.calculate_resolution_ranges(distance)
            if ranges is None:
                continue
                
            resolution_range, test_resolution_range = ranges
                
            self._process_single_movie(
                sample_movie, distance, rotation, exposure, 
                resolution_range, test_resolution_range,  # Add test_resolution_range
                filename
            )
    
    def _process_single_movie(self, sample_movie: str, distance: str, 
                            rotation: str, exposure: str, resolution_range: float,
                            test_resolution_range: float,  # Add this parameter
                            filename: str) -> None:
        """Process a single movie file."""
        if Path(sample_movie).exists():
            self.log_print(f"Already processed {filename}")
            return
            
        movie_dir = self._setup_movie_directories(
            sample_movie, distance, filename
        )
        if not movie_dir:
            return
            
        self._process_movie_data(
            sample_movie, distance, rotation, exposure,
            resolution_range, test_resolution_range,  # Add test_resolution_range
            filename
        )

def parse_arguments() -> ProcessingParameters:
    """Parse command line arguments and return ProcessingParameters instance."""
    parser = argparse.ArgumentParser(
        description='Process crystallography data files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--microscope', 
                       type=str, 
                       default='Arctica-CETA',
                       choices=list(MICROSCOPE_CONFIGS.keys()),
                       help='Choose instrument for default settings')
    
    # Add arguments for overriding default parameters
    parser.add_argument('--rotation-axis',
                       type=str,
                       help='Override rotation axis')
    
    parser.add_argument('--frame-size',
                       type=int,
                       help='Override frame size')
    
    parser.add_argument('--signal-pixel',
                       type=int,
                       help='Override signal pixel value')
    
    parser.add_argument('--min-pixel',
                       type=int,
                       help='Override minimum pixel value')
    
    parser.add_argument('--background-pixel',
                       type=int,
                       help='Override background pixel value')
    
    parser.add_argument('--pixel-size',
                       type=float,
                       help='Override pixel size value')
    
    parser.add_argument('--beam-center-x',
                       type=int,
                       help='Override beam center X coordinate')
    
    parser.add_argument('--beam-center-y',
                       type=int,
                       help='Override beam center Y coordinate')
    
    parser.add_argument('--file-extension',
                       type=str,
                       help='Override input file extension')
    
    parser.add_argument('--output-extension',
                       type=str,
                       help='Override output file extension')

    # Add new arguments for runtime parameters
    parser.add_argument('--detector-distance',
                       type=str,
                       help='Override detector distance (in mm)')
    
    parser.add_argument('--exposure',
                       type=str,
                       help='Override exposure time')
    
    parser.add_argument('--rotation',
                       type=str,
                       help='Override rotation value')

    args = parser.parse_args()
    
    # Start with the default configuration for the selected microscope
    base_params = MICROSCOPE_CONFIGS[args.microscope]
    params = ProcessingParameters(
        microscope=args.microscope,
        rotation_axis=base_params.rotation_axis,
        frame_size=base_params.frame_size,
        signal_pixel=base_params.signal_pixel,
        min_pixel=base_params.min_pixel,
        background_pixel=base_params.background_pixel,
        pixel_size=base_params.pixel_size,
        wavelength=base_params.wavelength,
        beam_center_x=base_params.beam_center_x,
        beam_center_y=base_params.beam_center_y,
        file_extension=base_params.file_extension,
        output_extension=base_params.output_extension,
        detector_distance=None,  # Initialize new parameters
        exposure=None,
        rotation=None
    )
    
    # Override parameters if specified in command line arguments
    if args.rotation_axis:
        params.rotation_axis = args.rotation_axis
    if args.frame_size:
        params.frame_size = args.frame_size
    if args.signal_pixel:
        params.signal_pixel = args.signal_pixel
    if args.min_pixel:
        params.min_pixel = args.min_pixel
    if args.background_pixel:
        params.background_pixel = args.background_pixel
    if args.pixel_size:
        params.pixel_size = args.pixel_size
    if args.beam_center_x:
        params.beam_center_x = args.beam_center_x
    if args.beam_center_y:
        params.beam_center_y = args.beam_center_y
    if args.file_extension:
        params.file_extension = args.file_extension
    if args.output_extension:
        params.output_extension = args.output_extension
    if args.detector_distance:
        params.detector_distance = args.detector_distance
    if args.exposure:
        params.exposure = args.exposure
    if args.rotation:
        params.rotation = args.rotation
    
    return params

def main():
    # Parse arguments and create ProcessingParameters instance
    params = parse_arguments()
    
    # Initialize processor with parsed parameters
    processor = CrystallographyProcessor(params)
    
    # Log the current parameters being used
    processor.log_print("\nUsing processing parameters:")
    processor.log_print(f"Microscope: {params.microscope}")
    processor.log_print(f"Rotation Axis: {params.rotation_axis}")
    processor.log_print(f"Frame Size: {params.frame_size}")
    processor.log_print(f"File Extension: {params.file_extension}")
    processor.log_print(f"Output Extension: {params.output_extension}")
    processor.log_print(f"Signal Pixel: {params.signal_pixel}")
    processor.log_print(f"Min Pixel: {params.min_pixel}")
    processor.log_print(f"Background Pixel: {params.background_pixel}")
    processor.log_print(f"Pixel Size: {params.pixel_size}")
    processor.log_print(f"Wavelength: {params.wavelength} (fixed)")
    processor.log_print(f"Beam Center X: {params.beam_center_x}")
    processor.log_print(f"Beam Center Y: {params.beam_center_y}\n")
    
    # Setup logging and start processing
    processor.setup_logging(log_file="autoprocess.log", dir_name="autoprocess_logs")
    processor.print_banner()
    processor.process_movie()

if __name__ == "__main__":
    main()
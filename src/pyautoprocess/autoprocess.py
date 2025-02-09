"""
Crystallography data processing script
Originally by Jessica Burch, modified by Dmitry Eremin
Refactored version with improved structure and error handling
"""
import os
import importlib.resources
import json
import sys
import re
import random
import argparse
from subprocess import run, PIPE
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import numpy as np

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
    detector_distance: Optional[str] = None
    exposure: Optional[str] = None
    rotation: Optional[str] = None
    microscope_config: str = "default"  # Move to end with default

class ConfigLoader:
    def __init__(self, config_path: str = "microscope_configs.json"):
        self.config_path = config_path
        self.configs = self._load_configs()

    def _load_configs(self) -> Dict:
        try:
            # Try loading from package first
            try:
                with importlib.resources.open_text('pyautoprocess.data', 'microscope_configs.json') as f:
                    return json.load(f)
            except Exception as pkg_error:
                logging.debug(f"Could not load from package: {pkg_error}")
                
            # Try loading from local path
            if Path(self.config_path).exists():
                with open(self.config_path) as f:
                    return json.load(f)
                    
            logging.warning("No config file found, using default configuration")
            return {"default": self._get_default_config()}
            
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return {"default": self._get_default_config()}

    @staticmethod
    def _get_default_config() -> Dict:
        """Return default configuration"""
        return {
            "rotation_axis": "-1 0 0",
            "frame_size": 2048,
            "signal_pixel": 7,
            "min_pixel": 7,
            "background_pixel": 4,
            "pixel_size": 0.028,
            "wavelength": "0.0251",
            "beam_center_x": 1030,
            "beam_center_y": 1040,
            "file_extension": ".ser"
        }

    def get_config(self, microscope_name: str) -> ProcessingParameters:
        if microscope_name not in self.configs:
            logging.warning(f"Configuration '{microscope_name}' not found, using default")
            config = self._get_default_config()
        else:
            config = self.configs[microscope_name]
            logging.info(f"Loaded configuration for {microscope_name}")
            
        # Remove microscope_config from parameters if present
        if isinstance(config, dict):
            config = {k: v for k, v in config.items() if k != 'microscope_config'}
            
        return ProcessingParameters(**config, microscope_config=microscope_name)

    def get_available_configs(self) -> list:
        """Return list of available microscope configurations"""
        return list(self.configs.keys())

class FileConverter:
    """Handles conversion between different file formats."""
    
    def __init__(self, params: ProcessingParameters):
        self.params = params
        self.current_path = Path.cwd()
        
    def convert_file(self, sample_movie: str, filename: str, distance: str = None, 
                    rotation: str = None, exposure: str = None) -> bool:
        """Convert files based on extension type."""
        if self.params.file_extension == '.mrc':
            return self._convert_mrc_to_tif(sample_movie, filename)
        elif self.params.file_extension == '.ser':
            return self._convert_ser_to_tif(sample_movie, filename)
        return True
        
    def _convert_mrc_to_tif(self, sample_movie: str, filename: str) -> bool:
        """Convert MRC to TIF using mrc2tif.py script."""
        try:
            # Get the parent directory where the MRC file is located
            mrc_folder = Path.cwd().parent  # Go up one level from 'images' directory
                
            # Construct the conversion command with absolute paths
            conversion_cmd = [
                'mrc2tif',
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
    
    def _convert_ser_to_tif(self, sample_movie: str, filename: str) -> bool:
        """Convert SER to TIF using ser2tif.py script."""
        try:    
            # Get the parent directory where the SER file is located
            ser_folder = Path.cwd().parent  # Go up one level from 'images' directory
                
            # Construct the conversion command with absolute paths
            conversion_cmd = [
                'ser2tif',
                "--tif-name", sample_movie,
                "--folder", str(ser_folder),  # Point to the directory containing the SER file
                "--ped", "200"  # Add pedestal value
            ]
            
            # Run the conversion process
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
    
    def print_sub_banner(self):
        """Print subbanner."""

        version_banner = [
            "",
            "================================================================",
            "                        AutoProcess 2.0                         ",
            "================================================================",
            ""
        ]
        
        for line in version_banner:
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
            image_files = list(Path().glob(f"*.tif"))
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

NAME_TEMPLATE_OF_DATA_FRAMES= {data_path}_???.tif

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
        result = run("pointless XDS_ASCII.HKL > pointless.LP",
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

        processed_movie = False
        
        for filename in files:
            file_info = self.parse_filename(filename)
            if not file_info:
                continue
            
            sample_movie, distance, rotation, exposure = file_info
            ranges = self.calculate_resolution_ranges(distance)
            if ranges is None:
                continue
                
            resolution_range, test_resolution_range = ranges
                
            processed_movie = True
            self._process_single_movie(
                sample_movie, distance, rotation, exposure, 
                resolution_range, test_resolution_range,  # Add test_resolution_range
                filename
            )
        
        if not processed_movie:
            self.log_print(f'Found no movies to process, searched {files}')
    
    def _process_single_movie(self, sample_movie: str, distance: str, 
                            rotation: str, exposure: str, resolution_range: float,
                            test_resolution_range: float,  # Add this parameter
                            filename: str) -> None:
        """Process a single movie file."""
        if Path(sample_movie).exists():
            self.log_print(f"Already processed {filename}")
            return
        
        # Log processing parameters for this movie
        self.log_print(f"\nProcessing parameters for {filename}:")
        self.log_print(f"Detector Distance: {distance} mm")
        self.log_print(f"Exposure Time: {exposure} s")
        self.log_print(f"Rotation Rate: {rotation} deg/s")
        self.log_print(f"Resolution Range: {resolution_range} Å")
        self.log_print(f"Test Resolution Range: {test_resolution_range} Å\n")
            
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
    # Create initial parser for microscope config
    pre_parser = argparse.ArgumentParser(add_help=False)
    config_loader = ConfigLoader()
    available_configs = config_loader.get_available_configs()
    
    pre_parser.add_argument('--microscope-config', 
                           type=str, 
                           default='default',
                           choices=available_configs)
    
    # Get microscope config
    known_args, _ = pre_parser.parse_known_args()
    config = config_loader.get_config(known_args.microscope_config)
    
    # Create main parser with all arguments
    parser = argparse.ArgumentParser(
        description='Process crystallography data files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add all arguments including microscope config
    parser.add_argument('--microscope-config', 
                       type=str, 
                       default='default',
                       choices=available_configs,
                       help='Choose instrument configuration')
                       
    parser.add_argument('--config-file',
                       type=str,
                       default='microscope_configs.json',
                       help='Path to microscope configuration file')
                       
    parser.add_argument('--rotation-axis', 
                       type=str, 
                       default=config.rotation_axis,
                       help='Override rotation axis')
                       
    parser.add_argument('--frame-size', 
                       type=int, 
                       default=config.frame_size,
                       help='Override frame size')
                       
    parser.add_argument('--signal-pixel', 
                       type=int, 
                       default=config.signal_pixel,
                       help='Override signal pixel value')
                       
    parser.add_argument('--min-pixel', 
                       type=int, 
                       default=config.min_pixel,
                       help='Override minimum pixel value')
                       
    parser.add_argument('--background-pixel', 
                       type=int, 
                       default=config.background_pixel,
                       help='Override background pixel value')
                       
    parser.add_argument('--pixel-size', 
                       type=float, 
                       default=config.pixel_size,
                       help='Override pixel size value')
                       
    parser.add_argument('--wavelength', 
                       type=str, 
                       default=config.wavelength,
                       help='Override wavelength value')
                       
    parser.add_argument('--beam-center-x', 
                       type=int, 
                       default=config.beam_center_x,
                       help='Override beam center X coordinate')
                       
    parser.add_argument('--beam-center-y', 
                       type=int, 
                       default=config.beam_center_y,
                       help='Override beam center Y coordinate')
                       
    parser.add_argument('--file-extension', 
                       type=str, 
                       default=config.file_extension,
                       help='Override input file extension')
                       
    parser.add_argument('--detector-distance', 
                       type=str, 
                       default=config.detector_distance,
                       help='Override detector distance (in mm)')
                       
    parser.add_argument('--exposure', 
                       type=str, 
                       default=config.exposure,
                       help='Override exposure time')
                       
    parser.add_argument('--rotation', 
                       type=str, 
                       default=config.rotation,
                       help='Override rotation value')

    args = parser.parse_args()
    
    return ProcessingParameters(
        rotation_axis=args.rotation_axis,
        frame_size=args.frame_size,
        signal_pixel=args.signal_pixel,
        min_pixel=args.min_pixel,
        background_pixel=args.background_pixel,
        pixel_size=args.pixel_size,
        wavelength=args.wavelength,
        beam_center_x=args.beam_center_x,
        beam_center_y=args.beam_center_y,
        file_extension=args.file_extension,
        detector_distance=args.detector_distance,
        exposure=args.exposure,
        rotation=args.rotation,
        microscope_config=args.microscope_config
    )

def main():
    # Parse arguments and create ProcessingParameters instance
    params = parse_arguments()
    
    # Initialize processor with parsed parameters
    processor = CrystallographyProcessor(params)
    
    # Setup logging and start processing
    processor.setup_logging(log_file="autoprocess.log", dir_name="autoprocess_logs")
    processor.print_banner()
    processor.print_sub_banner()

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
    processor.log_print(f"Beam Center Y: {params.beam_center_y}\n")

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
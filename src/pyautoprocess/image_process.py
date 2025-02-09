"""
Process pre-converted crystallography images
Based on autoprocess.py and batch_reprocess.py
Handles pre-converted TIF images in existing directory structure
"""
import os
from datetime import datetime
import argparse
from pathlib import Path
import shutil
from dataclasses import dataclass
from subprocess import run, PIPE
from typing import Optional, List
from .autoprocess import (CrystallographyProcessor, ProcessingParameters,
                           ConfigLoader)

@dataclass
class ExtendedProcessingParameters(ProcessingParameters):
    """Extended processing parameters to include SMV flag and frame trimming"""
    smv: bool = False
    trim_front: int = 0
    trim_end: int = 0

class PreConvertedProcessor:
    """Process pre-converted crystallography images with backup functionality"""
    
    OUTPUT_FOLDER = "auto_process_direct"
    BACKUP_FOLDER = "processing_backups"
    
    def __init__(self, params: ProcessingParameters):
        self.params = params
        self.processor = CrystallographyProcessor(params)
        self.current_path = Path.cwd()
        self.file_extension = '.img' if hasattr(params, 'smv') and params.smv else '.tif'

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
        """Print the program banner first, then batch processing banner."""
        # Print the standard program banner first
        self.processor.print_banner()
        
        # Print batch processing banner
        image_banner = [
            "",
            "================================================================",
            "                        AutoProcess 2.0                         ",
            "                     IMAGE PROCESSING MODE                      ",
            "================================================================",
            ""
        ]
        
        for line in image_banner:
            self.processor.log_print(line)

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

    def process_data(self, sample_path: Path) -> None:
        """Process data using the same logic as autoprocess"""
        try:
            # Change to images directory first
            image_dir = sample_path / "images"
            os.chdir(str(image_dir))
            
            # Determine file extension based on SMV flag
            ext = '*.img' if self.params.smv else '*.tif'
            
            # Count images
            image_files = sorted(list(Path().glob(ext)))
            if not image_files:
                self.processor.log_print(f"No images found in {image_dir}")
                os.chdir(str(self.current_path))
                return
                
            image_number = str(len(image_files))
            base_name = image_files[0].name.rsplit('_', 1)[0]
            self.processor.log_print(f"Found {image_number} images")
            
            # Change to auto_process_direct directory for XDS processing
            os.chdir(str(Path("..") / self.OUTPUT_FOLDER))
            
            # Calculate resolution ranges using processor's method
            ranges = self.processor.calculate_resolution_ranges(self.params.detector_distance)
            if ranges is None:
                self.processor.log_print("Could not calculate resolution ranges")
                os.chdir(str(self.current_path))
                return
                
            resolution_range, test_resolution_range = ranges
            
            # Log processing parameters
            # Extract sample name from the path
            sample_name = sample_path.name
            
            # Log processing parameters using values from self.params
            self.processor.log_print(f"\nProcessing parameters for {sample_name}:")
            self.processor.log_print(f"Detector Distance: {self.params.detector_distance} mm")
            self.processor.log_print(f"Exposure Time: {self.params.exposure} s")
            self.processor.log_print(f"Rotation Rate: {self.params.rotation} deg/s")
            self.processor.log_print(f"Resolution Range: {resolution_range} Å")
            self.processor.log_print(f"Test Resolution Range: {test_resolution_range} Å\n")

            # Create initial XDS.INP with .tif extension - we'll modify it later
            params = {
                'distance': self.params.detector_distance,
                'rotation': self.params.rotation,
                'exposure': self.params.exposure,
                'resolution_range': resolution_range,
                'test_resolution_range': test_resolution_range,
                'image_number': image_number,
                'background_pixel': self.params.background_pixel,
                'signal_pixel': self.params.signal_pixel,
                'min_pixel': self.params.min_pixel,
            }
            
            data_path = os.path.join(str(Path("..") / "images"), base_name)
            xds_content = self.processor.create_xds_input(data_path, params)
                
            # Write initial XDS.INP
            with open('XDS.INP', 'w') as xds_inp:
                xds_inp.write(xds_content)
                
            # If we're in SMV mode, modify the template line
            if self.params.smv:
                self._modify_xds_template_extension()
            
            # Modify spot range if trim parameters are set
            if self.params.trim_front > 0 or self.params.trim_end > 0:
                self._modify_spot_range(int(image_number))
            
            # Run XDS
            with open('XDS.LP', "w+") as xds_out:
                self.processor.log_print(f"\nProcessing {sample_path.name}...\n")
                run("xds", stdout=xds_out)
                
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
    
    def _modify_spot_range(self, total_images: int) -> None:
        """Modify SPOT_RANGE in XDS.INP based on trim parameters"""
        try:
            # Calculate new start and end frame numbers
            start_frame = 1 + self.params.trim_front
            end_frame = total_images - self.params.trim_end
            
            # Validate the range
            if start_frame >= end_frame or start_frame < 1 or end_frame > total_images:
                self.processor.log_print(
                    f"Warning: Invalid frame range {start_frame}-{end_frame}, using full range"
                )
                return
                
            with open('XDS.INP', 'r') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines):
                if line.strip().startswith('SPOT_RANGE='):
                    lines[i] = f'SPOT_RANGE= {start_frame} {end_frame}\n'
                    
            with open('XDS.INP', 'w') as f:
                f.writelines(lines)
                
            self.processor.log_print(f"Modified spot range to: {start_frame}-{end_frame}")
                
        except Exception as e:
            self.processor.log_print(f"Error modifying spot range in XDS.INP: {str(e)}")
            raise

    def process_folder(self, sample_path: Path) -> bool:
        """Process a single folder containing pre-converted images"""
        try:
            # Setup processing directory
            process_dir = sample_path / self.OUTPUT_FOLDER
            
            # Handle existing directory
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

    def process_all(self) -> None:
        """Process all valid folders in current directory"""
        valid_folders = self.find_valid_folders()
        
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
                       default="960",  # Set default value here
                       help='Override detector distance (in mm)')
                       
    parser.add_argument('--exposure', 
                       type=str, 
                       default="3.0",  # Set default value here
                       help='Override exposure time')
                       
    parser.add_argument('--rotation', 
                       type=str, 
                       default="0.3",  # Set default value here
                       help='Override rotation value')
                       
    parser.add_argument('--smv',
                       action='store_true',
                       help='Process SMV (.img) files instead of TIF files')
    
    parser.add_argument('--trim-front',
                       type=int,
                       default=0,
                       help='Number of frames to trim from the start of the range')
                       
    parser.add_argument('--trim-end',
                       type=int,
                       default=0,
                       help='Number of frames to trim from the end of the range')

    args = parser.parse_args()
    
    return ExtendedProcessingParameters(
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
        microscope_config=args.microscope_config,
        smv=args.smv,
        trim_front=args.trim_front,
        trim_end=args.trim_end
    )

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
    processor.processor.log_print(f"Detector Distance: {params.detector_distance}")
    processor.processor.log_print(f"Exposure Time: {params.exposure}")
    processor.processor.log_print(f"Rotation Rate: {params.rotation}")
    if params.trim_front > 0 or params.trim_end > 0:
        processor.processor.log_print(
            f"Frame trimming: {params.trim_front} frames from start, {params.trim_end} frames from end"
        )
    processor.processor.log_print("")
    
    processor.process_all()

if __name__ == "__main__":
    main()
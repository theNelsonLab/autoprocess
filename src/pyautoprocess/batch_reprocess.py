"""
Crystallography batch reprocessing script
Based on autoprocess.py by Jessica Burch, modified by Dmitry Eremin
Allows reprocessing of data with specific space group and unit cell parameters
"""
import os
from pathlib import Path
import argparse
import shutil
from typing import Optional
from dataclasses import dataclass
from .autoprocess import (CrystallographyProcessor, 
                        MICROSCOPE_CONFIGS)

@dataclass
class BatchParameters:
    """Data class to hold batch processing parameters"""
    space_group: int
    unit_cell_a: float
    unit_cell_b: float
    unit_cell_c: float
    unit_cell_alpha: float
    unit_cell_beta: float
    unit_cell_gamma: float
    subfolder: str
    microscope: str
    signal_pixel: Optional[int] = None
    min_pixel: Optional[int] = None
    background_pixel: Optional[int] = None

    @classmethod
    def from_input(cls, args: Optional[argparse.Namespace] = None) -> 'BatchParameters':
        """Create BatchParameters from command line args or interactive input."""
        def get_value(arg_value: Optional[str], prompt: str, convert_type=float) -> any:
            if arg_value is not None:
                return convert_type(arg_value)
            return convert_type(input(prompt))
        
        # Use args directly since we know they exist from argparse
        microscope = args.microscope
        
        # Get required parameters
        space_group = get_value(args.space_gr, 'Space group #? ', int)
        unit_cell_a = get_value(args.a, 'Unit cell a? ')
        unit_cell_b = get_value(args.b, 'Unit cell b? ')
        unit_cell_c = get_value(args.c, 'Unit cell c? ')
        unit_cell_alpha = get_value(args.alpha, 'Unit cell alpha? ')
        unit_cell_beta = get_value(args.beta, 'Unit cell beta? ')
        unit_cell_gamma = get_value(args.gamma, 'Unit cell gamma? ')
        subfolder = get_value(args.folder, 'Subfolder name? ', str)

        # Get processing parameters directly
        signal_pixel = args.signal_pixel
        min_pixel = args.min_pixel
        background_pixel = args.background_pixel
        
        # If signal_pixel is provided but others aren't, use matching defaults
        if signal_pixel is not None:
            min_pixel = min_pixel if min_pixel is not None else signal_pixel
            background_pixel = background_pixel if background_pixel is not None else min(4, signal_pixel - 3)

        return cls(
            space_group=space_group,
            unit_cell_a=unit_cell_a,
            unit_cell_b=unit_cell_b,
            unit_cell_c=unit_cell_c,
            unit_cell_alpha=unit_cell_alpha,
            unit_cell_beta=unit_cell_beta,
            unit_cell_gamma=unit_cell_gamma,
            subfolder=subfolder,
            microscope=microscope,
            signal_pixel=signal_pixel,
            min_pixel=min_pixel,
            background_pixel=background_pixel
        )

class BatchProcessor:
    def __init__(self, batch_params: BatchParameters):
        self.params = batch_params
        # Use microscope-specific parameters
        microscope_params = MICROSCOPE_CONFIGS[batch_params.microscope]
        self.processor = CrystallographyProcessor(microscope_params)
        self.current_path = Path.cwd()

    def print_batch_banner(self):
        """Print the program banner first, then batch processing banner."""
        # Print the standard program banner first
        self.processor.print_banner()
        
        # Print batch processing banner
        batch_banner = [
            "",
            "================================================================",
            "                   BATCH REPROCESSING MODE                        ",
            "================================================================",
            ""
        ]
        
        for line in batch_banner:
            self.processor.log_print(line)

    def _copy_inp_files(self, source_dir: Path, target_dir: Path) -> None:
        """Copy only .INP files from source to target directory."""
        for item in source_dir.iterdir():
            if item.is_file() and item.name.endswith('.INP'):
                try:
                    shutil.copy2(item, target_dir)
                    self.processor.log_print(f"Copied {item.name}")
                except Exception as e:
                    self.processor.log_print(f"Error copying {item.name}: {str(e)}")

    def _modify_xds_inp(self, xds_path: Path) -> None:
        """Modify XDS.INP with batch processing parameters."""
        try:
            with open(xds_path, 'r') as f:
                lines = f.readlines()

            modified_lines = []
            modified_lines.append("JOB=XYCORR INIT COLSPOT IDXREF DEFPIX INTEGRATE CORRECT\n")
            modified_lines.append("!JOB=DEFPIX INTEGRATE CORRECT\n")
            
            for line in lines[2:]:
                if line.strip().startswith(('!SPACE_GROUP_NUMBER=', 'SPACE_GROUP_NUMBER=')):
                    modified_lines.append(f"SPACE_GROUP_NUMBER= {self.params.space_group}\n")
                elif line.strip().startswith(('!UNIT_CELL_CONSTANTS=', 'UNIT_CELL_CONSTANTS=')):
                    modified_lines.append(
                        f"UNIT_CELL_CONSTANTS= {self.params.unit_cell_a} {self.params.unit_cell_b} "
                        f"{self.params.unit_cell_c} {self.params.unit_cell_alpha} "
                        f"{self.params.unit_cell_beta} {self.params.unit_cell_gamma}\n"
                    )
                elif line.strip().startswith('SIGNAL_PIXEL=') and self.params.signal_pixel is not None:
                    modified_lines.append(f"SIGNAL_PIXEL= {self.params.signal_pixel}\n")
                elif line.strip().startswith('MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT=') and self.params.min_pixel is not None:
                    modified_lines.append(f"MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT= {self.params.min_pixel}\n")
                elif line.strip().startswith('BACKGROUND_PIXEL=') and self.params.background_pixel is not None:
                    modified_lines.append(f"BACKGROUND_PIXEL= {self.params.background_pixel}\n")
                else:
                    modified_lines.append(line)

            with open(xds_path, 'w') as f:
                f.writelines(modified_lines)

            self.processor.log_print("Updated XDS.INP with batch processing parameters")
            if any([self.params.signal_pixel, self.params.min_pixel, self.params.background_pixel]):
                self.processor.log_print("Applied custom processing parameters:")
                if self.params.signal_pixel is not None:
                    self.processor.log_print(f"  Signal Pixel: {self.params.signal_pixel}")
                if self.params.min_pixel is not None:
                    self.processor.log_print(f"  Min Pixel: {self.params.min_pixel}")
                if self.params.background_pixel is not None:
                    self.processor.log_print(f"  Background Pixel: {self.params.background_pixel}")

        except Exception as e:
            self.processor.log_print(f"Error modifying XDS.INP: {str(e)}")
            raise

    def process_directory(self) -> None:
        """Process all valid directories in the current path."""
        processed_count = 0
        skipped_count = 0
        error_count = 0

        for name in os.listdir(self.current_path):
            dir_path = self.current_path / name
            if not dir_path.is_dir():
                continue

            auto_process_dir = dir_path / "auto_process"
            if not (auto_process_dir.exists() and (auto_process_dir / "XDS.INP").exists()):
                self.processor.log_print(f"Skipping {name}: No auto_process directory or XDS.INP")
                skipped_count += 1
                continue

            try:
                # Create or clean subfolder at the same level as auto_process
                subfolder_path = dir_path / self.params.subfolder
                if subfolder_path.exists():
                    self.processor.log_print(f"\nReprocessing {name} in existing subfolder {self.params.subfolder}")
                    shutil.rmtree(subfolder_path)
                else:
                    self.processor.log_print(f"\nProcessing {name} in new subfolder {self.params.subfolder}")
                
                subfolder_path.mkdir(parents=True)
                
                # Change to subfolder directory
                original_dir = os.getcwd()
                os.chdir(subfolder_path)
                
                # Copy necessary files
                self.processor.log_print(f"Copying .INP files from {auto_process_dir}")
                self._copy_inp_files(auto_process_dir, subfolder_path)
                
                # Modify XDS.INP
                self._modify_xds_inp(subfolder_path / "XDS.INP")
                
                # Process the data
                success = False
                try:
                    self.processor.log_print(f"\nProcessing {name}...")
                    self.processor._run_xds("XDS is running...")

                    # Let process_check handle everything
                    success = self.processor.process_check(name)
                    
                except Exception as e:
                    self.processor.log_print(f"Error during processing: {str(e)}")
                    success = False

                if success:
                    processed_count += 1
                    self.processor.log_print(f"Successfully processed {name}")
                else:
                    error_count += 1
                    self.processor.log_print(f"Failed to process {name}")
                
                # Return to original directory
                os.chdir(original_dir)

            except Exception as e:
                self.processor.log_print(f"Error setting up processing for {name}: {str(e)}")
                error_count += 1
                os.chdir(self.current_path)
                continue

        # Print summary
        self.processor.log_print("\nProcessing Summary:")
        self.processor.log_print(f"Successfully processed: {processed_count}")
        self.processor.log_print(f"Skipped: {skipped_count}")
        self.processor.log_print(f"Errors: {error_count}")

def parse_arguments() -> Optional[argparse.Namespace]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Batch reprocess crystallography data with specific space group and unit cell parameters.'
    )
    
    # Add microscope argument
    parser.add_argument('--microscope', 
                       type=str, 
                       default='Arctica-CETA',
                       choices=list(MICROSCOPE_CONFIGS.keys()),
                       help='Choose instrument for default settings')
    
    # Unit cell and space group parameters
    parser.add_argument('--space-gr', help='Space group number')
    parser.add_argument('--a', help='Unit cell parameter a')
    parser.add_argument('--b', help='Unit cell parameter b')
    parser.add_argument('--c', help='Unit cell parameter c')
    parser.add_argument('--alpha', help='Unit cell angle alpha')
    parser.add_argument('--beta', help='Unit cell angle beta')
    parser.add_argument('--gamma', help='Unit cell angle gamma')
    parser.add_argument('--folder', help='Name of subfolder for reprocessed data')
    
    # Processing parameters group
    processing_group = parser.add_mutually_exclusive_group()
    processing_group.add_argument('--default-params', action='store_true', 
                                help='Use default processing parameters (signal=7, min=7, background=4)')
    processing_group.add_argument('--signal-pixel', type=int, help='Signal pixel value')
    
    # These can be used with --signal-pixel but not with --default-params
    parser.add_argument('--min-pixel', type=int, help='Minimum pixel value')
    parser.add_argument('--background-pixel', type=int, help='Background pixel value (max 5)')

    args = parser.parse_args()
    
    # Apply defaults if requested
    if args.default_params:
        args.signal_pixel = 7
        args.min_pixel = 7
        args.background_pixel = 4
    
    # Validate background pixel
    if args.background_pixel is not None and args.background_pixel > 5:
        parser.error("Background pixel value cannot be larger than 5")
    
    return args

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Get parameters from args or interactive input
    batch_params = BatchParameters.from_input(args)
    
    # Initialize processor
    processor = BatchProcessor(batch_params)
    
    # Setup logging exactly as in autoprocess.py
    processor.processor.setup_logging("batch_reprocess.log", "autoprocess_logs")
    
    # Print banners
    processor.print_batch_banner()
    
    # Log batch parameters
    processor.processor.log_print("\nBatch reprocessing with parameters:")
    processor.processor.log_print(f"Microscope: {batch_params.microscope}")
    processor.processor.log_print(f"Space Group: {batch_params.space_group}")
    processor.processor.log_print(f"Unit Cell Parameters:")
    processor.processor.log_print(f"  a = {batch_params.unit_cell_a}")
    processor.processor.log_print(f"  b = {batch_params.unit_cell_b}")
    processor.processor.log_print(f"  c = {batch_params.unit_cell_c}")
    processor.processor.log_print(f"  α = {batch_params.unit_cell_alpha}")
    processor.processor.log_print(f"  β = {batch_params.unit_cell_beta}")
    processor.processor.log_print(f"  γ = {batch_params.unit_cell_gamma}")
    processor.processor.log_print(f"Subfolder: {batch_params.subfolder}")

    # Log optional processing parameters if provided
    if any([batch_params.signal_pixel, batch_params.min_pixel, batch_params.background_pixel]):
        processor.processor.log_print("\nCustom processing parameters:")
        if batch_params.signal_pixel is not None:
            processor.processor.log_print(f"  Signal Pixel: {batch_params.signal_pixel}")
        if batch_params.min_pixel is not None:
            processor.processor.log_print(f"  Min Pixel: {batch_params.min_pixel}")
        if batch_params.background_pixel is not None:
            processor.processor.log_print(f"  Background Pixel: {batch_params.background_pixel}")
    processor.processor.log_print("")  # Add empty line for readability
    
    # Process directories
    processor.process_directory()

if __name__ == "__main__":
    main()
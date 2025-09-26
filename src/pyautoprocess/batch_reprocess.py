"""
Crystallography batch reprocessing script
Based on autoprocess.py by Jessica Burch, modified by Dmitry Eremin
Allows reprocessing of data with specific space group and unit cell parameters
"""
import os
import re
from pathlib import Path
import argparse
import shutil
from typing import Optional
from dataclasses import dataclass
from .autoprocess import CrystallographyProcessor
from .config.config_manager import ConfigLoader

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
    reindexing_matrix: Optional[list] = None

    @classmethod
    def from_input(cls, args: Optional[argparse.Namespace] = None) -> 'BatchParameters':
        """Create BatchParameters from command line args, smart detection, or interactive input."""
        # Smart mode: parameters will be detected automatically per directory
        if args and not args.manual:
            # Smart mode - parameters will be detected for each directory during processing
            # Return placeholder parameters that will be replaced per directory
            return cls(
                space_group=0,  # Placeholder
                unit_cell_a=0.0,  # Placeholder
                unit_cell_b=0.0,  # Placeholder
                unit_cell_c=0.0,  # Placeholder
                unit_cell_alpha=0.0,  # Placeholder
                unit_cell_beta=0.0,  # Placeholder
                unit_cell_gamma=0.0,  # Placeholder
                subfolder="smart",  # Placeholder
                reindexing_matrix=None  # Will be set per directory
            )

        def get_value(arg_value: Optional[str], prompt: str, convert_type=float) -> any:
            if arg_value is not None:
                return convert_type(arg_value)
            return convert_type(input(prompt).strip())
        
        # Get required parameters
        space_group = get_value(args.space_gr, 'Space group #? ', int)
        unit_cell_a = get_value(args.a, 'Unit cell a? ')
        unit_cell_b = get_value(args.b, 'Unit cell b? ')
        unit_cell_c = get_value(args.c, 'Unit cell c? ')
        unit_cell_alpha = get_value(args.alpha, 'Unit cell alpha? ')
        unit_cell_beta = get_value(args.beta, 'Unit cell beta? ')
        unit_cell_gamma = get_value(args.gamma, 'Unit cell gamma? ')
        subfolder = get_value(args.folder, 'Subfolder name? ', str)

        # Handle optional reindexing matrix from command line
        reindexing_matrix = None
        if args and hasattr(args, 'reidx') and args.reidx:
            try:
                # Parse reindexing matrix from space-separated string
                matrix_values = [float(x) for x in args.reidx.split()]
                if len(matrix_values) == 12:
                    reindexing_matrix = matrix_values
                else:
                    print(f"Warning: Reindexing matrix must have 12 values, got {len(matrix_values)}. Ignoring --reidx.")
            except ValueError:
                print(f"Warning: Invalid reindexing matrix format '{args.reidx}'. Ignoring --reidx.")

        return cls(
            space_group=space_group,
            unit_cell_a=unit_cell_a,
            unit_cell_b=unit_cell_b,
            unit_cell_c=unit_cell_c,
            unit_cell_alpha=unit_cell_alpha,
            unit_cell_beta=unit_cell_beta,
            unit_cell_gamma=unit_cell_gamma,
            subfolder=subfolder,
            reindexing_matrix=reindexing_matrix
        )

class BatchProcessor:
    def __init__(self, batch_params: BatchParameters, smart_mode: bool = False, target_space_group: Optional[int] = None, custom_folder: Optional[str] = None):
        self.params = batch_params
        self.smart_mode = smart_mode
        self._target_space_group = target_space_group
        self._custom_folder = custom_folder
        # Use default microscope parameters
        config_loader = ConfigLoader()
        microscope_params = config_loader.get_config('default')
        self.processor = CrystallographyProcessor(microscope_params)
        self.current_path = Path.cwd()

        # Fix empty bravais_data if needed
        if not self.processor.bravais_data:
            try:
                import json
                bravais_path = Path(__file__).parent / "data" / "bravais_lattices.json"
                if bravais_path.exists():
                    with open(bravais_path, 'r') as f:
                        self.processor.bravais_data = json.load(f)
            except Exception as e:
                self.processor.log_print(f"Warning: Could not load bravais_lattices.json: {e}")

        # Disable space group optimization in batch processing mode
        self.processor.disable_space_group_optimization = True

        # Override the _modify_xds_job method to preserve our JOB= CORRECT configuration
        self.processor._modify_xds_job = self._batch_modify_xds_job

    def _batch_modify_xds_job(self) -> None:
        """
        Batch processing version of _modify_xds_job that preserves JOB= CORRECT configuration.
        Unlike the original method that hardcodes JOB=DEFPIX INTEGRATE CORRECT,
        this preserves our carefully set JOB lines from _modify_xds_inp().
        """
        # Do nothing - preserve the existing JOB line configuration that we set in _modify_xds_inp
        pass

    def print_batch_banner(self, manual_mode: bool = True):
        """Print the program banner first, then batch processing banner."""
        # Use DisplayManager for consistent banner printing
        mode = "batch_manual" if manual_mode else "batch_smart"
        self.processor.display.print_full_banner(mode)

    def _detect_smart_parameters(self, process_dir: Path, target_space_group: Optional[int] = None, custom_folder: Optional[str] = None) -> Optional[BatchParameters]:
        """
        Smart mode: automatically detect optimal space group and unit cell parameters
        from existing CORRECT.LP file in the process directory.

        Args:
            process_dir: Directory containing CORRECT.LP file
            target_space_group: Optional specific space group to target
            custom_folder: Optional custom subfolder name (overrides auto-generated name)
        """
        correct_lp_path = process_dir / "CORRECT.LP"
        if not correct_lp_path.exists():
            self.processor.log_print(f"CORRECT.LP not found in {process_dir.name}")
            return None

        try:
            with open(correct_lp_path, 'r') as f:
                content = f.read()

            # Parse ALL lattice candidates (both starred and non-starred)
            candidates = self._parse_all_lattice_candidates(content)

            if not candidates:
                self.processor.log_print("No lattice candidates found in CORRECT.LP")
                return None

            selected_candidate = None
            selection_reason = ""

            if target_space_group is not None:
                # Look up Bravais lattice for the target space group
                target_bravais = self.processor.get_bravais_for_space_group(target_space_group)

                # Verify JSON lookup worked
                if not target_bravais:
                    self.processor.log_print(f"ERROR: Could not determine Bravais lattice for space group {target_space_group}")
                    return None

                # Filter candidates matching the target Bravais lattice
                matching_candidates = [c for c in candidates if c['bravais_lattice'] == target_bravais]

                if not matching_candidates:
                    self.processor.log_print(f"No candidates found for space group {target_space_group} (Bravais: {target_bravais})")
                    return None

                # Prioritize starred candidates
                starred_matches = [c for c in matching_candidates if c['starred']]

                if starred_matches:
                    if len(starred_matches) > 1:
                        # Multiple starred matches - use best fit with table order tiebreaker
                        selected_candidate = self._select_best_starred_candidate(starred_matches)
                        selection_reason = f"Best fit among {len(starred_matches)} starred matches for space group {target_space_group}"
                    else:
                        # Single starred match
                        selected_candidate = starred_matches[0]
                        selection_reason = f"Single starred match for space group {target_space_group}"
                else:
                    # No starred matches, use non-starred
                    if len(matching_candidates) > 1:
                        # Multiple non-starred matches - use lowest quality fit with table order tiebreaker
                        selected_candidate = self._select_best_non_starred_candidate(matching_candidates)
                        selection_reason = f"Least problematic among {len(matching_candidates)} non-starred matches for space group {target_space_group} (lowest quality value)"
                    else:
                        # Single non-starred match
                        selected_candidate = matching_candidates[0]
                        selection_reason = f"Single non-starred match for space group {target_space_group}"

                space_group = target_space_group
            else:
                # No target space group - find overall best starred candidate
                starred_candidates = [c for c in candidates if c['starred']]
                if not starred_candidates:
                    self.processor.log_print("No starred lattice candidates found in CORRECT.LP")
                    return None

                # Find best starred candidate (highest quality of fit with table order tiebreaker)
                selected_candidate = self._select_best_starred_candidate(starred_candidates)
                selection_reason = f"Best starred candidate (quality: {selected_candidate['quality_of_fit']})"

                # Get space group for best Bravais lattice
                space_group = self.processor.get_minimal_space_group_for_bravais(selected_candidate['bravais_lattice'])
                if not space_group:
                    self.processor.log_print(f"Could not determine space group for Bravais lattice {selected_candidate['bravais_lattice']}")
                    return None

            # Extract unit cell parameters and reindexing matrix
            a, b, c, alpha, beta, gamma = selected_candidate['unit_cell']
            reindexing_matrix = selected_candidate['reindexing_matrix']

            # Use custom folder if provided, otherwise generate based on Bravais lattice and space group
            if custom_folder:
                subfolder = custom_folder
            else:
                subfolder = f"{selected_candidate['bravais_lattice']}-{space_group}"

            # Format reindexing matrix for display (3x4 matrix as space-separated string)
            reidx_formatted = ' '.join(map(str, reindexing_matrix))

            self.processor.log_print(f"Smart mode selection:")
            self.processor.log_print(f"  Reason: {selection_reason}")
            self.processor.log_print(f"  Bravais lattice: {selected_candidate['bravais_lattice']}")
            self.processor.log_print(f"  Quality of fit: {selected_candidate['quality_of_fit']}")
            self.processor.log_print(f"  Space group: {space_group}")
            self.processor.log_print(f"  Unit cell: {a:.2f} {b:.2f} {c:.2f} {alpha:.1f} {beta:.1f} {gamma:.1f}")
            self.processor.log_print(f"  Reindexing matrix: {reidx_formatted}")
            self.processor.log_print(f"  Subfolder: {subfolder}")

            return BatchParameters(
                space_group=space_group,
                unit_cell_a=a,
                unit_cell_b=b,
                unit_cell_c=c,
                unit_cell_alpha=alpha,
                unit_cell_beta=beta,
                unit_cell_gamma=gamma,
                subfolder=subfolder,
                reindexing_matrix=reindexing_matrix
            )

        except Exception as e:
            self.processor.log_print(f"Error detecting smart parameters: {str(e)}")
            return None

    def _select_best_starred_candidate(self, starred_candidates):
        """
        Select best starred candidate using quality of fit with table order tiebreaker.
        For starred lines: later entries preferred (higher quality ranking by XDS).
        """
        # Find maximum quality of fit
        max_quality = max(c['quality_of_fit'] for c in starred_candidates)

        # Get all candidates with the maximum quality
        best_quality_candidates = [c for c in starred_candidates if c['quality_of_fit'] == max_quality]

        if len(best_quality_candidates) == 1:
            return best_quality_candidates[0]

        # Tiebreaker: for starred lines, prefer later entries (higher table_order)
        return max(best_quality_candidates, key=lambda c: c['table_order'])

    def _select_best_non_starred_candidate(self, non_starred_candidates):
        """
        Select best non-starred candidate using quality of fit with table order tiebreaker.
        For non-starred lines: earlier entries preferred (minimize problems).
        """
        # Find minimum quality of fit (least problematic)
        min_quality = min(c['quality_of_fit'] for c in non_starred_candidates)

        # Get all candidates with the minimum quality
        best_quality_candidates = [c for c in non_starred_candidates if c['quality_of_fit'] == min_quality]

        if len(best_quality_candidates) == 1:
            return best_quality_candidates[0]

        # Tiebreaker: for non-starred lines, prefer earlier entries (lower table_order)
        return min(best_quality_candidates, key=lambda c: c['table_order'])

    def _parse_all_lattice_candidates(self, content: str):
        """Parse both starred and non-starred lattice candidates from CORRECT.LP content"""
        candidates = []
        lines = content.split('\n')

        # Find lattice table
        table_start = -1
        for i, line in enumerate(lines):
            if "LATTICE-" in line and "BRAVAIS-" in line and "QUALITY" in line:
                table_start = i + 2  # Skip header and character/lattice line
                break

        if table_start == -1:
            return candidates

        # Parse all lines in the table
        table_order = 0  # Track table order for tiebreaker
        for i in range(table_start, len(lines)):
            line = lines[i].strip()
            if not line:
                continue

            # Stop if we hit a line that doesn't look like a lattice entry
            if not (line.startswith('*') or line.startswith(' ')):
                # Check if this could be a lattice line (has numbers and letters)
                if not any(c.isdigit() for c in line) or not any(c.isalpha() for c in line):
                    break

            try:
                # Parse line format: [*]  44        aP          0.0       5.2   10.4   21.0  90.3  90.5  90.0   -1  0  0...
                starred = line.startswith('*')

                # Remove * and split by whitespace
                clean_line = line.lstrip('* ')
                parts = [part for part in clean_line.split() if part]

                if len(parts) < 19:  # Need: number, lattice, quality, 6 unit cell, 12 reindexing
                    continue

                # parts[0] = number, parts[1] = lattice type, parts[2] = quality
                bravais_lattice = parts[1]  # aP, mP, etc.
                quality_of_fit = float(parts[2])

                # Unit cell parameters: a, b, c, alpha, beta, gamma (parts[3] through parts[8])
                unit_cell = [float(parts[i]) for i in range(3, 9)]

                # Reindexing matrix: 12 values (parts[9] through parts[20])
                reindexing_matrix = [int(parts[i]) for i in range(9, 21)]

                candidate = {
                    'starred': starred,
                    'bravais_lattice': bravais_lattice,
                    'quality_of_fit': quality_of_fit,
                    'unit_cell': unit_cell,
                    'reindexing_matrix': reindexing_matrix,
                    'table_order': table_order
                }
                candidates.append(candidate)
                table_order += 1  # Increment for next candidate

            except (ValueError, IndexError):
                # Skip malformed lines
                continue

        return candidates

    def _copy_xds_files(self, source_dir: Path, target_dir: Path) -> None:
        """Copy all necessary XDS files from source to target directory."""
        # Essential files for JOB=CORRECT processing
        essential_files = [
            "XDS.INP",           # Input parameters
            "INTEGRATE.HKL",     # Integrated reflections (required for CORRECT)
            "XPARM.XDS",         # Crystal parameters (required for CORRECT)
        ]

        # Optional but commonly needed files (based on actual XDS output)
        optional_files = [
            # XDS log files
            "DEFPIX.LP",         # Defective pixel information
            "CORRECT.LP",        # Previous correction results
            "INTEGRATE.LP",      # Integration log
            "IDXREF.LP",         # Indexing and refinement log
            "COLSPOT.LP",        # Spot finding log
            "INIT.LP",           # Initialization log
            "XYCORR.LP",         # XY correction log

            # Crystal parameters and calibration
            "GXPARM.XDS",        # Global crystal parameters
            "SPOT.XDS",          # Spot positions

            # Detector calibration and correction files
            "GAIN.cbf",          # Gain map
            "BKGPIX.cbf",        # Background pixel mask
            "BKGINIT.cbf",       # Initial background
            "X-CORRECTIONS.cbf", # X-ray corrections
            "Y-CORRECTIONS.cbf", # Y-ray corrections
            "DX-CORRECTIONS.cbf",# Delta-X corrections
            "DY-CORRECTIONS.cbf",# Delta-Y corrections
            "GX-CORRECTIONS.cbf",# Global X corrections
            "GY-CORRECTIONS.cbf",# Global Y corrections

            # Additional calibration files
            "ABS.cbf",           # Absorption correction
            "ABSORP.cbf",        # Absorption parameters
            "DECAY.cbf",         # Decay correction
            "MODPIX.cbf",        # Modified pixel mask
            "BLANK.cbf",         # Blank image

            # Output files (may be overwritten but good to preserve)
            "XDS_ASCII.HKL",     # Final output reflections

            # Visualization files (optional)
            "SHOW_BKG.cbf",      # Background visualization
            "SHOW_HKL.cbf",      # HKL visualization
            "SHOW_SPOT.cbf",     # Spot visualization

            # Custom processing files
            "stats.LP",          # Processing statistics
            "xds_ap.LP",         # XDS terminal output log
        ]

        copied_files = []
        missing_essential = []

        # Copy essential files (must exist)
        for filename in essential_files:
            source_file = source_dir / filename
            if source_file.is_file():
                try:
                    shutil.copy2(source_file, target_dir)
                    copied_files.append(filename)
                except Exception as e:
                    self.processor.log_print(f"Error copying {filename}: {str(e)}")
            else:
                missing_essential.append(filename)

        # Copy optional files (if they exist)
        for filename in optional_files:
            source_file = source_dir / filename
            if source_file.is_file():
                try:
                    shutil.copy2(source_file, target_dir)
                    copied_files.append(filename)
                except Exception as e:
                    self.processor.log_print(f"Warning: Could not copy optional file {filename}: {str(e)}")

        # Report results
        if missing_essential:
            raise FileNotFoundError(f"Missing essential XDS files: {', '.join(missing_essential)}")

        # Determine source directory name (auto_process or auto_process_direct)
        source_dir_name = source_dir.name
        self.processor.log_print(f"Copied XDS files from {source_dir_name}")

    def _modify_xds_inp(self, xds_path: Path, params: Optional[BatchParameters] = None) -> None:
        """Modify XDS.INP with batch processing parameters for CORRECT-only processing."""
        if params is None:
            params = self.params
        try:
            with open(xds_path, 'r') as f:
                lines = f.readlines()

            modified_lines = []

            # Check if this is new format (has JOB= CORRECT line) or old format
            has_correct_job = any('JOB= CORRECT' in line or 'JOB=CORRECT' in line for line in lines)

            if has_correct_job:
                # NEW FORMAT: Handle existing JOB lines by commenting/uncommenting appropriately
                params_processed = {'space_group': False, 'unit_cell': False, 'reidx': False, 'signal_pixel': False, 'min_pixel': False, 'background_pixel': False}

                for i, line in enumerate(lines):
                    stripped = line.strip()

                    # Handle JOB lines: We want only "JOB= CORRECT" active, everything else commented
                    if 'JOB=' in stripped:
                        # Remove existing comment character to normalize the line for comparison
                        clean_line = line.lstrip()
                        if clean_line.startswith('!'):
                            clean_line = clean_line[1:]

                        # Check if this is exactly the "JOB= CORRECT" line (with possible spaces)
                        clean_stripped = clean_line.strip()

                        if clean_stripped == 'JOB= CORRECT' or clean_stripped == 'JOB=CORRECT':
                            # This is our target line - make sure it's active (uncommented)
                            modified_lines.append(clean_line)
                        else:
                            # This is any other JOB line - make sure it's commented out
                            # Always add ! prefix for non-target JOB lines
                            original_line_stripped = line.strip()
                            if original_line_stripped.startswith('!'):
                                # Already commented
                                modified_lines.append(line)
                            else:
                                # Add comment
                                new_line = f"!{line}"
                                modified_lines.append(new_line)
                        continue

                    # Handle space group parameter (replace first occurrence only)
                    elif stripped.startswith(('!SPACE_GROUP_NUMBER=', 'SPACE_GROUP_NUMBER=')) and not params_processed['space_group']:
                        modified_lines.append(f"SPACE_GROUP_NUMBER= {params.space_group}\n")
                        params_processed['space_group'] = True
                    elif stripped.startswith(('!UNIT_CELL_CONSTANTS=', 'UNIT_CELL_CONSTANTS=')) and not params_processed['unit_cell']:
                        modified_lines.append(
                            f"UNIT_CELL_CONSTANTS= {params.unit_cell_a} {params.unit_cell_b} "
                            f"{params.unit_cell_c} {params.unit_cell_alpha} "
                            f"{params.unit_cell_beta} {params.unit_cell_gamma}\n"
                        )
                        params_processed['unit_cell'] = True
                    elif stripped.startswith(('!REIDX=', 'REIDX=')) and params.reindexing_matrix is not None and not params_processed['reidx']:
                        # Format reindexing matrix as space-separated string
                        reidx_formatted = ' '.join(map(str, params.reindexing_matrix))
                        modified_lines.append(f"REIDX= {reidx_formatted}\n")
                        params_processed['reidx'] = True

                    # Pixel parameters not modified for JOB=CORRECT batch reprocessing
                    else:
                        modified_lines.append(line)
            else:
                # OLD FORMAT: Comment out existing JOB lines and insert new JOB= CORRECT at line 3
                job_lines_found = 0

                for i, line in enumerate(lines):
                    stripped = line.strip()

                    # Comment out any existing JOB lines
                    if stripped.startswith('JOB='):
                        if not stripped.startswith('!'):
                            modified_lines.append(f"!{line}")
                        else:
                            modified_lines.append(line)
                        job_lines_found += 1
                    else:
                        modified_lines.append(line)

                    # Insert JOB= CORRECT as line 3 (after first 2 lines)
                    if i == 1:  # After line 2 (0-indexed)
                        modified_lines.append("JOB= CORRECT\n")

                # Now handle space group and unit cell parameters in the modified lines
                final_lines = []
                params_processed = {'space_group': False, 'unit_cell': False, 'reidx': False, 'signal_pixel': False, 'min_pixel': False, 'background_pixel': False}

                for line in modified_lines:
                    stripped = line.strip()

                    if stripped.startswith(('!SPACE_GROUP_NUMBER=', 'SPACE_GROUP_NUMBER=')) and not params_processed['space_group']:
                        final_lines.append(f"SPACE_GROUP_NUMBER= {params.space_group}\n")
                        params_processed['space_group'] = True
                    elif stripped.startswith(('!UNIT_CELL_CONSTANTS=', 'UNIT_CELL_CONSTANTS=')) and not params_processed['unit_cell']:
                        final_lines.append(
                            f"UNIT_CELL_CONSTANTS= {params.unit_cell_a} {params.unit_cell_b} "
                            f"{params.unit_cell_c} {params.unit_cell_alpha} "
                            f"{params.unit_cell_beta} {params.unit_cell_gamma}\n"
                        )
                        params_processed['unit_cell'] = True
                    elif stripped.startswith(('!REIDX=', 'REIDX=')) and params.reindexing_matrix is not None and not params_processed['reidx']:
                        # Format reindexing matrix as space-separated string
                        reidx_formatted = ' '.join(map(str, params.reindexing_matrix))
                        final_lines.append(f"REIDX= {reidx_formatted}\n")
                        params_processed['reidx'] = True
                    # Pixel parameters not modified for JOB=CORRECT batch reprocessing
                    else:
                        final_lines.append(line)

                modified_lines = final_lines

            with open(xds_path, 'w') as f:
                f.writelines(modified_lines)

            self.processor.log_print("Updated XDS.INP for CORRECT-only processing with batch parameters")

        except Exception as e:
            self.processor.log_print(f"Error modifying XDS.INP: {str(e)}")
            raise
    
    def _extract_processing_parameters(self, xds_path: Path) -> dict:
        """Extract processing parameters from XDS.INP file."""
        parameters = {}
        try:
            with open(xds_path, 'r') as f:
                content = f.read()
                
            # Extract parameters using regular expressions
            
            # Get detector distance
            distance_match = re.search(r'DETECTOR_DISTANCE=\s*([-+]?\d*\.?\d+)', content)
            if distance_match:
                parameters['distance'] = distance_match.group(1)
                
            # Get oscillation range
            osc_match = re.search(r'OSCILLATION_RANGE=\s*([-+]?\d*\.?\d+)', content)
            if osc_match:
                parameters['oscillation'] = float(osc_match.group(1))
                
            # Get resolution ranges
            res_match = re.search(r'INCLUDE_RESOLUTION_RANGE=\s*\d+\s+([-+]?\d*\.?\d+)', content)
            test_res_match = re.search(r'TEST_RESOLUTION_RANGE=\s*\d+\s+([-+]?\d*\.?\d+)', content)
            
            if res_match:
                parameters['resolution_range'] = float(res_match.group(1))
            if test_res_match:
                parameters['test_resolution_range'] = float(test_res_match.group(1))
                
        except Exception as e:
            self.processor.log_print(f"Error extracting parameters from XDS.INP: {str(e)}")
            return {}
            
        return parameters

    def _log_processing_parameters(self, name: str, parameters: dict) -> None:
        """Log processing parameters."""
        self.processor.log_print(f"\nProcessing parameters for {name}:")
        
        if 'distance' in parameters:
            self.processor.log_print(f"Detector Distance: {parameters['distance']} mm")
        if 'oscillation' in parameters:
            self.processor.log_print(f"Oscillation Range: {parameters['oscillation']} deg")
        if 'resolution_range' in parameters:
            self.processor.log_print(f"Resolution Range: {parameters['resolution_range']} Å")
        if 'test_resolution_range' in parameters:
            self.processor.log_print(f"Test Resolution Range: {parameters['test_resolution_range']} Å\n")

    def process_directory(self, path: Optional[Path] = None) -> None:
        """Process directories at the specified path or current working directory.

        Args:
            path: Path to process. Can be:
                - None: Process current working directory (default behavior)
                - Path to a single auto_process/auto_process_direct directory
                - Path to a parent directory containing multiple datasets
        """
        processed_count = 0
        skipped_count = 0
        error_count = 0

        # Determine the target path
        if path is None:
            target_path = self.current_path
        else:
            target_path = Path(path).resolve()

        # Check if the path is a single auto_process directory
        if self._is_autoprocess_directory(target_path):
            # Process single auto_process directory directly
            try:
                if self.smart_mode:
                    # Find the auto_process directory within the dataset directory
                    process_dir = self._find_autoprocess_directory(target_path)
                    if process_dir is None:
                        self.processor.log_print(f"No auto_process directory found in {target_path.name}")
                        skipped_count += 1
                        return

                    # Check if a target space group was provided via command line
                    target_sg = None
                    if hasattr(self, '_target_space_group') and self._target_space_group:
                        target_sg = self._target_space_group

                    smart_params = self._detect_smart_parameters(process_dir, target_sg, self._custom_folder)
                    if smart_params is None:
                        self.processor.log_print(f"Failed to detect smart parameters for {target_path.name}, skipping")
                        skipped_count += 1
                        return
                    current_params = smart_params
                else:
                    current_params = self.params

                # Process the single directory
                self._process_single_dataset(target_path, current_params, target_path.name)
                processed_count += 1
            except Exception as e:
                self.processor.log_print(f"Error processing {target_path.parent.name}: {e}")
                error_count += 1
        else:
            # Process as parent directory containing multiple datasets
            if not target_path.exists():
                self.processor.log_print(f"Error: Path does not exist: {target_path}")
                return

            if not target_path.is_dir():
                self.processor.log_print(f"Error: Path is not a directory: {target_path}")
                return

            dir_list = os.listdir(target_path)

            for name in dir_list:
                dir_path = target_path / name
                if not dir_path.is_dir():
                    continue

                # Check for either auto_process or auto_process_direct
                process_dir = self._find_autoprocess_directory(dir_path)
                if process_dir is None:
                    self.processor.log_print(f"Skipping {name}: No auto_process_direct or auto_process directory with XDS.INP")
                    skipped_count += 1
                    continue

                try:
                    # Smart mode: detect parameters for this specific directory
                    if self.smart_mode:
                        # Check if a target space group was provided via command line
                        target_sg = None
                        if hasattr(self, '_target_space_group') and self._target_space_group:
                            target_sg = self._target_space_group

                        smart_params = self._detect_smart_parameters(process_dir, target_sg, self._custom_folder)
                        if not smart_params:
                            self.processor.log_print(f"Failed to detect smart parameters for {name}, skipping")
                            skipped_count += 1
                            continue
                        # Use detected parameters for this directory
                        current_params = smart_params
                    else:
                        # Manual mode: use provided parameters
                        current_params = self.params

                    # Create or clean subfolder
                    subfolder_path = dir_path / current_params.subfolder
                    if subfolder_path.exists():
                        self.processor.log_print(f"\nReprocessing {name} in existing subfolder {current_params.subfolder}")
                        shutil.rmtree(subfolder_path)
                    else:
                        self.processor.log_print(f"\nProcessing {name} in new subfolder {current_params.subfolder}")

                    subfolder_path.mkdir(parents=True)

                    # Extract and log processing parameters before copying files
                    parameters = self._extract_processing_parameters(process_dir / "XDS.INP")
                    self._log_processing_parameters(name, parameters)

                    # Change to subfolder directory and continue processing
                    original_dir = os.getcwd()

                    try:
                        os.chdir(subfolder_path)

                        # Copy all necessary XDS files and modify XDS.INP
                        self._copy_xds_files(process_dir, subfolder_path)
                        self._modify_xds_inp(subfolder_path / "XDS.INP", current_params)

                        # Process the data
                        self.processor.log_print(f"\nProcessing {name}...")

                        # Run XDS CORRECT and save terminal output to both xds_ap.LP and XDS.LP
                        # This ensures compatibility with both space group processing (xds_ap.LP) and regular processing (XDS.LP)
                        try:
                            with open("xds_ap.LP", "w") as output_file:
                                self.processor._run_xds_command(output_file)

                            # Copy xds_ap.LP to XDS.LP so process_check() can find the ISa values
                            shutil.copy2("xds_ap.LP", "XDS.LP")
                            self.processor.log_print("XDS CORRECT processing completed")
                        except Exception as xds_error:
                            raise Exception(f"XDS CORRECT execution failed: {xds_error}")

                        # Use the standard process_check method now that XDS.LP exists
                        self.processor.process_check(name)

                        processed_count += 1
                        self.processor.log_print(f"Successfully processed {name}")

                    finally:
                        # Always restore original directory
                        os.chdir(original_dir)

                except Exception as e:
                    self.processor.log_print(f"Error processing {name}: {str(e)}")
                    error_count += 1

        # Print summary
        self.processor.log_print("\nProcessing Summary:")
        self.processor.log_print(f"Successfully processed: {processed_count}")
        self.processor.log_print(f"Skipped: {skipped_count}")
        self.processor.log_print(f"Errors: {error_count}")

    def _find_autoprocess_directory(self, parent_dir: Path) -> Optional[Path]:
        """Find auto_process or auto_process_direct directory within parent directory."""
        for process_dirname in ["auto_process_direct", "auto_process"]:
            potential_dir = parent_dir / process_dirname
            if potential_dir.exists() and (potential_dir / "XDS.INP").exists():
                return potential_dir
        return None

    def _is_autoprocess_directory(self, path: Path) -> bool:
        """Check if the given path contains an auto_process or auto_process_direct directory."""
        return self._find_autoprocess_directory(path) is not None

    def _process_single_dataset(self, dataset_dir: Path, current_params: BatchParameters, dataset_name: str) -> None:
        """Process a single dataset directory that contains auto_process/auto_process_direct."""
        # Find the auto_process directory within the dataset directory
        process_dir = self._find_autoprocess_directory(dataset_dir)
        if process_dir is None:
            raise Exception(f"No auto_process directory found in {dataset_name}")

        # Create or clean subfolder in the dataset directory
        subfolder_path = dataset_dir / current_params.subfolder

        if subfolder_path.exists():
            self.processor.log_print(f"\nReprocessing {dataset_name} in existing subfolder {current_params.subfolder}")
            shutil.rmtree(subfolder_path)
        else:
            self.processor.log_print(f"\nProcessing {dataset_name} in new subfolder {current_params.subfolder}")

        subfolder_path.mkdir(parents=True)

        # Extract and log processing parameters before copying files
        parameters = self._extract_processing_parameters(process_dir / "XDS.INP")
        self._log_processing_parameters(dataset_name, parameters)

        # Change to subfolder directory and continue processing
        original_dir = os.getcwd()

        try:
            os.chdir(subfolder_path)

            # Copy all necessary XDS files and modify XDS.INP
            self._copy_xds_files(process_dir, subfolder_path)
            self._modify_xds_inp(subfolder_path / "XDS.INP", current_params)

            # Process the data
            self.processor.log_print(f"\nProcessing {dataset_name}...")

            # Run XDS CORRECT and save terminal output to both xds_ap.LP and XDS.LP
            # This ensures compatibility with both space group processing (xds_ap.LP) and regular processing (XDS.LP)
            try:
                with open("xds_ap.LP", "w") as output_file:
                    self.processor._run_xds_command(output_file)

                # Copy xds_ap.LP to XDS.LP so process_check() can find the ISa values
                shutil.copy2("xds_ap.LP", "XDS.LP")
                self.processor.log_print("XDS CORRECT processing completed")
            except Exception as xds_error:
                raise Exception(f"XDS CORRECT execution failed: {xds_error}")

            # Use the standard process_check method now that XDS.LP exists
            self.processor.process_check(dataset_name)

            self.processor.log_print(f"Successfully processed {dataset_name}")

        finally:
            # Always restore original directory
            os.chdir(original_dir)

def parse_arguments() -> Optional[argparse.Namespace]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Batch reprocess crystallography data with specific space group and unit cell parameters.'
    )

    # Mode selection
    parser.add_argument('--manual', action='store_true', default=True,
                       help='Manual mode: prompt for parameters interactively (default: True)')
    parser.add_argument('--smart', dest='manual', action='store_false',
                       help='Smart mode: automated parameter detection and processing')
    
    # Unit cell and space group parameters
    parser.add_argument('--space-gr', help='Space group number')
    parser.add_argument('--a', help='Unit cell parameter a')
    parser.add_argument('--b', help='Unit cell parameter b')
    parser.add_argument('--c', help='Unit cell parameter c')
    parser.add_argument('--alpha', help='Unit cell angle alpha')
    parser.add_argument('--beta', help='Unit cell angle beta')
    parser.add_argument('--gamma', help='Unit cell angle gamma')
    parser.add_argument('--folder', help='Name of subfolder for reprocessed data')
    parser.add_argument('--reidx', help='Reindexing matrix as space-separated string (e.g., "1 0 0 0 0 1 0 0 0 0 1 0")')

    # Positional arguments for paths (consistent with autoprocess.py and image_process.py)
    parser.add_argument('paths', nargs='*',
                       help='Path(s) to process: folders containing auto_process/auto_process_direct directories, or single auto_process directory. If not specified, processes current directory.')

    args = parser.parse_args()
    
    return args

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Get parameters from args or interactive input
    batch_params = BatchParameters.from_input(args)
    
    # Initialize processor
    target_space_group = int(args.space_gr) if args.space_gr else None
    custom_folder = args.folder if hasattr(args, 'folder') and args.folder else None
    processor = BatchProcessor(batch_params, smart_mode=not args.manual, target_space_group=target_space_group, custom_folder=custom_folder)
    
    # Setup logging exactly as in autoprocess.py
    processor.processor.setup_logging("batch_reprocess.log", "autoprocess_logs")
    
    # Print banners
    processor.print_batch_banner(args.manual)
    
    # Log batch parameters (only for manual mode)
    if args.manual:
        processor.processor.log_print("\nBatch reprocessing with parameters:")
        processor.processor.log_print(f"Space Group: {batch_params.space_group}")
        processor.processor.log_print(f"Unit Cell Parameters:")
        processor.processor.log_print(f"  a = {batch_params.unit_cell_a}")
        processor.processor.log_print(f"  b = {batch_params.unit_cell_b}")
        processor.processor.log_print(f"  c = {batch_params.unit_cell_c}")
        processor.processor.log_print(f"  α = {batch_params.unit_cell_alpha}")
        processor.processor.log_print(f"  β = {batch_params.unit_cell_beta}")
        processor.processor.log_print(f"  γ = {batch_params.unit_cell_gamma}")
        processor.processor.log_print(f"Subfolder: {batch_params.subfolder}")

        # Log reindexing matrix if provided
        if batch_params.reindexing_matrix:
            reidx_formatted = ' '.join(map(str, batch_params.reindexing_matrix))
            processor.processor.log_print(f"Reindexing matrix: {reidx_formatted}")

        # Pixel parameters not used in batch reprocessing (JOB=CORRECT only)
    else:
        processor.processor.log_print("\nSmart mode: parameters will be detected automatically for each dataset")

    processor.processor.log_print("")  # Add empty line for readability

    # Process directories using specified paths or current directory
    if args.paths:
        for path in args.paths:
            processor.process_directory(Path(path))
    else:
        # Default behavior: process current directory
        processor.process_directory()

if __name__ == "__main__":
    main()
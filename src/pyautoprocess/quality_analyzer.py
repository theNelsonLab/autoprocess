"""
Diffraction quality analysis module for autoprocess
Adapted from the Jupyter notebook diffraction_quality.ipynb
Uses mrcfile instead of hyperspy for MRC file reading
"""
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .tvips import TvipsReader
import mrcfile
import numpy as np
import seremi
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops_table

@dataclass
class DiffractionResult:
    """Result of diffraction quality analysis for a single frame"""
    frame_number: int
    n_diffraction_spots: int
    n_pattern_spots: int
    total_sum: float
    dqi: float
    quality: str

@dataclass
class QualityParameters:
    """Parameters for diffraction quality analysis"""
    light_sigma_prim: float = 1.0        # Spot enhancement
    harsh_sigma_prim: float = 5.0        # Background subtraction
    threshold_std_prim: float = 15.0     # CRITICAL: Spot detection sensitivity
    min_pixels_prim: int = 3             # Minimum spot size
    exclude_center: float = 25.0         # Central beam exclusion (%)
    harsh_sigma_ft: float = 20.0         # FT processing
    threshold_std_ft: float = 3.0        # FT threshold
    min_pixels_ft: int = 1               # FT min pixels
    good_rule: float = 1.0               # DQI threshold for quality
    grid_rule: int = 3                   # Grid detection (added for completeness)

class DiffractionQualityAnalyzer:
    """Analyzes diffraction quality of MRC movies for autoprocess integration"""

    def __init__(self, params: Optional[QualityParameters] = None):
        self.params = params or QualityParameters()
        self.results: Dict[int, DiffractionResult] = {}
        self._conversion_logged = False

    def parse_mrc(self, mrc_file: str) -> Tuple[np.ndarray, bool]:
        """Parse MRC file using mrcfile library"""
        with mrcfile.open(mrc_file, mode='r') as mrc:
            data = mrc.data.copy()

        is_multiframe = len(data.shape) == 3

        if is_multiframe:
            logging.info(f"Detected multiframe MRC with {data.shape[0]} frames")

        # Convert integer types to float to prevent overflow
        if data.dtype in [np.uint8, np.uint16, np.uint32]:
            if not self._conversion_logged:
                logging.info(f"Converting {data.dtype} to float64 to prevent overflow")
                self._conversion_logged = True
            data = data.astype(np.float64)

        return data, is_multiframe

    def parse_ser(self, ser_file: str) -> Tuple[np.ndarray, bool]:
        """Parse SER file using seremi library"""
        with seremi.SERFile(ser_file) as ser:
            frames = ser.read_all_frames()

        # Convert list of frames to numpy array
        data = np.array(frames)
        is_multiframe = len(data.shape) == 3 and data.shape[0] > 1

        if is_multiframe:
            logging.info(f"Detected multiframe SER with {data.shape[0]} frames")
        else:
            logging.info("Detected single frame SER")

        # Convert integer types to float to prevent overflow
        if data.dtype in [np.uint8, np.uint16, np.uint32]:
            if not self._conversion_logged:
                logging.info(f"Converting {data.dtype} to float64 to prevent overflow")
                self._conversion_logged = True
            data = data.astype(np.float64)

        return data, is_multiframe
    
    def parse_tvips(self, tvips_file: str) -> Tuple[np.ndarray, bool]:
        """Parse tvips with custom reader"""
        with TvipsReader(tvips_file) as tvips:
            frames = tvips.read_all_frames()
            if len(frames) == 1:
                logging.info(f'Detected single frame TVIPS')
                return frames[0], False
            else:
                logging.info(f'Detected multi-frame TVIPS with {frames.shape[0]} frames')
                return np.array(frames), True


    def parse_file(self, file_path: str) -> Tuple[np.ndarray, bool]:
        """Parse file (MRC or SER) based on file extension"""
        file_path_obj = Path(file_path)

        if file_path_obj.suffix.lower() == '.mrc':
            return self.parse_mrc(file_path)
        elif file_path_obj.suffix.lower() == '.ser':
            return self.parse_ser(file_path)
        elif file_path_obj.suffix.lower() == '.tvips':
            return self.parse_tvips(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path_obj.suffix}. Only .mrc and .ser files are supported.")

    def create_binary_image(self, data: np.ndarray, light_sigma: float, harsh_sigma: float,
                          threshold_std: float, ft_done: bool = False) -> np.ndarray:
        """Create binary image for spot detection"""
        if data.dtype in [np.uint8, np.uint16, np.uint32]:
            data = data.astype(np.float64)

        if ft_done:
            harsh_blur_data = gaussian_filter(data, sigma=self.params.harsh_sigma_ft)
            difference_data = data - harsh_blur_data
            threshold_value = np.mean(difference_data) + (self.params.threshold_std_ft * np.std(difference_data))
        else:
            light_blur_data = gaussian_filter(data, sigma=light_sigma)
            harsh_blur_data = gaussian_filter(data, sigma=harsh_sigma)
            difference_data = light_blur_data - harsh_blur_data
            threshold_value = np.mean(difference_data) + (threshold_std * np.std(difference_data))

        binary_image = difference_data > threshold_value

        # Mask edges (3% of frame size)
        mask_size_x = int(binary_image.shape[1] * 0.03)
        mask_size_y = int(binary_image.shape[0] * 0.03)
        binary_image[:, :mask_size_x] = False
        binary_image[:, -mask_size_x:] = False
        binary_image[:mask_size_y, :] = False
        binary_image[-mask_size_y:, :] = False

        return binary_image

    def get_binary_ft(self, data: np.ndarray) -> np.ndarray:
        """Get Fourier transform magnitude"""
        ft_data = np.fft.fftshift(np.fft.fft2(data))
        ft_magnitude = np.log(np.abs(ft_data) + 1)
        return ft_magnitude

    def get_centroids(self, binary_image: np.ndarray, min_pixels: int,
                     exclude_center: Optional[float] = None, ft_done: bool = False) -> Tuple[List[Tuple[float, float]], int]:
        """Extract centroids from binary image"""
        labeled_array = label(binary_image)
        properties = regionprops_table(labeled_array, properties=('centroid', 'area'))

        min_pixels_to_use = self.params.min_pixels_ft if ft_done else min_pixels
        centroids = [(x, y) for x, y, area in zip(properties['centroid-0'], properties['centroid-1'],
                                                 properties['area']) if area >= min_pixels_to_use]

        if not ft_done and exclude_center is not None:
            image_shape = binary_image.shape
            center_x, center_y = image_shape[1] // 2, image_shape[0] // 2
            min_distance = min(center_x, center_y)
            exclude_radius = (exclude_center / 100) * min_distance

            filtered_centroids = [centroid for centroid in centroids
                                if not (center_x - exclude_radius < centroid[1] < center_x + exclude_radius and
                                        center_y - exclude_radius < centroid[0] < center_y + exclude_radius)]
        else:
            filtered_centroids = centroids

        return filtered_centroids, len(filtered_centroids)

    def _classify_diffraction(self, n_dif_spots: int, n_pat_spots: int) -> str:
        """Classify diffraction quality"""
        if n_dif_spots < 3:
            return 'No diffraction'
        elif n_dif_spots < 10:
            return 'Poor diffraction'
        elif self.params.good_rule * n_dif_spots > n_pat_spots:
            return 'Bad diffraction'
        else:
            return 'Good diffraction'

    def process_single_frame(self, data: np.ndarray, frame_number: int) -> Optional[DiffractionResult]:
        """Process a single frame for diffraction quality"""
        try:
            total_sum = np.sum(data)

            # Primary binary image for diffraction spot detection
            primary_binary_image = self.create_binary_image(
                data,
                self.params.light_sigma_prim,
                self.params.harsh_sigma_prim,
                self.params.threshold_std_prim,
                ft_done=False
            )

            # Get diffraction spots
            primary_centroids, n_dif_spots = self.get_centroids(
                primary_binary_image,
                self.params.min_pixels_prim,
                exclude_center=self.params.exclude_center,
                ft_done=False
            )

            # Fourier transform analysis for pattern quality
            ft_binary_image = self.get_binary_ft(primary_binary_image)
            secondary_binary_image = self.create_binary_image(
                ft_binary_image,
                None,
                self.params.harsh_sigma_ft,
                self.params.threshold_std_ft,
                ft_done=True
            )

            # Get pattern spots
            secondary_centroids, n_pat_spots = self.get_centroids(
                secondary_binary_image,
                self.params.min_pixels_ft,
                exclude_center=None,
                ft_done=True
            )

            # Calculate DQI and quality
            dqi = n_pat_spots / n_dif_spots if n_dif_spots > 0 else 0
            quality = self._classify_diffraction(n_dif_spots, n_pat_spots)

            result = DiffractionResult(
                frame_number=frame_number,
                n_diffraction_spots=n_dif_spots,
                n_pattern_spots=n_pat_spots,
                total_sum=total_sum,
                dqi=dqi,
                quality=quality
            )

            self.results[frame_number] = result
            return result

        except Exception as e:
            logging.error(f"Error processing frame {frame_number}: {str(e)}")
            return None

    def analyze_file(self, file_path: str) -> List[DiffractionResult]:
        """Analyze MRC or SER file and return quality results for all frames"""
        file_name = os.path.basename(file_path)
        file_type = Path(file_path).suffix.upper()
        logging.info(f"Starting diffraction quality analysis: {file_name} ({file_type})")

        # Log analysis parameters
        logging.info("Quality analysis parameters:")
        logging.info(f"  Light sigma: {self.params.light_sigma_prim}")
        logging.info(f"  Harsh sigma: {self.params.harsh_sigma_prim}")
        logging.info(f"  Threshold std: {self.params.threshold_std_prim}")
        logging.info(f"  Min pixels: {self.params.min_pixels_prim}")
        logging.info(f"  Exclude center: {self.params.exclude_center}%")
        logging.info(f"  FT harsh sigma: {self.params.harsh_sigma_ft}")
        logging.info(f"  FT threshold std: {self.params.threshold_std_ft}")
        logging.info(f"  DQI threshold: {self.params.good_rule}")

        try:
            data, is_multiframe = self.parse_file(file_path)
            results = []

            if is_multiframe:
                num_frames = data.shape[0]
                logging.info(f"Processing {num_frames} frames for quality analysis")

                for frame_idx in range(num_frames):
                    result = self.process_single_frame(data[frame_idx], frame_idx)
                    if result is not None:
                        results.append(result)

                logging.info(f"Completed quality analysis: {len(results)}/{num_frames} frames processed")
            else:
                result = self.process_single_frame(data, 0)
                if result is not None:
                    results.append(result)

            return results

        except Exception as e:
            logging.error(f"Error analyzing {file_name}: {str(e)}")
            return []

    def analyze_data(self, data: np.ndarray, filename: str) -> List[DiffractionResult]:
        """Analyze data directly without file reading - optimized for single-read workflow"""
        file_name = os.path.basename(filename)
        file_type = Path(filename).suffix.upper()
        logging.info(f"Starting diffraction quality analysis: {file_name} ({file_type}) - in-memory")

        # Log analysis parameters
        logging.info("Quality analysis parameters:")
        logging.info(f"  Light sigma: {self.params.light_sigma_prim}")
        logging.info(f"  Harsh sigma: {self.params.harsh_sigma_prim}")
        logging.info(f"  Threshold std: {self.params.threshold_std_prim}")
        logging.info(f"  Min pixels: {self.params.min_pixels_prim}")
        logging.info(f"  Exclude center: {self.params.exclude_center}%")
        logging.info(f"  FT harsh sigma: {self.params.harsh_sigma_ft}")
        logging.info(f"  FT threshold std: {self.params.threshold_std_ft}")
        logging.info(f"  DQI threshold: {self.params.good_rule}")

        try:
            # Convert integer types to float to prevent overflow
            if data.dtype in [np.uint8, np.uint16, np.uint32]:
                if not self._conversion_logged:
                    logging.info(f"Converting {data.dtype} to float64 to prevent overflow")
                    self._conversion_logged = True
                data = data.astype(np.float64)

            is_multiframe = len(data.shape) == 3 and data.shape[0] > 1
            results = []

            if is_multiframe:
                num_frames = data.shape[0]
                logging.info(f"Processing {num_frames} frames for quality analysis")

                for frame_idx in range(num_frames):
                    result = self.process_single_frame(data[frame_idx], frame_idx)
                    if result is not None:
                        results.append(result)

                logging.info(f"Completed quality analysis: {len(results)}/{num_frames} frames processed")
            else:
                result = self.process_single_frame(data, 0)
                if result is not None:
                    results.append(result)

            return results

        except Exception as e:
            logging.error(f"Error analyzing {file_name}: {str(e)}")
            return []

    def analyze_mrc_file(self, file_path: str) -> List[DiffractionResult]:
        """Backward compatibility method - redirects to analyze_file"""
        return self.analyze_file(file_path)

    def find_good_frame_range(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Find the frame range where diffraction quality is good.

        Returns start and end frame numbers (1-indexed for XDS compatibility).
        Start: First frame where two consecutive frames have DQI > 1 OR one frame has LQP > 100
        End: Last frame before quality drops (reading from end)
        """
        if not self.results:
            return None, None

        frame_numbers = sorted(self.results.keys())
        start_frame = None
        end_frame = None

        # Find start frame
        for i in range(len(frame_numbers) - 1):
            curr_frame = frame_numbers[i]
            next_frame = frame_numbers[i + 1]

            curr_result = self.results[curr_frame]
            next_result = self.results[next_frame]

            # Check conditions for start
            two_consecutive_dqi = (curr_result.dqi > 1.0 and next_result.dqi > 1.0)
            high_lqp = curr_result.n_pattern_spots > 100

            if two_consecutive_dqi or high_lqp:
                start_frame = curr_frame + 1  # Convert to 1-indexed
                break

        # Find end frame (reading from end)
        for i in range(len(frame_numbers) - 1, 0, -1):
            curr_frame = frame_numbers[i]
            prev_frame = frame_numbers[i - 1]

            curr_result = self.results[curr_frame]
            prev_result = self.results[prev_frame]

            # Check conditions for end
            prev_dqi_good = prev_result.dqi > 1.0
            curr_lqp_high = curr_result.n_pattern_spots > 100

            if prev_dqi_good or curr_lqp_high:
                end_frame = curr_frame + 1  # Convert to 1-indexed
                break

        # Validate range
        if start_frame is not None and end_frame is not None and start_frame <= end_frame:
            # Frame range will be logged in calling function
            return start_frame, end_frame
        else:
            logging.warning("No good frame range found, will use full range")
            if frame_numbers:
                return 1, len(frame_numbers)
            return None, None

    def get_quality_summary(self) -> Dict[str, Any]:
        """Get summary statistics of quality analysis"""
        if not self.results:
            return {}

        total_frames = len(self.results)
        quality_counts = {}
        dqi_values = []
        lqp_values = []

        for result in self.results.values():
            quality = result.quality
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            dqi_values.append(result.dqi)
            lqp_values.append(result.n_pattern_spots)

        summary = {
            'total_frames': total_frames,
            'quality_distribution': quality_counts,
            'mean_dqi': np.mean(dqi_values),
            'max_dqi': np.max(dqi_values),
            'mean_lqp': np.mean(lqp_values),
            'max_lqp': np.max(lqp_values)
        }

        return summary

    def save_quality_csv(self, output_dir: Path, filename: str) -> Optional[Path]:
        """
        Save frame-by-frame quality analysis results to CSV file.

        Args:
            output_dir: Directory where CSV should be saved (should be images/logs)
            filename: Base filename (without extension) for the CSV

        Returns:
            Path to saved CSV file or None if failed
        """
        if not self.results:
            logging.warning("No quality results to save")
            return None

        try:
            # Create logs directory if it doesn't exist
            logs_dir = output_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            # Prepare data for CSV
            csv_data = []
            for frame_num in sorted(self.results.keys()):
                result = self.results[frame_num]
                csv_data.append({
                    'frame_number': result.frame_number + 1,  # Convert to 1-indexed for XDS compatibility
                    'n_diffraction_spots': result.n_diffraction_spots,
                    'n_pattern_spots': result.n_pattern_spots,
                    'dqi': result.dqi,
                    'quality': result.quality,
                    'total_sum': result.total_sum
                })

            # Save CSV
            csv_path = logs_dir / f"{filename}_quality_analysis.csv"

            # Write CSV manually to avoid pandas dependency
            with open(csv_path, 'w') as f:
                # Write header
                f.write("frame_number,n_diffraction_spots,n_pattern_spots,dqi,quality,total_sum\n")

                # Write data
                for row in csv_data:
                    f.write(f"{row['frame_number']},{row['n_diffraction_spots']},{row['n_pattern_spots']},{row['dqi']:.6f},{row['quality']},{row['total_sum']:.2f}\n")

            return csv_path

        except Exception as e:
            logging.error(f"Error saving quality CSV: {str(e)}")
            return None

    def load_quality_from_csv(self, csv_path: Path) -> bool:
        """
        Load quality analysis results from existing CSV file.

        Args:
            csv_path: Path to the CSV file

        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            self.results = {}

            with open(csv_path, 'r') as f:
                # Skip header
                next(f)

                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 6:
                        frame_number = int(parts[0]) - 1  # Convert back to 0-indexed
                        n_diffraction_spots = int(parts[1])
                        n_pattern_spots = int(parts[2])
                        dqi = float(parts[3])
                        quality = parts[4]
                        total_sum = float(parts[5])

                        result = DiffractionResult(
                            frame_number=frame_number,
                            n_diffraction_spots=n_diffraction_spots,
                            n_pattern_spots=n_pattern_spots,
                            total_sum=total_sum,
                            dqi=dqi,
                            quality=quality
                        )

                        self.results[frame_number] = result

            logging.info(f"Loaded quality analysis from CSV: {len(self.results)} frames")
            return True

        except Exception as e:
            logging.error(f"Error loading quality CSV {csv_path}: {str(e)}")
            return False
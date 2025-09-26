"""
Processing tracking and log management
"""
from datetime import datetime
from pathlib import Path
from typing import Optional


class ProcessTracker:
    """Handles processing tracking and log management"""

    PROCESSED_FILES_LOG = "autoprocess_tracking.log"

    def __init__(self, log_print_func=None):
        self.log_print = log_print_func or self._default_log_print
        self.current_path = Path.cwd()

    def _default_log_print(self, message: str) -> None:
        """Default logging function"""
        print(message)

    def get_processed_files_log_path(self) -> Path:
        """Get path to the processed files tracking log"""
        log_dir = Path.cwd() / "autoprocess_logs"
        log_dir.mkdir(exist_ok=True)
        return log_dir / self.PROCESSED_FILES_LOG

    def is_file_already_processed(self, file_path: Path) -> Optional[str]:
        """Check if file has been processed before by looking at the tracking log.

        Returns the output folder name if found, None otherwise.
        """
        log_path = self.get_processed_files_log_path()

        if not log_path.exists():
            return None

        target_filename = file_path.name
        target_absolute_path = str(file_path.resolve())

        # Read from end of file to get most recent entries first
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()

            # Check in reverse order (most recent first)
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue

                parts = line.split('|')
                if len(parts) >= 4:
                    timestamp, filename, absolute_path, output_folder = parts[:4]

                    # Match by both filename and absolute path for accuracy
                    if filename == target_filename and absolute_path == target_absolute_path:
                        return output_folder

        except Exception as e:
            self.log_print(f"Warning: Could not read processing log: {e}")

        return None

    def add_to_processed_files_log(self, file_path: Path, output_folder: Path) -> None:
        """Add entry to processed files tracking log."""
        log_path = self.get_processed_files_log_path()

        try:
            # Create timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            filename = file_path.name
            absolute_file_path = str(file_path.resolve())
            absolute_output_folder = str(output_folder.resolve())

            # Format: timestamp|filename|absolute_path|output_folder
            log_entry = f"{timestamp}|{filename}|{absolute_file_path}|{absolute_output_folder}\n"

            with open(log_path, 'a') as f:
                f.write(log_entry)

        except Exception as e:
            self.log_print(f"Warning: Could not write to processing log: {e}")
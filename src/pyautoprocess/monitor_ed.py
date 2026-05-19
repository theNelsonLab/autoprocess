"""
monitorED - Active file monitor for MicroED data processing

State machine that watches for incoming diffraction data files and triggers
autoprocess or image_process when new data arrives.
"""
import argparse
import logging
import os
import sys
import time
from enum import Enum, auto
from pathlib import Path
from subprocess import PIPE, run
from typing import Dict, List, Optional, Set

from .config.config_manager import ConfigLoader
from .ui.display_manager import DisplayManager

# Supported movie file extensions
MOVIE_EXTENSIONS = {".mrc", ".ser", ".tvips"}


class MonitorState(Enum):
    SCANNING = auto()      # Actively scanning for new files / folders
    PROCESSING = auto()    # Running autoprocess / image_process on a file
    IDLE = auto()          # Waiting; nothing new found this cycle
    COMPLETED = auto()     # Terminal state


class MonitorED:
    """Active monitor that detects new MicroED data and triggers processing."""

    TRACKING_LOG = "monitored_tracking.log"
    POLL_INTERVAL = 5  # seconds between scans

    def __init__(
        self,
        mode: str,
        passthrough_args: List[str],
        watch_subdirs: bool = False,
        timeout: int = 7200,
        expect_count: Optional[int] = None,
        working_directory: str = ".",
    ):
        self.mode = mode                       # "autoprocess" or "image_process"
        self.passthrough_args = passthrough_args
        self.watch_subdirs = watch_subdirs
        self.timeout = timeout
        self.expect_count = expect_count
        self.working_directory = Path(working_directory).resolve()

        self.state = MonitorState.SCANNING
        self.processed_files: Set[str] = set()
        self.processed_count = 0
        self.last_activity_time = time.time()

        # Stability tracking: path -> (size, mtime) from last poll
        self._file_stability: Dict[str, tuple] = {}

        # Setup logging
        self.display = DisplayManager()
        self.display.setup_logging("monitored.log", "autoprocess_logs")
        self.logger = logging.getLogger(__name__)

        # Load existing tracking log
        self._load_tracking_log()

    # ─── Tracking log persistence ────────────────────────────────────────

    def _tracking_log_path(self) -> Path:
        return self.working_directory / self.TRACKING_LOG

    def _load_tracking_log(self) -> None:
        """Load previously processed entries so we never reprocess."""
        log_path = self._tracking_log_path()
        if log_path.exists():
            for line in log_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    self.processed_files.add(line)
            self.log(f"Loaded {len(self.processed_files)} entries from tracking log")

    def _record_processed(self, identifier: str) -> None:
        """Append a processed identifier to the tracking log."""
        self.processed_files.add(identifier)
        with open(self._tracking_log_path(), "a") as f:
            f.write(f"{identifier}\n")

    # ─── Logging helper ──────────────────────────────────────────────────

    def log(self, message: str) -> None:
        self.display.log_print(message)

    # ─── File stability check ────────────────────────────────────────────

    def _is_file_stable(self, path: Path) -> bool:
        """Return True if file size and mtime haven't changed since last poll."""
        key = str(path)
        try:
            stat = path.stat()
            current = (stat.st_size, stat.st_mtime)
        except OSError:
            return False

        previous = self._file_stability.get(key)
        self._file_stability[key] = current

        if previous is None:
            return False  # First time seeing this file; wait one more cycle
        return previous == current

    def _is_folder_stable(self, folder: Path) -> bool:
        """Return True if all image files inside the folder are stable."""
        image_files = list(folder.glob("*.tif")) + list(folder.glob("*.img"))
        if not image_files:
            return False
        return all(self._is_file_stable(f) for f in image_files)

    # ─── Filename validation ─────────────────────────────────────────────

    @staticmethod
    def _looks_numeric(value: str) -> bool:
        """Check if a string looks like a number, allowing 'p' as decimal separator.

        Accepts: '960', '0.3', '1p5', '0p6'
        Rejects: 'movie', 'bin4', 'g8sp7'
        """
        # Replace 'p' with '.' to normalize, then check if it's a valid float
        normalized = value.replace("p", ".")
        try:
            float(normalized)
            return True
        except ValueError:
            return False

    @classmethod
    def validate_movie_filename(cls, filename: str) -> bool:
        """Validate filename matches the expected schema.

        Mirrors autoprocess.parse_filename logic:
        split on '_', require at least 4 parts (sample, distance, rotation, exposure),
        with a supported extension (.mrc, .ser, .tvips).
        Fields 2-4 (distance, rotation, exposure) must look numeric
        (digits, dots, or 'p' as decimal separator).

        Example valid names:
          sample-mov1_960_0.3_3_notes.ser
          Lysozyme-NAG2-DC-xtal-05_960_1p5_0p6_p40ton60_g8sp7_bin4_0_movie.mrc
        Example rejected names:
          20260513_98917_0_movie.mrc  (4th field 'movie' is not numeric)
        """
        stem = Path(filename).stem
        ext = Path(filename).suffix.lower()
        if ext not in MOVIE_EXTENSIONS:
            return False
        parts = stem.split("_")
        # Need at least: sample, distance, rotation, exposure
        if len(parts) < 4:
            return False
        # Fields at index 1, 2, 3 must be numeric-like
        return all(cls._looks_numeric(parts[i]) for i in (1, 2, 3))

    # ─── Directory scanning ──────────────────────────────────────────────

    def _get_scan_dirs(self) -> List[Path]:
        """Return list of directories to scan."""
        dirs = [self.working_directory]
        if self.watch_subdirs:
            for item in self.working_directory.iterdir():
                if item.is_dir() and not item.name.startswith("."):
                    dirs.append(item)
        return dirs

    def _scan_autoprocess(self) -> List[Path]:
        """Find new movie files (.mrc/.ser/.tvips) ready for autoprocess."""
        new_files: List[Path] = []
        for scan_dir in self._get_scan_dirs():
            for ext in ("*.mrc", "*.ser", "*.tvips"):
                for filepath in scan_dir.glob(ext):
                    identifier = str(filepath.resolve())
                    if identifier in self.processed_files:
                        continue
                    if not self.validate_movie_filename(filepath.name):
                        continue
                    if not self._is_file_stable(filepath):
                        continue
                    new_files.append(filepath)
        return new_files

    def _scan_image_process(self) -> List[Path]:
        """Find new folders with images/ containing TIF or IMG files."""
        new_folders: List[Path] = []
        for scan_dir in self._get_scan_dirs():
            for item in scan_dir.iterdir():
                if not item.is_dir():
                    continue
                images_dir = item / "images"
                if not images_dir.is_dir():
                    continue

                identifier = str(item.resolve())
                if identifier in self.processed_files:
                    continue

                # Validate content: must have .tif or .img files
                tif_files = list(images_dir.glob("*.tif"))
                img_files = list(images_dir.glob("*.img"))
                if not tif_files and not img_files:
                    continue

                # Check stability of the images folder
                if not self._is_folder_stable(images_dir):
                    continue

                new_folders.append(item)
        return new_folders

    # ─── Command execution ───────────────────────────────────────────────

    def _build_command(self, target: Path) -> List[str]:
        """Build the subprocess command for autoprocess or image_process."""
        cmd = [self.mode]
        # Add the target path as positional argument
        cmd.append(str(target))
        # Append all user-provided passthrough flags
        cmd.extend(self.passthrough_args)
        return cmd

    def _run_processing(self, target: Path) -> bool:
        """Execute autoprocess or image_process on the target."""
        cmd = self._build_command(target)
        self.log(f"Running: {' '.join(cmd)}")
        try:
            result = run(
                cmd,
                stdout=PIPE,
                stderr=PIPE,
                text=True,
                cwd=str(target.parent) if self.mode == "autoprocess" else str(self.working_directory),
            )
            if result.stdout:
                self.log(result.stdout.rstrip())
            if result.stderr:
                self.log(f"[stderr] {result.stderr.rstrip()}")
            if result.returncode == 0:
                self.log(f"Successfully processed: {target.name}")
                return True
            else:
                self.log(f"Processing failed (exit {result.returncode}): {target.name}")
                return False
        except FileNotFoundError:
            self.log(f"Error: command '{self.mode}' not found. Is pyautoprocess installed?")
            return False
        except Exception as e:
            self.log(f"Error running {self.mode}: {e}")
            return False

    # ─── Main monitor loop ───────────────────────────────────────────────

    def run(self) -> None:
        """Main monitoring loop."""
        self._print_banner()
        self.log(f"monitorED started in {self.mode} mode")
        self.log(f"Watching: {self.working_directory}")
        if self.watch_subdirs:
            self.log("Subdirectory monitoring: enabled (1 level)")
        self.log(f"Inactivity timeout: {self.timeout}s ({self.timeout // 60} min)")
        if self.expect_count:
            self.log(f"Expected file count: {self.expect_count}")
        self.log(f"Passthrough args: {' '.join(self.passthrough_args) if self.passthrough_args else '(none)'}")
        self.log("")

        try:
            while self.state != MonitorState.COMPLETED:
                # Scan for new data
                if self.mode == "autoprocess":
                    targets = self._scan_autoprocess()
                else:
                    targets = self._scan_image_process()

                if targets:
                    self.state = MonitorState.PROCESSING
                    self.last_activity_time = time.time()

                    for target in targets:
                        identifier = str(target.resolve())
                        self.log(f"\nNew data detected: {target.name}")

                        success = self._run_processing(target)
                        self._record_processed(identifier)
                        self.processed_count += 1

                        if success:
                            self.log(f"Processed {self.processed_count} file(s) so far")
                        else:
                            self.log(f"Failed to process {target.name}, recorded to avoid retry")

                        # Check expect_count termination
                        if self.expect_count and self.processed_count >= self.expect_count:
                            self.log(f"\nReached expected count of {self.expect_count} files. Done.")
                            self.state = MonitorState.COMPLETED
                            break

                    if self.state == MonitorState.COMPLETED:
                        break

                    self.state = MonitorState.SCANNING
                else:
                    self.state = MonitorState.IDLE

                    # Check inactivity timeout
                    elapsed = time.time() - self.last_activity_time
                    if elapsed >= self.timeout:
                        self.log(f"\nInactivity timeout ({self.timeout // 60} min) reached. Shutting down.")
                        self.state = MonitorState.COMPLETED
                        break

                time.sleep(self.POLL_INTERVAL)

        except KeyboardInterrupt:
            self.log("\nMonitor interrupted by user.")

        self.log(f"\nmonitorED summary: processed {self.processed_count} file(s)")

    def _print_banner(self) -> None:
        banner = r"""
                     THANK YOU FOR USING

                        _ __           ________
   ____ ___  ____  ____(_) /_____  ____/ ____/ /_
  / __ `__ \/ __ \/ __ \/ / __/ __ \/ ___/ __/ __  \
 / / / / / / /_/ / / / / / /_/ /_/ / /  / /_/ / / /
/_/ /_/ /_/\____/_/ /_/_/\__/\____/_/  /\___/_/ /_/ dbe
"""
        self.log(banner)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    """Build the monitorED argument parser.

    monitorED-specific flags are parsed explicitly.
    Everything after the mode flag (--autoprocess / --image-process) that is
    not a monitorED flag is collected and forwarded verbatim to the child
    command.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Active monitor for MicroED data. Watches for new files/folders "
            "and triggers autoprocess or image_process automatically."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor current directory for new .mrc/.ser files, run autoprocess
  monitorED --autoprocess --microscope-config default

  # Monitor with subdirectories, expect 10 datasets, image_process mode
  monitorED --image-process --watch-subdirs --expect-count 10 --microscope-config default

  # Custom timeout and passthrough flags
  monitorED --autoprocess --timeout 3600 --parallel --dqa --microscope-config Arctica-CETA-ser-SM

Note:
  All flags not listed below are forwarded to the selected processing command
  (autoprocess or image_process). See their respective --help for details.
""",
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--autoprocess",
        action="store_true",
        help="Monitor for raw movie files (.mrc/.ser/.tvips) and run autoprocess",
    )
    mode_group.add_argument(
        "--image-process",
        action="store_true",
        help="Monitor for folders with images/ subdirectory and run image_process",
    )

    parser.add_argument(
        "--watch-subdirs",
        action="store_true",
        help="Also monitor immediate subdirectories (1 level deep)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Inactivity timeout in seconds (default: 7200 = 2 hours)",
    )
    parser.add_argument(
        "--expect-count",
        type=int,
        default=None,
        help="Stop after processing this many files/folders",
    )

    return parser


def main() -> int:
    # We need to separate monitorED's own flags from the passthrough flags.
    # Strategy: parse known args for monitorED, everything else is passthrough.
    parser = _build_parser()
    known_args, passthrough = parser.parse_known_args()

    mode = "autoprocess" if known_args.autoprocess else "image_process"

    monitor = MonitorED(
        mode=mode,
        passthrough_args=passthrough,
        watch_subdirs=known_args.watch_subdirs,
        timeout=known_args.timeout,
        expect_count=known_args.expect_count,
    )
    monitor.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

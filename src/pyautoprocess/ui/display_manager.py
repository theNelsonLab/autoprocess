"""
Display and logging management for pyautoprocess
"""
import logging
from pathlib import Path


class DisplayManager:
    """Handles banner display and logging setup"""

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

    def print_sub_banner(self, mode: str = "default"):
        """Print subbanner with optional mode specification."""
        if mode == "image_process":
            version_banner = [
                "",
                "================================================================",
                "                        AutoProcess 2.0                         ",
                "                     IMAGE PROCESSING MODE                      ",
                "================================================================",
                ""
            ]
        elif mode == "batch_manual":
            version_banner = [
                "",
                "================================================================",
                "                        AutoProcess 2.0                         ",
                "                 BATCH REPROCESSING - MANUAL MODE                ",
                "================================================================",
                ""
            ]
        elif mode == "batch_smart":
            version_banner = [
                "",
                "================================================================",
                "                        AutoProcess 2.0                         ",
                "                 BATCH REPROCESSING - SMART MODE                ",
                "================================================================",
                ""
            ]
        else:  # default mode for autoprocess
            version_banner = [
                "",
                "================================================================",
                "                        AutoProcess 2.0                         ",
                "================================================================",
                ""
            ]

        for line in version_banner:
            self.log_print(line)

    def print_full_banner(self, mode: str = "default"):
        """Print complete banner (main + sub) for specified mode."""
        self.print_banner()
        self.print_sub_banner(mode)
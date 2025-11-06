import struct
import numpy as np
from pathlib import Path
from typing import Iterator


class TvipsReader:
    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self.file = open(self.filepath, 'rb')
        self._read_series_header()

    def _read_series_header(self):
        """Read and parse the 256-byte series header"""
        header = self.file.read(256)

        # Parse key header fields (little-endian)
        self.header_size = struct.unpack('<i', header[0:4])[0]  # Should be 256
        self.version = struct.unpack('<i', header[4:8])[0]  # 1 or 2
        self.width = struct.unpack('<i', header[8:12])[0]  # X dimension
        self.height = struct.unpack('<i', header[12:16])[0]  # Y dimension
        self.bits_per_pixel = struct.unpack('<i', header[16:20])[0]  # 8 or 16

        if self.bits_per_pixel != 16:
            raise ValueError('expect tvips to use uint16')

        # Image header size depends on version
        self.img_header_size = 12 if self.version == 1 else struct.unpack('<i', header[48:52])[0]

    def read_frame(self) -> tuple[dict, np.ndarray]:
        """Read next frame, returns (metadata, image_data)"""
        # Read image header
        header = self.file.read(self.img_header_size)
        if not header:
            raise EOFError("End of file reached")

        # Parse image metadata
        metadata = {
            'counter': struct.unpack('<i', header[0:4])[0],  # 1-based frame counter
            'timestamp_sec': struct.unpack('<i', header[4:8])[0],
            'timestamp_ms': struct.unpack('<i', header[8:12])[0]
        }

        # Read image data
        pixels = self.width * self.height
        raw_data = self.file.read(pixels * 2)

        # tvips is already little endian by default, so no need to change
        image = np.frombuffer(raw_data, dtype=np.uint16).reshape(self.height, self.width)

        return metadata, image

    def read_all_frames(self) -> list[np.ndarray]:
        """
        Read all frames from the file and return as list of numpy arrays.

        Returns:
            list[np.ndarray]: List of image arrays
        """
        frames = []
        try:
            while True:
                _metadata, image = self.read_frame()
                frames.append(image)
        except EOFError:
            pass

        return frames

    def __iter__(self) -> Iterator[tuple[dict, np.ndarray]]:
        """Iterate through all frames in the file"""
        try:
            while True:
                yield self.read_frame()
        except EOFError:
            pass

    def close(self):
        """Close the file handle"""
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


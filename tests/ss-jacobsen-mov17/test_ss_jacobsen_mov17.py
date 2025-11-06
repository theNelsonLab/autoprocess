import subprocess
from pathlib import Path


def test_jacobsen_mov17():
    test_dir = Path(__file__).parent
    subprocess.call(['autoprocess', '--microscope-config', 'Arctica-CETA-mrc-SM'], cwd=test_dir)

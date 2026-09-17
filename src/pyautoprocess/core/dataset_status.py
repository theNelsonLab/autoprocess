"""
Per-dataset result records: autoprocess_logs/<dataset>_status.json

One file per dataset, rewritten each time the dataset is processed, so a caller can read
what happened without inferring success from intermediate files such as XDS.INP or
CORRECT.LP. The fields are a stable contract; see "Result files" in the README.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

SCHEMA_VERSION = 1
STATUS_SUFFIX = "_status.json"

SUCCESS = "success"
FAILED = "failed"
SKIPPED = "skipped"

# Output files reported when present, relative to the dataset's auto_process/ folder.
# "{dataset}" is replaced with the dataset name.
KEY_OUTPUTS = (
    "XDS.INP",
    "XPARM.XDS",
    "INTEGRATE.HKL",
    "CORRECT.LP",
    "XDS_ASCII.HKL",
    "{dataset}.ahkl",
    "{dataset}.hkl",
    "stats.LP",
    "pointless.LP",
)


def status_path(log_dir: Path, dataset: str) -> Path:
    safe_name = dataset.replace(os.sep, "_").replace("/", "_")
    return Path(log_dir) / f"{safe_name}{STATUS_SUFFIX}"


def collect_outputs(output_dir: Path, dataset: str) -> Dict[str, str]:
    """Absolute paths of the key output files that exist."""
    auto_process = Path(output_dir) / "auto_process"
    outputs = {}
    for template in KEY_OUTPUTS:
        name = template.format(dataset=dataset)
        path = auto_process / name
        if path.is_file():
            outputs[name] = os.path.abspath(path)
    return outputs


def write_status(log_dir: Path, dataset: str, status: str, reason: str,
                 source_file: Path, output_dir: Path) -> Path:
    """Write the record atomically, so a reader never sees a half-written file."""
    from .. import __version__

    record = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "status": status,
        "reason": reason,
        "source_file": os.path.abspath(source_file),
        "output_dir": os.path.abspath(output_dir),
        "outputs": collect_outputs(output_dir, dataset),
        "pyautoprocess_version": __version__,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = status_path(log_dir, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_status(log_dir: Path, dataset: str) -> Optional[dict]:
    """The record for a dataset, or None if there is none or it cannot be read."""
    path = status_path(log_dir, dataset)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return record if isinstance(record, dict) else None

"""
autoprocess-summary: collect the XSCALE statistics of every processed data set
under a directory into one text file.

A data set is a folder holding ``auto_process/XSCALE.LP``. The search goes down
``--depth`` levels from ``--dir`` (default 4: a REyes session keeps them at
``movies/<movie folder>/<name>/auto_process``), skipping hidden folders and
``processing_backups``. Folders that have ``auto_process/`` but no XSCALE.LP
are listed as not scaled, so failed data sets stay visible.

The file opens with a quick-glance table (total-row statistics and the highest
resolution shell of every data set), then gives the full XSCALE statistics
table of each data set between fences, so tables of different lengths never
run into each other.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, TextIO

from .core.xscale_stats import NOT_AVAILABLE, XscaleStats, parse_xscale_lp

DEFAULT_OUTPUT_NAME = "Summary_Results.txt"
DEFAULT_DEPTH = 4
SKIPPED_FOLDERS = {"processing_backups"}

DIVIDER_MAJOR = "=" * 120
DIVIDER_MINOR = "-" * 120
DATASET_SPACER = "\n" * 4


@dataclass
class Dataset:
    """A folder holding auto_process/, found under the search directory."""
    folder: Path
    relative: str                       # folder relative to the search directory, POSIX style
    label: str = ""                     # name shown in the summary
    stats: Optional[XscaleStats] = None  # None when auto_process/ has no XSCALE.LP

    @property
    def xscale_lp(self) -> Path:
        return self.folder / "auto_process" / "XSCALE.LP"

    @property
    def scaled(self) -> bool:
        return self.stats is not None


def natural_key(text: str) -> list:
    """Sort key that orders embedded numbers numerically: mov2 before mov10."""
    return [(0, int(part), "") if part.isdecimal() else (1, 0, part.lower())
            for part in re.split(r"(\d+)", text) if part]


def _walk(folder: Path, depth: int, max_depth: int,
          unreadable: List[Path]) -> Iterator[Path]:
    """Yield every folder holding auto_process/, down to max_depth levels below the start.

    Folders that cannot be listed are appended to ``unreadable`` and skipped. Symbolic links
    to folders are followed (data are often linked in); --depth bounds any link loop, and
    find_datasets drops a data set reached twice.
    """
    # os.path.isdir/isfile return False instead of raising on a folder we may not enter.
    if os.path.isdir(folder / "auto_process"):
        yield folder
        return                          # a data set's own subfolders are not searched
    if depth >= max_depth:
        return
    try:
        with os.scandir(folder) as entries:
            children = sorted(entry.name for entry in entries
                              if not entry.name.startswith(".")
                              and entry.name not in SKIPPED_FOLDERS
                              and _is_dir(entry))
    except OSError:
        unreadable.append(folder)
        return
    for name in children:
        yield from _walk(folder / name, depth + 1, max_depth, unreadable)


def _is_dir(entry: os.DirEntry) -> bool:
    try:
        return entry.is_dir()
    except OSError:
        return False


def find_datasets(root: Path, max_depth: int = DEFAULT_DEPTH,
                  unreadable: Optional[List[Path]] = None) -> List[Dataset]:
    """Every data set under ``root`` in natural order, labelled and parsed.

    ``max_depth`` is how many folder levels below ``root`` the data-set folder may sit
    (0: ``root`` itself is the data set). Folders that could not be listed are appended to
    ``unreadable`` when a list is given. A data set reached by more than one path (through
    a symbolic link) is listed once, preferring a path without links.
    """
    if unreadable is None:
        unreadable = []
    by_real = {}
    for folder in _walk(root, 0, max_depth, unreadable):
        relative = folder.relative_to(root).as_posix()
        candidate = Dataset(folder=folder, relative=root.name if relative == "." else relative)
        real = os.path.realpath(folder)
        kept = by_real.get(real)
        if kept is None or (os.path.realpath(kept.folder) != str(kept.folder)
                            and real == str(folder)):
            by_real[real] = candidate
    datasets = list(by_real.values())

    _assign_labels(datasets)
    for d in datasets:
        if os.path.isfile(d.xscale_lp):
            d.stats = parse_xscale_lp(d.xscale_lp)

    datasets.sort(key=lambda d: (natural_key(d.folder.name), natural_key(d.relative)))
    return datasets


def _assign_labels(datasets: List[Dataset]) -> None:
    """Label = folder name when unique; otherwise the shortest trailing part of the relative
    path that tells it apart (e.g. 'grid-2/sample-mov1'), up to the whole relative path."""
    parts = [tuple(d.relative.split("/")) for d in datasets]
    for d, mine in zip(datasets, parts, strict=True):
        for k in range(1, len(mine) + 1):
            suffix = mine[-k:]
            if sum(1 for other in parts if other[-k:] == suffix) == 1:
                break
        d.label = "/".join(suffix)


# ---------------------------------------------------------------- output

def _total_text(stats: XscaleStats, column: str) -> str:
    return stats.total.text(column) if stats.total is not None else NOT_AVAILABLE


def format_cell(cell: Optional[str]) -> str:
    """Unit cell in fixed widths (lengths 7.2f, angles 8.3f) so columns line up across rows;
    the string as XSCALE printed it when it does not hold six numbers."""
    if not cell:
        return NOT_AVAILABLE
    try:
        values = [float(v) for v in cell.split()]
    except ValueError:
        return cell
    if len(values) != 6:
        return cell
    return "".join(f"{v:7.2f}" for v in values[:3]) + "".join(f"{v:8.3f}" for v in values[3:])


def write_quick_glance(f: TextIO, scaled: Sequence[Dataset]) -> None:
    """Total-row statistics and the highest resolution shell of every data set.

    Every column is as wide as its widest entry (at least the defaults), so rows never
    shift; the dividers are as wide as the table (at least 120).
    """
    header = ("Dataset", "SG", "Unit Cell", "Hi-Res", "Cmpl%", "R-meas%", "I/Sigma", "CC1/2")
    rows = []
    for d in scaled:
        s = d.stats
        rows.append((d.label, s.sg or NOT_AVAILABLE, format_cell(s.cell),
                     s.resolution or NOT_AVAILABLE,
                     _total_text(s, "completeness"), _total_text(s, "r_meas"),
                     _total_text(s, "i_sigma"), _total_text(s, "cc_half")))
    minimum = (len("Dataset") + 2, 4, 46, 8, 9, 9, 9, 8)
    widths = [max([m] + [len(r[k]) + (2 if k == 0 else 0) for r in rows] + [len(header[k])])
              for k, m in enumerate(minimum)]
    left = (True, True, True, False, False, False, False, False)

    def row(values):
        cells = [f"{v:<{w}}" if lft else f"{v:>{w}}" for v, w, lft in zip(values, widths, left, strict=True)]
        return "  " + " ".join(cells)

    lines = [row(header)] + [row(r) for r in rows]
    width = max([120] + [len(line) for line in lines])
    f.write("=" * width + "\n")
    f.write("  QUICK-GLANCE TOTALS\n")
    f.write("  (Total row statistics + highest resolution shell for each dataset)\n")
    f.write("-" * width + "\n")
    f.write(lines[0] + "\n")
    f.write("-" * width + "\n")
    for line in lines[1:]:
        f.write(line + "\n")
    f.write("=" * width + "\n")


def write_not_scaled(f: TextIO, not_scaled: Sequence[Dataset]) -> None:
    f.write(DIVIDER_MAJOR + "\n")
    f.write("  NOT SCALED (auto_process/ without XSCALE.LP)\n")
    f.write(DIVIDER_MINOR + "\n")
    for d in not_scaled:
        f.write(f"  {d.label}\n")
    f.write(DIVIDER_MAJOR + "\n")


def write_dataset(f: TextIO, d: Dataset, index: int, count: int) -> None:
    s = d.stats
    f.write(DIVIDER_MAJOR + "\n")
    f.write(f"  [{index:>3} / {count}]  {d.label}\n")
    f.write(DIVIDER_MINOR + "\n")
    f.write(f"  Space group : {s.sg or NOT_AVAILABLE}\n")
    f.write(f"  Unit cell   : {s.cell or NOT_AVAILABLE}\n")
    f.write(f"  Folder      : {d.relative}\n")
    if s.total is not None:
        f.write(DIVIDER_MINOR + "\n")
        f.write(f"  Hi-res : {s.resolution or NOT_AVAILABLE:>6}   "
                f"Cmpl : {_total_text(s, 'completeness'):>6}   "
                f"R-meas : {_total_text(s, 'r_meas'):>7}   "
                f"I/Sigma : {_total_text(s, 'i_sigma'):>6}   "
                f"CC1/2 : {_total_text(s, 'cc_half'):>6}\n")
    f.write(DIVIDER_MINOR + "\n\n")
    if s.table_text.strip():
        f.write(s.table_text)
        if s.error:
            f.write(f"\n\n  ({s.error})")
    else:
        f.write(f"  (No statistics table found: {s.error})\n" if s.error
                else "  (No statistics table found in XSCALE.LP)\n")
    f.write("\n\n")
    f.write(DIVIDER_MAJOR + "\n")
    f.write(f"  END  {d.label}\n")
    f.write(DIVIDER_MAJOR + "\n")


def write_summary(datasets: Sequence[Dataset], out_path: Path, root: Path,
                  now: Optional[datetime] = None) -> None:
    """Write the summary file. ``now`` fixes the timestamp (for tests)."""
    stamp = (now or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    scaled = [d for d in datasets if d.scaled]
    not_scaled = [d for d in datasets if not d.scaled]

    # Written to a temporary file next to the target and renamed into place, so a failure
    # never leaves a half-written summary. surrogateescape writes folder names that are not
    # valid UTF-8 back as the bytes they came from.
    tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
    try:
        os.unlink(tmp_path)             # left over from a crashed run with the same PID
    except OSError:
        pass
    try:
        with open(tmp_path, "x", encoding="utf-8", errors="surrogateescape") as f:
            _write_summary_body(f, datasets, root, stamp, scaled, not_scaled)
        os.replace(tmp_path, out_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _write_summary_body(f: TextIO, datasets: Sequence[Dataset], root: Path, stamp: str,
                        scaled: Sequence[Dataset], not_scaled: Sequence[Dataset]) -> None:
    f.write(DIVIDER_MAJOR + "\n")
    f.write("  microED XSCALE SUMMARY\n")
    f.write(f"  Generated : {stamp}\n")
    f.write(f"  Directory : {root}\n")
    f.write(f"  Datasets  : {len(scaled)}\n")
    if not_scaled:
        f.write(f"  Not scaled: {len(not_scaled)}\n")
    f.write(DIVIDER_MAJOR + "\n")

    if scaled:
        f.write(DATASET_SPACER)
        write_quick_glance(f, scaled)
    if not_scaled:
        f.write(DATASET_SPACER)
        write_not_scaled(f, not_scaled)

    if scaled:
        f.write(DATASET_SPACER)
        f.write(DIVIDER_MAJOR + "\n")
        f.write("  FULL XSCALE TABLES\n")
        f.write(DIVIDER_MAJOR + "\n")
        for index, d in enumerate(scaled, 1):
            f.write(DATASET_SPACER)
            write_dataset(f, d, index, len(scaled))

    f.write(DATASET_SPACER)
    f.write(DIVIDER_MAJOR + "\n")
    f.write(f"  Summary complete — {len(scaled)} dataset(s)\n")
    f.write(DIVIDER_MAJOR + "\n")


# ---------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autoprocess-summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Collect the XSCALE statistics of every processed dataset under a directory\n"
            "into one text file.\n\n"
            "A dataset is a folder holding auto_process/XSCALE.LP. The file opens with a\n"
            "quick-glance table (total-row statistics and the highest resolution shell of\n"
            "every dataset), followed by the full XSCALE statistics table of each dataset\n"
            "between fences. Folders with auto_process/ but no XSCALE.LP are listed as\n"
            "not scaled. Hidden folders and processing_backups/ are skipped."
        ),
        epilog="""
Examples:
  # Summarise the folder you are in; writes Summary_Results.txt there
  cd /data/session/grid-3
  autoprocess-summary

  # A whole REyes session (datasets under movies/<movie folder>/<name>/)
  autoprocess-summary --dir /data/session

  # Write somewhere else, searching deeper
  autoprocess-summary --dir /data/session --out /data/reports/session_summary.txt --depth 6

Exit status: 0 if at least one dataset with XSCALE.LP was found, 1 otherwise.
""",
    )
    parser.add_argument("--dir", metavar="PATH", default=".",
                        help="directory to search (default: the current directory)")
    parser.add_argument("--out", metavar="PATH", default=None,
                        help=f"output file (default: {DEFAULT_OUTPUT_NAME} in the search directory)")
    parser.add_argument("--depth", metavar="N", type=int, default=DEFAULT_DEPTH,
                        help="how many folder levels below --dir a dataset folder may sit "
                             f"(default: {DEFAULT_DEPTH})")
    return parser


def _printable(value) -> str:
    """Text safe for any console encoding (odd folder names become '?' instead of crashing)."""
    text = str(value).encode("utf-8", "surrogateescape").decode("utf-8", "replace")
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, "replace").decode(encoding, "replace")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.depth < 0:
        print("ERROR: --depth must be 0 or more", file=sys.stderr)
        return 1

    root = Path(args.dir).expanduser().resolve()
    if not os.path.isdir(root):
        print(f"ERROR: directory not found: {root}", file=sys.stderr)
        return 1
    out_path = (Path(args.out).expanduser().resolve() if args.out
                else root / DEFAULT_OUTPUT_NAME)
    if os.path.isdir(out_path):
        out_path = out_path / DEFAULT_OUTPUT_NAME

    print(f"Searching for datasets in: {_printable(root)}")
    unreadable: List[Path] = []
    datasets = find_datasets(root, args.depth, unreadable)
    if unreadable:
        print(f"Skipped {len(unreadable)} unreadable folder(s):")
        for folder in unreadable:
            print(f"  {_printable(folder)}")
    scaled = [d for d in datasets if d.scaled]
    not_scaled = [d for d in datasets if not d.scaled]

    if not_scaled:
        print(f"Not scaled ({len(not_scaled)}): auto_process/ without XSCALE.LP")
        for d in not_scaled:
            print(f"  {_printable(d.label)}")
    if not scaled:
        print("No datasets found.\n"
              "Expected structure:  <dataset_name>/auto_process/XSCALE.LP")
        return 1

    print(f"Found {len(scaled)} dataset(s):")
    for d in scaled:
        note = f"  ({d.stats.error})" if d.stats.error else ""
        print(f"  {_printable(d.label)}{note}")

    try:
        write_summary(datasets, out_path, root)
    except (OSError, UnicodeError) as exc:
        print(f"ERROR: could not write {_printable(out_path)}: {_printable(exc)}", file=sys.stderr)
        return 1
    print(f"\nWrote: {_printable(out_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

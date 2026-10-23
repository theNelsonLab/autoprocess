"""
Read the merging statistics that XSCALE writes to XSCALE.LP.

One parser, shared by the statistics block in autoprocess.log and by the
``autoprocess-summary`` command, so the two can never disagree.

The numbers come from the table headed

    SUBSET OF INTENSITY DATA WITH SIGNAL/NOISE >= -3.0 AS FUNCTION OF RESOLUTION

Its numbered rows are resolution shells and its ``total`` row covers all of
them. When XSCALE.LP holds several such tables, the LAST one is used. The
resolution reported for the data set is the limit of the last numbered shell,
which is the highest-resolution shell XSCALE actually reported (0.80 when it
was asked for 0.78, say).

Every value is kept in two forms: a number for code that wants to compare or
sort, and the token exactly as XSCALE printed it ("72.6%", "95.1*", "-99.9%")
for code that wants to show it. The parser never raises: a missing file, a
missing table or a cut-off table sets ``error`` and leaves the rest ``None``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

NOT_AVAILABLE = "n/a"

_TABLE_TITLE = re.compile(
    r"SUBSET OF INTENSITY DATA WITH SIGNAL/NOISE\s*>=\s*\S+\s+AS FUNCTION OF RESOLUTION")

# Columns of a whitespace-split row of that table:
#   0 RESOLUTION LIMIT   1 OBSERVED   2 UNIQUE   3 POSSIBLE   4 COMPLETENESS
#   5 R-FACTOR observed  6 R-FACTOR expected     7 COMPARED   8 I/SIGMA
#   9 R-meas            10 CC(1/2)  (then Anomal Corr, SigAno, Nano, unused here)
_COLUMNS = ("observed", "unique", "possible", "completeness", "r_obs", "r_exp",
            "compared", "i_sigma", "r_meas", "cc_half")
_INTEGER_COLUMNS = {"observed", "unique", "possible", "compared"}


def _number(token: Optional[str], integer: bool) -> Optional[float]:
    if token is None:
        return None
    text = token.rstrip("%*")
    try:
        return int(text) if integer else float(text)
    except ValueError:
        return None


@dataclass(frozen=True)
class XscaleRow:
    """One row of the statistics table: a resolution shell or the ``total`` row."""
    resolution: str                    # "0.80", or "total" for the total row
    tokens: Tuple[str, ...]            # the row split on whitespace, as printed
    observed: Optional[int] = None
    unique: Optional[int] = None
    possible: Optional[int] = None
    completeness: Optional[float] = None   # percent
    r_obs: Optional[float] = None          # percent
    r_exp: Optional[float] = None          # percent
    compared: Optional[int] = None
    i_sigma: Optional[float] = None
    r_meas: Optional[float] = None         # percent
    cc_half: Optional[float] = None        # percent
    cc_significant: Optional[bool] = None  # XSCALE's '*': significant at the 0.1% level

    @classmethod
    def from_line(cls, line: str) -> "XscaleRow":
        tokens = tuple(line.split())
        values = {}
        for index, name in enumerate(_COLUMNS, start=1):
            token = tokens[index] if index < len(tokens) else None
            values[name] = _number(token, name in _INTEGER_COLUMNS)
        cc_token = tokens[10] if len(tokens) > 10 else None
        significant = None if cc_token is None else cc_token.endswith("*")
        return cls(resolution=tokens[0], tokens=tokens, cc_significant=significant, **values)

    def text(self, column: str) -> str:
        """The value of ``column`` as XSCALE printed it, or "n/a" when the row lacks it.

        The CC(1/2) significance mark ``*`` is kept; strip it for display if unwanted.
        """
        index = _COLUMNS.index(column) + 1
        return self.tokens[index] if index < len(self.tokens) else NOT_AVAILABLE

    def percent(self, column: str) -> str:
        """Like :meth:`text`, without the ``*`` and always ending in ``%`` (unless "n/a")."""
        value = self.text(column)
        if value == NOT_AVAILABLE:
            return value
        value = value.rstrip("*")
        return value if value.endswith("%") else value + "%"


@dataclass
class XscaleStats:
    """What XSCALE.LP reports for one scaled data set."""
    path: Path
    sg: Optional[str] = None            # space group number, as XSCALE assumed it
    cell: Optional[str] = None          # unit cell constants, as XSCALE printed them
    resolution: Optional[str] = None    # limit of the last numbered shell, e.g. "0.80"
    shells: List[XscaleRow] = field(default_factory=list)
    total: Optional[XscaleRow] = None
    table_text: str = ""                # the table, verbatim, title to total row
    error: Optional[str] = None         # why the statistics are incomplete, if they are

    # Totals, copied from the total row for convenience (None when absent).
    observed: Optional[int] = None
    unique: Optional[int] = None
    possible: Optional[int] = None
    completeness: Optional[float] = None
    r_obs: Optional[float] = None
    r_exp: Optional[float] = None
    compared: Optional[int] = None
    i_sigma: Optional[float] = None
    r_meas: Optional[float] = None
    cc_half: Optional[float] = None
    cc_significant: Optional[bool] = None

    @property
    def ok(self) -> bool:
        """True when the total row and the resolution of the last shell were both found."""
        return self.error is None


def _crystal_parameters(lines: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Space group and cell from the 'DATA COLLECTION STATISTICS REPORTED BELOW ASSUMES'
    section when present, else from the last SPACE_GROUP_NUMBER= / UNIT_CELL_CONSTANTS= lines."""
    sg = cell = None
    start = None
    for index, line in enumerate(lines):
        if "STATISTICS REPORTED BELOW ASSUMES" in line:
            start = index
    for line in reversed(lines) if start is None else lines[start:]:
        if sg is None and "SPACE_GROUP_NUMBER=" in line:
            sg = line.split("=", 1)[1].strip() or None
        elif cell is None and "UNIT_CELL_CONSTANTS=" in line:
            cell = line.split("=", 1)[1].strip() or None
        if sg is not None and cell is not None:
            break
    return sg, cell


def _starts_with_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def parse_xscale_lp(path) -> XscaleStats:
    """Parse XSCALE.LP at ``path``. Never raises; see ``XscaleStats.error``."""
    stats = XscaleStats(path=Path(path))
    try:
        content = stats.path.read_text(errors="replace")
    except FileNotFoundError:
        stats.error = "XSCALE.LP not found"
        return stats
    except OSError as exc:
        stats.error = f"could not read XSCALE.LP ({exc})"
        return stats

    lines = content.splitlines()
    stats.sg, stats.cell = _crystal_parameters(lines)

    starts = [i for i, line in enumerate(lines) if _TABLE_TITLE.search(line)]
    if not starts:
        stats.error = "no statistics table in XSCALE.LP"
        return stats

    table = []
    rows_started = False
    for line in lines[starts[-1]:]:
        stripped = line.strip()
        first = stripped.split()[0] if stripped else ""
        if first == "total":
            table.append(line)
            stats.total = XscaleRow.from_line(stripped)
            break
        if first and _starts_with_number(first):
            table.append(line)
            stats.shells.append(XscaleRow.from_line(stripped))
            rows_started = True
        elif rows_started and stripped:
            break               # the table ended without a total row
        else:
            table.append(line)  # title, column headers, blank lines
    stats.table_text = "\n".join(table).rstrip()

    if stats.shells:
        stats.resolution = stats.shells[-1].resolution
    if stats.total is not None:
        for name in _COLUMNS + ("cc_significant",):
            setattr(stats, name, getattr(stats.total, name))

    if stats.total is None:
        stats.error = "the statistics table in XSCALE.LP has no total row"
    elif not stats.shells:
        stats.error = "the statistics table in XSCALE.LP has no resolution shells"
    return stats


def format_log_block(stats: XscaleStats) -> List[str]:
    """The lines autoprocess.log shows for a scaled data set.

    Raises ValueError when ``stats`` is not usable; callers print
    "XSCALE statistics are not available: <reason>" instead.
    """
    if not stats.ok:
        raise ValueError(stats.error)
    total = stats.total
    return [
        f"At {stats.resolution} Å total statistics are:",
        f"Observed reflections: {total.text('observed')}",
        f"Unique reflections: {total.text('unique')}",
        f"Possible: {total.text('possible')}",
        f"Completeness: {total.percent('completeness')}",
        f"R-factor: {total.percent('r_obs')}",
        f"R-meas: {total.percent('r_meas')}",
        f"I/sigma: {total.text('i_sigma')}",
        f"CC1/2: {total.percent('cc_half')}",
    ]

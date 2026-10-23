"""
autoprocess-summary: discovery, labels, ordering, output file, exit codes and output format.
"""
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from pyautoprocess import summary
from pyautoprocess.summary import find_datasets, format_cell, main, natural_key, write_summary

DATA = Path(__file__).parent / "data"
FULL = DATA / "xscale" / "XSCALE.LP"
WEAK = DATA / "xscale" / "XSCALE_weak.LP"
GOLDEN = DATA / "summary_golden.txt"


def dataset(parent: Path, name: str, lp=FULL) -> Path:
    """parent/name/auto_process/XSCALE.LP (no XSCALE.LP when lp is None)."""
    auto = parent / name / "auto_process"
    auto.mkdir(parents=True)
    if lp is not None:
        shutil.copy(lp, auto / "XSCALE.LP")
    return parent / name


def reyes_session(root: Path) -> Path:
    """A REyes-style session: movies/<movie folder>/<name>/auto_process/XSCALE.LP."""
    movies = root / "movies"
    for n in (10, 2, 1):
        dataset(movies / f"sample_420mm_mov{n}_sweep1_c", f"sample-mov{n}")
    dataset(movies / "sample_420mm_mov3_sweep1_c", "sample-mov3", lp=None)    # failed
    dataset(movies / "sample_420mm_mov1_sweep1_c" / "sample-mov1" / "processing_backups", "old")
    dataset(movies / ".hidden", "sample-mov99")
    return root


def labels(datasets):
    return [d.label for d in datasets]


# ------------------------------------------------------------------ discovery

def test_session_layout(tmp_path):
    found = find_datasets(reyes_session(tmp_path / "session"))
    assert labels(found) == ["sample-mov1", "sample-mov2", "sample-mov3", "sample-mov10"]
    assert [d.scaled for d in found] == [True, True, False, True]
    assert found[0].relative == "movies/sample_420mm_mov1_sweep1_c/sample-mov1"


def test_flat_layout(tmp_path):
    for name in ("b-mov2", "a-mov1", "b-mov10"):
        dataset(tmp_path, name)
    assert labels(find_datasets(tmp_path)) == ["a-mov1", "b-mov2", "b-mov10"]


def test_backups_and_hidden_folders_are_skipped(tmp_path):
    found = find_datasets(reyes_session(tmp_path))
    assert "old" not in labels(found) and "sample-mov99" not in labels(found)


def test_depth_bound(tmp_path):
    reyes_session(tmp_path / "session")
    # datasets sit 4 levels below tmp_path: session/movies/<folder>/<name>
    assert len(find_datasets(tmp_path, max_depth=4)) == 4
    assert find_datasets(tmp_path, max_depth=3) == []


def test_search_directory_that_is_itself_a_dataset(tmp_path):
    root = dataset(tmp_path, "sample-mov7")
    found = find_datasets(root, max_depth=0)
    assert labels(found) == ["sample-mov7"] and found[0].scaled


def test_duplicate_names_get_relative_paths(tmp_path):
    dataset(tmp_path / "grid-1", "sample-mov1")
    dataset(tmp_path / "grid-2", "sample-mov1")
    dataset(tmp_path / "grid-2", "sample-mov2")
    assert labels(find_datasets(tmp_path)) == [
        "grid-1/sample-mov1", "grid-2/sample-mov1", "sample-mov2"]


def test_duplicate_labels_use_the_shortest_distinguishing_path(tmp_path):
    dataset(tmp_path / "a" / "movie_folder", "sample-mov1")
    dataset(tmp_path / "b" / "movie_folder", "sample-mov1")
    found = find_datasets(tmp_path)
    assert labels(found) == ["a/movie_folder/sample-mov1", "b/movie_folder/sample-mov1"]


def test_natural_key():
    names = ["mov10", "mov2", "Mov1", "mov2b", "mov"]
    assert sorted(names, key=natural_key) == ["mov", "Mov1", "mov2", "mov2b", "mov10"]


def test_unparsable_xscale_lp_still_listed(tmp_path):
    folder = dataset(tmp_path, "sample-mov1")
    (folder / "auto_process" / "XSCALE.LP").write_text(" ***** XSCALE *****\n")
    (found,) = find_datasets(tmp_path)
    assert found.scaled and found.stats.error == "no statistics table in XSCALE.LP"


# ------------------------------------------------------------------ CLI

def test_main_writes_default_file(tmp_path, capsys):
    root = reyes_session(tmp_path / "session")
    assert main(["--dir", str(root)]) == 0
    out = root / "Summary_Results.txt"
    text = out.read_text()
    assert "  Datasets  : 3\n" in text and "  Not scaled: 1\n" in text
    glance = text.split("QUICK-GLANCE TOTALS")[1].split("NOT SCALED")[0]
    rows = [line.split()[0] for line in glance.splitlines() if line.startswith("  sample-")]
    assert rows == ["sample-mov1", "sample-mov2", "sample-mov10"]
    assert "NOT SCALED (auto_process/ without XSCALE.LP)\n" + "-" * 120 + "\n  sample-mov3\n" in text
    assert "[  3 / 3]  sample-mov10" in text
    printed = capsys.readouterr().out
    assert f"Wrote: {out}" in printed
    assert "Not scaled (1)" in printed


def test_main_out_option(tmp_path):
    reyes_session(tmp_path / "session")
    target = tmp_path / "reports" / "mine.txt"
    target.parent.mkdir()
    assert main(["--dir", str(tmp_path / "session"), "--out", str(target)]) == 0
    assert target.exists()
    assert not (tmp_path / "session" / "Summary_Results.txt").exists()


def test_main_depth_option(tmp_path):
    reyes_session(tmp_path / "session")
    assert main(["--dir", str(tmp_path), "--depth", "3"]) == 1
    assert not (tmp_path / "Summary_Results.txt").exists()
    assert main(["--dir", str(tmp_path), "--depth", "4"]) == 0


def test_main_cwd_default(tmp_path, monkeypatch):
    dataset(tmp_path, "sample-mov1")
    monkeypatch.chdir(tmp_path)
    assert main([]) == 0
    assert (tmp_path / "Summary_Results.txt").exists()


@pytest.mark.parametrize("make", [
    lambda p: p,                                                    # empty
    lambda p: dataset(p, "sample-mov1", lp=None).parent,            # only not scaled
])
def test_main_no_datasets_exit_1(tmp_path, make, capsys):
    root = make(tmp_path)
    assert main(["--dir", str(root)]) == 1
    assert not (root / "Summary_Results.txt").exists()
    assert "No datasets found." in capsys.readouterr().out


def test_main_missing_directory_exit_1(tmp_path):
    assert main(["--dir", str(tmp_path / "nope")]) == 1


def test_main_negative_depth_exit_1(tmp_path):
    assert main(["--dir", str(tmp_path), "--depth", "-1"]) == 1


def test_main_unwritable_output_exit_1(tmp_path):
    dataset(tmp_path, "sample-mov1")
    assert main(["--dir", str(tmp_path), "--out", str(tmp_path / "no" / "such" / "dir.txt")]) == 1


def test_help_mentions_examples(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "autoprocess-summary --dir /data/session" in text
    assert "--depth N" in text and "--out PATH" in text


def test_module_entry_point(tmp_path):
    dataset(tmp_path, "sample-mov1")
    result = subprocess.run([sys.executable, "-m", "pyautoprocess.summary", "--dir", str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "Summary_Results.txt").exists()


# ------------------------------------------------------------------ format

def golden_tree(root: Path) -> Path:
    dataset(root, "sample-mov1", FULL)
    dataset(root, "sample-mov2", WEAK)
    dataset(root, "sample-mov3", lp=None)
    return root


def test_golden_output(tmp_path):
    root = golden_tree(tmp_path / "tree")
    out = tmp_path / "summary.txt"
    write_summary(find_datasets(root), out, Path("/data/session"), now=datetime(2026, 1, 2, 3, 4, 5))
    assert out.read_text() == GOLDEN.read_text()


def test_golden_output_through_main(tmp_path):
    """Same text from the CLI once the timestamp and directory are masked."""
    root = golden_tree(tmp_path / "tree")
    assert main(["--dir", str(root)]) == 0
    text = (root / "Summary_Results.txt").read_text()
    text = re.sub(r"(  Generated : ).*", r"\g<1>2026-01-02 03:04:05", text)
    text = text.replace(f"  Directory : {root.resolve()}", "  Directory : /data/session")
    assert text == GOLDEN.read_text()


def test_console_script_is_declared():
    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    assert 'autoprocess-summary = "pyautoprocess.summary:main"' in pyproject
    assert callable(summary.main)


def test_format_cell():
    assert format_cell("3.72     3.96    13.88  90.000  90.000  90.000") == (
        "   3.72   3.96  13.88  90.000  90.000  90.000")
    assert len(format_cell("124.10 6.6 101.2 114.215 77.562 109.916")) == 45
    assert format_cell("not a cell") == "not a cell"
    assert format_cell("1 2 3") == "1 2 3"
    assert format_cell(None) == "n/a"


def test_quick_glance_rows_align_with_mixed_widths(tmp_path):
    """Mixed label lengths, 1- to 3-digit cell lengths, 2- vs 3-digit angles and a long
    duplicate label: every row has the header's length, columns line up, dividers fit."""
    cells = ["3.72 3.96 13.88 90.000 90.000 90.000",
             "124.10 106.60 101.20 114.215 77.562 109.916",
             "8.31 21.68 44.46 99.551 88.130 104.698"]
    names = ["a", "sample-with-a-much-longer-name-mov12", "b-mov2"]
    for name, cell in zip(names, cells, strict=True):
        folder = dataset(tmp_path / "x", name)
        lp = folder / "auto_process" / "XSCALE.LP"
        text = lp.read_text().replace(
            "3.72     3.96    13.88  90.000  90.000  90.000", cell)
        lp.write_text(text)
    dataset(tmp_path / "very_long_movie_folder_name_number_one_sweep1", "dup", WEAK)
    dataset(tmp_path / "very_long_movie_folder_name_number_two_sweep1", "dup", WEAK)
    out = tmp_path / "s.txt"
    write_summary(find_datasets(tmp_path), out, tmp_path, now=datetime(2026, 1, 1))
    lines = out.read_text().split("QUICK-GLANCE TOTALS")[1].split("\n")
    block = lines[:lines.index(next(l for l in lines[3:] if l.startswith("="))) + 1]
    divider = block[2]
    header = block[3]
    rows = [l for l in block[5:] if l and not l.startswith("=")]
    assert len(rows) == 5
    assert {len(r) for r in rows} == {len(header)}
    assert len(divider) == max(120, len(header)) and set(divider) == {"-"}
    assert len(block[-1]) == len(divider)
    # right-aligned numeric columns end at the same position as their header titles
    for title in ("Hi-Res", "Cmpl%", "R-meas%", "I/Sigma", "CC1/2"):
        end = header.index(title) + len(title)
        for r in rows:
            assert r[end - 1] != " " and (end == len(r) or r[end] == " ")
    # the unit cell column starts at the same place in every row
    start = header.index("Unit Cell")
    assert all(r[start - 1] == " " for r in rows)
    sg_start = header.index("SG")
    assert all(r[sg_start] != " " for r in rows)


# ------------------------------------------------------------------ robustness

needs_posix_permissions = pytest.mark.skipif(
    sys.platform.startswith("win") or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs POSIX permissions and a non-root user")


@needs_posix_permissions
def test_unreadable_folder_is_skipped(tmp_path, capsys):
    dataset(tmp_path / "a", "sample-mov1")
    locked = tmp_path / "b" / "locked"
    dataset(locked, "sample-mov2")
    locked.chmod(0)
    try:
        unreadable = []
        found = find_datasets(tmp_path, unreadable=unreadable)
        assert labels(found) == ["sample-mov1"]
        assert unreadable == [locked]
        assert main(["--dir", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "Skipped 1 unreadable folder(s):" in out and str(locked) in out
    finally:
        locked.chmod(0o755)


@needs_posix_permissions
def test_unreadable_auto_process_does_not_crash(tmp_path):
    folder = dataset(tmp_path, "sample-mov1")
    dataset(tmp_path, "sample-mov2")
    (folder / "auto_process").chmod(0)
    try:
        assert main(["--dir", str(tmp_path)]) == 0
    finally:
        (folder / "auto_process").chmod(0o755)


def test_natural_key_with_non_ascii_digits(tmp_path):
    assert sorted(["mov2", "1²", "mov10"], key=natural_key) == ["1²", "mov2", "mov10"]
    dataset(tmp_path, "1²")
    dataset(tmp_path, "mov2")
    assert labels(find_datasets(tmp_path)) == ["1²", "mov2"]


@pytest.mark.skipif(sys.platform.startswith("win"), reason="symlinks")
def test_symlinked_dataset_is_listed_once(tmp_path):
    dataset(tmp_path / "grid-2", "sample-mov1")
    (tmp_path / "latest").symlink_to(tmp_path / "grid-2", target_is_directory=True)
    (tmp_path / "loop").symlink_to(tmp_path, target_is_directory=True)     # bounded by --depth
    found = find_datasets(tmp_path, max_depth=6)
    assert [d.relative for d in found] == ["grid-2/sample-mov1"]   # the path without links wins


@pytest.mark.skipif(sys.platform.startswith("win"), reason="symlinks")
def test_dataset_only_reachable_through_a_link_is_found(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    dataset(elsewhere, "sample-mov1")
    session = tmp_path / "session"
    session.mkdir()
    (session / "linked").symlink_to(elsewhere, target_is_directory=True)
    assert [d.relative for d in find_datasets(session)] == ["linked/sample-mov1"]


def test_out_existing_directory(tmp_path):
    dataset(tmp_path / "data", "sample-mov1")
    reports = tmp_path / "reports"
    reports.mkdir()
    assert main(["--dir", str(tmp_path / "data"), "--out", str(reports)]) == 0
    assert (reports / "Summary_Results.txt").is_file()


def test_failed_write_leaves_no_file(tmp_path, monkeypatch):
    dataset(tmp_path, "sample-mov1")
    out = tmp_path / "Summary_Results.txt"
    out.write_text("previous summary\n")

    def boom(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(summary, "write_dataset", boom)
    assert main(["--dir", str(tmp_path)]) == 1
    assert out.read_text() == "previous summary\n"            # old file untouched
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


@pytest.mark.skipif(sys.platform in ("darwin", "win32"),
                    reason="needs a filesystem that accepts non-UTF-8 names")
def test_non_utf8_folder_name(tmp_path, capsys):
    name = os.fsdecode(b"sample-\xff-mov1")
    dataset(tmp_path, name)
    assert main(["--dir", str(tmp_path)]) == 0
    assert b"sample-\xff-mov1" in (tmp_path / "Summary_Results.txt").read_bytes()
    assert "sample-" in capsys.readouterr().out


def test_unencodable_label_in_console(tmp_path, monkeypatch, capsys):
    """A label that the console cannot encode is printed with replacements, not a crash."""
    dataset(tmp_path, "sample-é-mov1")
    assert summary._printable("sample-\udcff-mov1") == "sample-�-mov1"
    assert main(["--dir", str(tmp_path)]) == 0


def test_surrogate_escaped_label_is_written_as_original_bytes(tmp_path):
    """Names that are not valid UTF-8 reach Python as surrogate escapes; the file gets the
    original bytes back instead of failing with UnicodeEncodeError."""
    dataset(tmp_path / "tree", "sample-mov1")
    found = find_datasets(tmp_path / "tree")
    found[0].label = os.fsdecode(b"sample-\xff-mov1")
    out = tmp_path / "s.txt"
    write_summary(found, out, tmp_path, now=datetime(2026, 1, 1))
    assert b"sample-\xff-mov1" in out.read_bytes()

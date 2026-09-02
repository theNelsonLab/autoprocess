"""
Output must be written next to the path the USER gave, not next to a symlink's target.

`Path.resolve()` follows symlinks. Because every output directory (`images/`,
`auto_process/`, `processing_backups/`) is created beside the source path, resolving the
input made a symlinked movie write into the link's target directory -- silently, and often
somewhere the user believes is read-only.

This was found the hard way: a validation run that mirrored read-only archive data by
symlink wrote three concurrent processing trees back into the archive, and the arms then
raced each other on the same resolved path.

Identity is a different question from location: the tracking log still resolves, because
there the point is "is this the same physical file", not "where should output go". These
tests pin the distinction so a future tidy-up does not collapse the two.
"""
import os
from pathlib import Path

import pytest

from pyautoprocess.autoprocess import CrystallographyProcessor
from pyautoprocess.config.config_manager import ConfigLoader


def make_processor(paths):
    params = ConfigLoader().get_config('default')
    params.paths = [str(p) for p in paths]
    return CrystallographyProcessor(params)


@pytest.fixture
def linked_movie(tmp_path):
    """A movie in `archive/`, reachable through a symlink in `workdir/`."""
    archive = tmp_path / "archive"
    workdir = tmp_path / "workdir"
    archive.mkdir()
    workdir.mkdir()

    real = archive / "sample-mov1_960_0.3_3_P50toN-50.mrc"
    real.write_bytes(b"not a real movie, never opened by these tests")

    link = workdir / real.name
    link.symlink_to(real)
    return real, link, archive, workdir


def test_symlinked_file_keeps_the_user_supplied_location(linked_movie):
    real, link, archive, workdir = linked_movie
    found = make_processor([link])._get_files_to_process()

    assert len(found) == 1
    resolved_parent = Path(found[0]).parent
    assert resolved_parent == workdir, (
        f"output would be written to {resolved_parent}, i.e. into the archive, "
        f"instead of the directory the user pointed at ({workdir})")
    assert Path(found[0]).parent != archive


def test_symlinked_directory_keeps_the_user_supplied_location(linked_movie):
    """The same hazard one level up: a symlink to a whole dataset directory."""
    real, link, archive, workdir = linked_movie
    link.unlink()
    dir_link = workdir / "linked_archive"
    dir_link.symlink_to(archive, target_is_directory=True)

    found = make_processor([dir_link])._get_files_to_process()

    assert len(found) == 1
    assert Path(found[0]).parent == dir_link
    assert archive not in Path(found[0]).parents


def test_relative_paths_are_still_absolutised(linked_movie, monkeypatch):
    """abspath must still normalise '..' and make the path absolute -- just without
    following links, which is the only behaviour we changed."""
    real, link, archive, workdir = linked_movie
    monkeypatch.chdir(workdir)

    found = make_processor([Path("..") / "workdir" / link.name])._get_files_to_process()

    assert len(found) == 1
    assert Path(found[0]).is_absolute()
    assert ".." not in found[0]
    assert Path(found[0]).parent == workdir


def test_a_real_path_is_unaffected(linked_movie):
    """The fix must not disturb the ordinary, non-symlinked case."""
    real, link, archive, workdir = linked_movie
    found = make_processor([real])._get_files_to_process()
    assert [Path(p) for p in found] == [real]


def test_two_links_to_one_movie_stay_independent(tmp_path):
    """The concrete failure from the validation run: separate arms must not collide.

    Two directories each holding a link to the same movie must yield two distinct output
    locations. Under resolve() both collapsed to the single archive directory, so the arms
    overwrote each other's XDS files mid-run.
    """
    archive = tmp_path / "archive"
    archive.mkdir()
    real = archive / "sample-mov1_960_0.3_3_P50toN-50.mrc"
    real.write_bytes(b"x")

    arms = []
    for name in ("armA", "armB", "armC"):
        arm = tmp_path / name
        arm.mkdir()
        (arm / real.name).symlink_to(real)
        arms.append(arm / real.name)

    found = make_processor(arms)._get_files_to_process()
    parents = {Path(p).parent for p in found}

    assert len(found) == 3
    assert len(parents) == 3, f"arms collapsed onto {parents}"
    assert archive not in parents


def test_tracking_log_still_resolves_for_identity(tmp_path):
    """The deliberate other half: dedupe SHOULD see a link and its target as one file."""
    from pyautoprocess.core.process_tracker import ProcessTracker

    archive = tmp_path / "archive"
    workdir = tmp_path / "workdir"
    archive.mkdir()
    workdir.mkdir()
    real = archive / "sample-mov1_960_0.3_3_P50toN-50.mrc"
    real.write_bytes(b"x")
    link = workdir / real.name
    link.symlink_to(real)

    tracker = ProcessTracker()
    monkey_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        tracker.add_to_processed_files_log(real, workdir / "sample-mov1")
        assert tracker.is_file_already_processed(link), (
            "a symlink to an already-processed movie should still count as processed")
    finally:
        os.chdir(monkey_cwd)

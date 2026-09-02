"""
--seed makes the indexing-retry search reproducible.

When first-pass indexing fails, `_handle_missing_xparm` rewrites BACKGROUND_PIXEL,
SIGNAL_PIXEL and MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT with random values and retries, up to ten
times. Unseeded, that makes such a dataset genuinely irreproducible: four v0.4.2 runs of one
real dataset gave ISa 1.06-3.39, Rmeas 72.9-257.2%, and space group 1 or 5 with a different
unit cell each time. --seed pins the search without changing its strategy.

These tests exercise the RNG rather than XDS: the retry loop itself needs a real XDS run, but
the reproducibility guarantee lives entirely in how the RNG is seeded.
"""
import pytest

from pyautoprocess.autoprocess import CrystallographyProcessor
from pyautoprocess.config.config_manager import ConfigLoader


def make_processor(seed=None):
    params = ConfigLoader().get_config('default')
    params.seed = seed
    return CrystallographyProcessor(params)


def draw(processor, n=30):
    """The same three draws per attempt that _handle_missing_xparm makes."""
    return [(processor._rng.randrange(3, 5, 1),
             processor._rng.randrange(4, 9, 1),
             processor._rng.randrange(5, 9, 1)) for _ in range(n)]


def test_same_seed_and_dataset_reproduce_exactly():
    a, b = make_processor(seed=42), make_processor(seed=42)
    a.reseed_for_dataset("sample-mov1")
    b.reseed_for_dataset("sample-mov1")
    assert draw(a) == draw(b)


def test_different_seeds_diverge():
    a, b = make_processor(seed=1), make_processor(seed=2)
    a.reseed_for_dataset("sample-mov1")
    b.reseed_for_dataset("sample-mov1")
    assert draw(a) != draw(b)


def test_different_datasets_diverge_under_one_seed():
    """Otherwise every movie in a run would try the identical retry parameters."""
    a, b = make_processor(seed=42), make_processor(seed=42)
    a.reseed_for_dataset("sample-mov1")
    b.reseed_for_dataset("sample-mov2")
    assert draw(a) != draw(b)


def test_a_dataset_is_independent_of_what_preceded_it():
    """The reason for seeding per dataset rather than per run.

    `autoprocess a.mrc b.mrc` and `autoprocess b.mrc` must give b the same retries; with a
    single per-run stream, a's draws would shift b's.
    """
    both = make_processor(seed=42)
    both.reseed_for_dataset("movie-a")
    draw(both, 17)                       # movie-a consumes an arbitrary amount
    both.reseed_for_dataset("movie-b")
    after_a = draw(both)

    alone = make_processor(seed=42)
    alone.reseed_for_dataset("movie-b")
    assert after_a == draw(alone)


def test_unseeded_runs_differ_from_each_other():
    """No --seed must mean the historical behaviour, not a hidden default seed."""
    results = []
    for _ in range(5):
        processor = make_processor(seed=None)
        processor.reseed_for_dataset("sample-mov1")
        results.append(draw(processor, 40))
    assert len({tuple(r) for r in results}) > 1, "unseeded runs should not all agree"


def test_draws_stay_in_the_documented_ranges():
    """The seed pins the search; it must not change what the search explores."""
    processor = make_processor(seed=7)
    processor.reseed_for_dataset("sample-mov1")
    for background, signal, min_pixels in draw(processor, 200):
        assert background in (3, 4)
        assert signal in range(4, 9)
        assert min_pixels in range(5, 9)


def test_seed_defaults_to_none_and_is_parsed():
    import sys
    from pyautoprocess.ui.cli_parser import parse_arguments

    sys.argv = ['autoprocess']
    assert parse_arguments('autoprocess').seed is None

    sys.argv = ['autoprocess', '--seed', '1234']
    assert parse_arguments('autoprocess').seed == 1234

    sys.argv = ['image_process', '--seed', '99']
    assert parse_arguments('image_process').seed == 99


def test_processor_is_usable_before_any_reseed():
    """image_process and batch_reprocess drive process_check() directly, without going through
    _process_single_movie, so the RNG must be valid straight out of __init__."""
    processor = make_processor(seed=5)
    assert draw(processor, 3)

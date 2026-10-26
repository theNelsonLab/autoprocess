# Changelog

Earlier versions are listed under "Version History" in README.md.

## 0.5.5

### Added
- `autoprocess.log` shows the XSCALE total statistics after scaling: observed, unique and
  possible reflections, completeness, R-factor, R-meas, I/sigma and CC1/2, at the resolution of the
  last shell XSCALE reported. When there are none, one line says why
  ("XSCALE statistics are not available: ..."); this never changes the outcome of a run.
- `autoprocess-summary`: collects the XSCALE statistics of every dataset under a directory
  (`<dataset>/auto_process/XSCALE.LP`, searched recursively) into `Summary_Results.txt`.
- `pyautoprocess.core.xscale_stats`: the XSCALE.LP parser behind both.

### Changed
- In `autoprocess.log` the "I found space group ... and a unit cell of" lines now follow
  "I scaled the data in XSCALE." (directly above the statistics) instead of preceding
  "Proceeding with scaling". Every path that logged them still does. `stats.LP`, `XSCALE.INP`
  and exit codes are unchanged.
- `--beam-center` (experimental, off by default): the halo-circle RANSAC fit uses a named seed
  constant with a new value.

### Fixed
- TIF conversion: a value above 65535 after the pedestal (+200 for `.ser`, +1 for `.mrc` and
  `.tvips` in `autoprocess`; `--ped` in the tools) wrapped around to a small number (65536 became
  0; a raw `.ser` 65535 became 199) instead of being clipped. Such pixels are now stored as 65535,
  and `autoprocess` logs how many were clipped ("WARNING: <n> pixel(s) in <k> frame(s) of <file>
  exceeded 65535 after the pedestal and were clipped to 65535 ..."). Applies to `autoprocess`, its
  verification and `mrc2tif`, `ser2tif`, `tvips2tif` (`mrc2tif` already clipped).
- Float frames (MRC modes 2 and 12) are rounded to the nearest integer (halves to even) instead of
  truncated toward zero. This can change TIF values by at most 1 count for float MRC data;
  integer data within range and whole-number floats convert exactly as before.
- README: unclosed code block in the installation section; `--ped` defaults of `ser2tif` and
  `tvips2tif` (200, not 0).

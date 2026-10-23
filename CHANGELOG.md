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
- README: unclosed code block in the installation section.

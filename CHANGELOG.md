# Changelog

Changes since the last push to `main` on GitHub. Versions follow
`spbin.__version__`.

## Unreleased

### Added

- `pms` region in `locate` and `fit_star`, with a youth correction of the
  network for pre-main-sequence stars. The single-star fit starts from the
  best seed of every age.
- JAX port of the `pms` route in the white-dwarf frame.
- `fit_star` returns an MS+MS threshold of 102 when extinction is fitted
  under `dust_prior`.
- `scripts/ms_threshold_dust.py` refits the threshold controls and the Gaia
  SB2 completeness under the Edenhofer dust prior.
- `docs/overview.md` holds the model description, the validation numbers and
  the list of cases not covered.
- GitHub Actions workflow that runs `ruff check .` and `pytest -q` on
  Python 3.10 and 3.12 for pushes to `main` and for pull requests.
- `ruff` configuration in `pyproject.toml`; `ruff` joins the `test` extra.

### Changed

- The README is 108 lines: install, the `fit` and `fit_star` examples, the
  region table, documentation links and credits. Its longer sections are in
  `docs/overview.md`.
- 2MASS bands enter a fit only with clean contamination flags (`cc_flg` and
  `gal_contam` from the IRSA point source catalogue).
- `fit_star` returns a threshold only for the configuration and the
  extinction prior it was calibrated under. Other configurations return no
  threshold.
- `StellarModel(pms=True)` is refused where no pms fit is set up, and
  `hot_ms` is refused together with `use_wise`.
- `galex()` matches GALEX through `crossmatch_uv`.
- The package version is defined once, in `spbin.__version__`.

### Fixed

- `download` requests XP spectra with the Gaia archive's `DATAMODEL_GAIA`
  data structure.
- The JAX fit holds a fixed age and [M/H] at the requested values.
- The JAX white-dwarf frame matches NumPy at missing table neighbours and on
  `log_age`.
- `fit_star` keeps a parallax fit request in the giant region.
- The cool-star UV extrapolation is reddened at its own wavelengths.
- GALEX pivot wavelengths are integrated with SciPy's `trapezoid`.
- A mass error is NaN when a one-sigma step leaves the orbit branch.
- `scripts/bench_fit.py` places mock pairs at one distance for flux and
  parallax.
- The `fg_ms` star's age is described as free, not coeval with the giant.

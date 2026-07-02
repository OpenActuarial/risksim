# Changelog

## 0.3.0

### Changed
- **Breaking (numeric):** `var` and `tvar` follow the ecosystem-wide
  convention: VaR is the inverted-CDF order statistic x_(ceil(nq)) and TVaR
  the Acerbi-Tasche average-quantile estimator, so TVaR(q) >= VaR(q) always,
  including at exact integer nq boundaries. Conformance is pinned by a test
  shared byte-for-byte with lossmodels and extremeloss.

### Added
- `exceedance_probability` alias and vectorized `q` support in the metrics
  module.

### Fixed
- Layer/component name mappings in `SimulationResult` now zip strictly,
  surfacing length mismatches instead of silently truncating.

## 0.2.1

### Fixed (packaging)
- Added an explicit `[build-system]` table (setuptools) and `[tool.setuptools]`
  src-layout configuration. The package built previously only via pip's legacy
  fallback; the build is now explicit and reproducible (e.g. with `python -m build`).
- Corrected stale repository links in the README (`michaelabryant` -> `actuarialpy`).

## 0.2.0

### Fixed
- `metrics.summary` (and `SimulationResult.summary`) produced colliding,
  mislabeled keys for sub-percent quantiles: both `0.995` and `0.999` mapped to
  `var_100` / `tvar_100`, silently overwriting one another. Percentile labels now
  preserve sub-percent precision (`0.995 -> "99.5"`, `0.999 -> "99.9"`); integer
  percentiles are unchanged (`0.95 -> "95"`, `0.99 -> "99"`), so existing default
  output is unaffected.
- `AggregateLayer.attachment_probability` now reports `P(loss > attachment)`
  directly, instead of `P(ceded > 0)`, which collapsed to 0 under degenerate
  parameters such as `share = 0` or `limit = 0`.
- Version mismatch between `__init__.__version__` (was `0.1.0`) and
  `pyproject.toml` (was `0.1.1`); both are now `0.2.0`.

### Added
- `apply_contract` is now exported at the top level (`from risksim import apply_contract`).

### Changed
- Consolidated the duplicated `_as_1d_float_array` input validator (previously
  copied in `metrics`, `contracts`, and `results`) into a single internal
  `risksim._validation` module. No behavior change.

# Changelog

## 0.5.3

Bump version number to update PyPI with updated README.

## 0.5.2

Documentation only; no behavior changes.

### Fixed
- `metrics.mean`, `metrics.variance`, `metrics.std`, `metrics.prob_exceeding`,
  and `metrics.summary` had no docstrings and were silently omitted from the
  autodoc API reference (only `var` and `tvar` appeared). All five are now
  documented in the module's style, and `metrics` gained an explicit
  `__all__` covering the eight public names including the
  `exceedance_probability` alias.

## 0.5.1

Documentation only; no library changes.

### Fixed
- The `Portfolio` docstring and the README/PyPI overview described the
  package as independence-only, predating the 0.5.0 `dependence` module.
  Both now document the default-independent sampler alongside the
  supported `impose_rank_correlation` workflow; `Portfolio.variance` /
  `.std` / `.summary` are noted as independence-only closed forms
  unaffected by downstream reordering; and the README package-structure
  list now includes `dependence.py` and `uncertainty.py` (the latter
  stale since 0.4.0). No behavior change -- 0.5.0 code is unaffected.

## 0.5.0

### Added
- **`risksim.dependence`: correlation between components.**
  Independence across portfolio components overstates diversification in
  exactly the tail metrics this package computes.
  `impose_rank_correlation` adds dependence by Iman-Conover reordering:
  simulate each component with the existing untouched machinery, then
  permute columns to a target rank correlation -- marginals preserved
  *exactly*, numpy-only, post-hoc to any sampler. Two limits stated
  loudly in the docs: rank correlation is not tail dependence (normal
  scores give asymptotically independent joint extremes at any rho; use
  `scores="t"` with small `df` when the question is whether components
  blow up together), and this imposes asserted dependence rather than
  estimating it.


## 0.3.1

### Added

- `Portfolio.sample`, `Portfolio.sample_components`, `Portfolio.simulate`,
  and `PortfolioItem.sample` accept the shared `rng` argument; one resolved
  generator is threaded through the components so composed simulations are
  bit-reproducible. Models without an `rng` parameter keep working when
  `rng` is omitted.

### Added

- Conformance, identity, and integration test suites (scipy/closed-form
  conformance, mathematical identities, cross-package seams). Example
  scripts are now executed by the test suite.

### Changed

- More descriptive package `description` metadata.

## 0.3.0

### Changed
## 0.4.0

### Added
- **`risksim.uncertainty`: Monte Carlo error quantification.** A simulated
  VaR without an error estimate is a random number with confidence; this
  module answers "how much of this is signal" for a loss vector of any
  origin. `mean_ci` (normal theory), `quantile_ci` (distribution-free
  order-statistic interval on the same `ceil(n*q)` rank convention as
  `metrics.var`; `se` is deliberately `nan` because a quantile has no
  distribution-free standard error -- the interval is the uncertainty
  statement), `bootstrap_ci` (percentile bootstrap for anything, TVaR in
  particular, reproducible via `rng`), and `summary_with_error`, whose
  point estimates match `metrics.summary` exactly with each metric
  carrying the interval its sampling theory supports.


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

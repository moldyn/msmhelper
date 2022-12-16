# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and [Element](https://github.com/vector-im/element-android)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

[//]: # (Available sections in changelog)
[//]: # (### API changes warning ⚠️:)
[//]: # (### Added Features and Improvements 🙌:)
[//]: # (### Bugfix 🐛:)
[//]: # (### Other changes:)


## [Unreleased]
### API changes warning ⚠️:
- Added `dyncor` module containing an implementation of dynamical coring
- Removed Python 3.6 support.

### Added Features and Improvements 🙌:
- Add all new mkdocs documentation 🎉


## [0.6.2] - 2022-09-20
### Other changes:
- Moved to package to src directory
- Add `nvals` parameter to all eigenvalues/-vectors functions
- Add `ntimescales` parameter to `implied_timescales`
- Improved tests for `implied_timescales`

### Bugfix 🐛:
- Fix complex matrices for LumpedStateTraj due to complex eigenvalues


## [0.6.1] - 2022-08-01
### Bugfix 🐛:
- Include extra-requirements via MANIFEST.in


## [0.6.0] - 2022-08-01
### API changes warning ⚠️:
- Remove `pyemma` to ensure better pip support
- Remove `build_MSM` use instead `estimate_markov_model`

### Added Features and Improvements 🙌:
- Upload test coverage to Codecov
- Add extra-requirements to pip installation
- Added Changelog file :tada:
- Add new parameter `allow_non_ergodic` to `mh.equilibrium_population`
- Upload docs to gh-pages :rocket:
- Add module `md` with functions for estimating timescales and pathways

### Other changes:
- Added `beartype` dependency for adding dynamic type checking
- Parallelize and refactor `compare_discretization`
- replaced decorators with `decorit` package

### Bugfix 🐛:
- Fix source code rendering in documentation
- Fix deprecated warnings of `numpy` and `pytest`
- Fix most LGTM warnings
- Many more minor fixes
- Add parameter `atol` to tests functions to make them more robust


## [0.5.0] - 2021-07-22
### Added Features and Improvements 🙌:
- Add `LumpedStateTraj` class which allows optimal projection of microstate dynamics to macrostates, method taken from Szabo and Hummer
- Add estimation of MD waiting times
- Add tests for `is_transition_matrix`, `is_ergodic`

### Bugfix 🐛:
- Minor improvements and tweaks.

### Other changes:


## [0.4.0] - 2020-09-14
### API changes warning ⚠️:
- Removed `n_iterations` parameter from `Clustering`

### Added Features and Improvements 🙌:
- Added clustering mode `mode='kmedoids'` for k-medoids clustering
- Added tools module to reopen clusters

### Other changes:
- Improved some test functions


## [0.3.0] - 2020-08-25
### Added Features and Improvements 🙌:
- Add `StateTraj` class to speed up calcualations by a factor of 50.
- Refactor code to use/be compatible with `StateTraj` class
- Add `benchmark` module with an numba optimized version of the
Chapman-Kolmogorov test
- Add function for estimating implied timescales

### Bugfix 🐛:
- Minor bug fixes and im

### Other changes:
- Improving documentation/docstrings
- Adding tests including benchmarks


## [0.2.0] - 2020-08-14
### Added Features and Improvements 🙌:
- Major speed improvement by rewriting parts of msm module with numba


## [0.1.0] - 2020-02-20
- Initial release


[Unreleased]: https://github.com/moldyn/msmhelper/compare/v0.6.2...main
[0.6.2]: https://github.com/moldyn/msmhelper/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/moldyn/msmhelper/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/moldyn/msmhelper/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/moldyn/msmhelper/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/moldyn/msmhelper/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/moldyn/msmhelper/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/moldyn/msmhelper/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/moldyn/msmhelper/tree/v0.1.0

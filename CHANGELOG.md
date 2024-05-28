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
### Added Features and Improvements 🙌:
- Supporting latest Python 3.12 🎉


## [1.1.1] - 2023-11-10
### Bugfix 🐛:
- Fix mutable properties of `mh.StateTraj` and `mh.LumpedStateTraj`, #43

### Other changes:
- Improved performance of `mh.LumpedStateTraj.microtrajs`


## [1.1.0] - 2023-11-03
### API changes warning ⚠️:
- Python 3.7 support was dropped!

### Added Features and Improvements 🙌:
- Supporting latest Python 3.11 🎉

### Bugfix 🐛:
- Fixed wrong connectivity in `mh.datasets.utils.propagate_MCMC` for different datatypes, #39


## [1.0.4] - 2023-05-22
### Other changes:
- Improvements of the README and the documentation, suggested by the JOSS reviewers @yuxuanzhuang and @lorenzo-rovigatti
- Added zenodo setup
- Added issue templates


## [1.0.3] - 2023-04-17
### Bugfix 🐛:
- Fix that `mh.LumpedStateTraj.index_trajs` and `mh.LumpedStateTraj.index_trajs_flatten` return now index trajectories corresponding to the macrostate trajectories

### Other changes:
- Improve the x-axis limits, use more distinguishable colors, and lower the number of bins for the MD for the cli `msmhelper waiting-times`
- Updated contribution and added maintenance guidelines
- Minor improvements of docs suggested by JOSS reviewer


## [1.0.2] - 2023-03-13
### Added Features and Improvements 🙌:
- Added cli for `compare-discretization`

### Bugfix 🐛:
- Fix missing import of `msm.utils.linalg` when importing `msm.utils`
- Fix bug of similarity measure (`mh.md.compare_discretization`) being infinite


## [1.0.1] - 2023-03-08
### Other changes:
- Added basic test for `msmhelper waiting-times` command-line interface
- Using scientific y-labels for `msmhelper waiting-times` and `msmhelper waiting-time-dist` command-line interfaces to improve figures for real data

### Bugfix 🐛:
- Fix bug where `--frames-per-unit` and `unit` where neglected in the cli `msmhelper waiting-time-dist`
- Fix undesired behavior where `msmhelper implied-timescales` used gray spines instead of true black


## [1.0.0] - 2023-03-03
### API changes warning ⚠️:
- Completely refactoring of the API, this release has many breaking changes to v0.6.2
  - Renamed module `iotext` to `io`
  - Moved all functions related to msm to `msm` module
  - Moved all functions related to raw state trajectories to `md` module
  - Moved all remaining functions into `utils` module
- Removed `StateTraj.trajs` property to reduce confusion between index and
  state trajectories.
- Removed Python 3.6 support.

### Added Features and Improvements 🙌:
- Add an all new mkdocs documentation with material design 🎉
- Add a command line interface for standalone tasks 🎉
- Add follow-along tutorials, FAQ, and better code references 🎉
- Add the submodule `plot` to create commonly-used figures
- Improved MSM generation by a factor of 2-3 for continuously named states.

### Other changes:
- Add `gaussian_filter` functionality
- Add implementation of `dynamical_coring`
- Add estimation of transition times
- Add contact representation of states
- And many more improvements

### Bugfix 🐛:
- Fixed bug where `propagate_MCMC` returns index trajectory instead of a state trajectory


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


[Unreleased]: https://github.com/moldyn/msmhelper/compare/v1.1.1...main
[1.1.1]: https://github.com/moldyn/msmhelper/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/moldyn/msmhelper/compare/v1.0.4...v1.1.0
[1.0.4]: https://github.com/moldyn/msmhelper/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/moldyn/msmhelper/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/moldyn/msmhelper/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/moldyn/msmhelper/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/moldyn/msmhelper/compare/v0.6.2...v1.0.0
[0.6.2]: https://github.com/moldyn/msmhelper/compare/v0.6.1...v0.6.2
[0.6.2]: https://github.com/moldyn/msmhelper/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/moldyn/msmhelper/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/moldyn/msmhelper/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/moldyn/msmhelper/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/moldyn/msmhelper/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/moldyn/msmhelper/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/moldyn/msmhelper/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/moldyn/msmhelper/tree/v0.1.0

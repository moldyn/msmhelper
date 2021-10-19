![GitHub Workflow Status](https://img.shields.io/github/workflow/status/moldyn/msmhelper/Python%20package)
![GitHub All Releases](https://img.shields.io/github/downloads/moldyn/msmhelper/total)
![GitHub last commit](https://img.shields.io/github/last-commit/moldyn/msmhelper)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/moldyn/msmhelper)
![LGTM Grade](https://img.shields.io/lgtm/grade/python/github/moldyn/msmhelper?label=code%20quality&logo=lgtm)
![wemake-python-styleguide](https://img.shields.io/badge/style-wemake-000000.svg)

# msmhelper

This is a package with helper functions to work with discrete state trajectories and Markov state models. In contrast to `pyemma` and `msmbuilder` it features a very limited set of functionality. But using `numba`, it offers a solid performance and having many raise condition it enforces the correct usage.

# Usage
This package is mainly based on `numpy` and `numba` for all computational complex tasks.
## Usage
```python
import msmhelper as mh
...
```
## Requirements:
- Python 3.6-3.9
- Numba 0.49.0+
- Numpy 1.16.2+

# Changelog:
- tba:
  - Added `beartype` dependency for adding dynamic type checking
  - Remove `pyemma` to ensure better pip support
  - Remove `build_MSM` use instead `estimate_markov_model`
  - Add new function `ergodic_mask`
  - Parallelize and refactor `compare_discretization`
  - Fix deprecated warnings of `numpy` and `pytest`
  - replaced decorators with `decorit` package
  - Add gh-pages
  - Add module `md` with functions for estimating timescales and pathways
- v0.5:
  - Add `LumpedStateTraj` class which allows optimal projection of microstate dynamics to macrostates, method taken from Szabo and Hummer
  - Add estimation of MD waiting times
  - Minor improvements and tweaks.
- v0.4:
  - Add `compare` module to compare two different state discretizations
  - Upgrade pydoc to `0.9.1` with search option and change css style.
- v0.3:
  - Add `StateTraj` class to speed up calcualations by a factor of 50.
  - Refactor code to use/be compatible with `StateTraj` class
  - Add `benchmark` module with an numba optimized version of the Chapman
    Kolmogorov test.
- v0.2:
  - parts of msm module are rewritten in numba
- v0.1:
  - initial release

# Roadmap:
- write roadmap

# Development
## Additional Requirements:
- wemake-python-styleguide
- flake8-spellcheck

## Pytest
Running pytest with numba needs an additional flag
```
export NUMBA_DISABLE_JIT=1 && pytest
```

# Credits:
- [numpy](https://docs.scipy.org/doc/numpy)
- [realpython](https://realpython.com/)

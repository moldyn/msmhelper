![GitHub Workflow Status](https://img.shields.io/github/workflow/status/moldyn/msmhelper/Python%20package)
![GitHub last commit](https://img.shields.io/github/last-commit/moldyn/msmhelper)
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

# Roadmap:
- Add unit tests for all functions
- Add examples usage scripts
- Create typing module

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

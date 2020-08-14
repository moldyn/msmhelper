![GitHub Workflow Status](https://img.shields.io/github/workflow/status/moldyn/msmhelper/Python%20package)
![GitHub All Releases](https://img.shields.io/github/downloads/moldyn/msmhelper/total)
![GitHub last commit](https://img.shields.io/github/last-commit/moldyn/msmhelper)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/moldyn/msmhelper)
![LGTM Grade](https://img.shields.io/lgtm/grade/python/github/moldyn/msmhelper?label=code%20quality&logo=lgtm)
![wemake-python-styleguide](https://img.shields.io/badge/style-wemake-000000.svg)

# msmhelper

This is a package with helper functions to work with state trajectories. Hence, it is mainly used for Markov State Models.

# Usage
This package is mainly based on numpy and numba.
## Usage
```python
import msmhelper
...
```
## Known Bugs
- not known

## Requirements:
- Python 3.6+
- Numba 0.49.0+
- Numpy 1.16.2+
- Pyemma 2.5.7+

# Changelog:
- tba:
  -
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

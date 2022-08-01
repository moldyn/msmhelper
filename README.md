<div align="center">
  <img class="darkmode" style="width: 400px;" src="https://github.com/moldyn/msmhelper/blob/main/docs/logo_large_dark.svg?raw=true#gh-dark-mode-only" />
  <img class="lightmode" style="width: 400px;" src="https://github.com/moldyn/msmhelper/blob/main/docs/logo_large_light.svg?raw=true#gh-light-mode-only" />

  <p>
    <a href="https://github.com/wemake-services/wemake-python-styleguide" alt="wemake-python-styleguide">
        <img src="https://img.shields.io/badge/style-wemake-000000.svg" /></a>
    <a href="https://beartype.rtfd.io" alt="bear-ified">
        <img src="https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg" /></a>
    <a href="https://pypi.org/project/msmhelper" alt="PyPI">
        <img src="https://img.shields.io/pypi/v/msmhelper" /></a>
    <a href="https://anaconda.org/conda-forge/msmhelper" alt="conda version">
	<img src="https://img.shields.io/conda/vn/conda-forge/msmhelper" /></a>
    <a href="https://pepy.tech/project/msmhelper" alt="Downloads">
        <img src="https://pepy.tech/badge/msmhelper" /></a>
    <a href="https://github.com/moldyn/msmhelper/actions/workflows/pytest.yml" alt="GitHub Workflow Status">
        <img src="https://img.shields.io/github/workflow/status/moldyn/msmhelper/Pytest%20with%20Codecov"></a>
    <a href="https://codecov.io/gh/moldyn/msmhelper" alt="Code coverage">
        <img src="https://codecov.io/gh/moldyn/msmhelper/branch/main/graph/badge.svg?token=Ce2eW5JICI" /></a>
    <a href="https://lgtm.com/projects/g/moldyn/msmhelper" alt="LGTM">
	<img src="https://img.shields.io/lgtm/grade/python/github/moldyn/msmhelper" alt="LGTM Grade" /></a>
    <a href="https://img.shields.io/pypi/pyversions/msmhelper" alt="PyPI - Python Version">
        <img src="https://img.shields.io/pypi/pyversions/msmhelper" /></a>
    <a href="https://moldyn.github.io/msmhelper" alt="Docs">
        <img src="https://img.shields.io/badge/pdoc3-Documentation-brightgreen" /></a>
    <a href="https://github.com/moldyn/MoSAIC/blob/main/LICENSE" alt="License">
        <img src="https://img.shields.io/github/license/moldyn/msmhelper" /></a>
  </p>

  <p>
    <a href="https://moldyn.github.io/msmhelper">Docs</a> •
    <a href="#features">Features</a> •
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a>
  </p>
</div>

# msmhelper

> **Warning**
> This package is still in beta stage. Please open an issue if you encounter
> any bug/error.

This is a package with helper functions to work with discrete state
trajectories and Markov state models. In contrast to `pyemma` and `msmbuilder`
it features a very limited set of functionality. This repo is prepared to be
published. In the next weeks the source code will be cleaned up, tutorials will
be added and this readme will be extended.

This package will be published soon:
> D. Nagel, and G. Stock,
> *msmhelper: A Python Package for Markov State Modeling of Protein Dynamics*,
> in preparation

We kindly ask you to cite this article in case you use this software package
for published works.


## Features
- Simple usage with sleek function-based API
- Supports latest Python 3.10
- Extensive [documentation](https://moldyn.github.io/msmhelper) with
  many command line scripts
- ...


## Installation
The package is called `msmhelper` and is available via
[PyPI](https://pypi.org/project/msmhelper) or
[conda](https://anaconda.org/conda-forge/msmhelper). To install it,
simply call:
```bash
python3 -m pip install --upgrade msmhelper
```
or
```
conda install -c conda-forge msmhelper
```

or for the latest dev version
```bash
# via ssh key
python3 -m pip install git+ssh://git@github.com/moldyn/msmhelper.git

# or via password-based login
python3 -m pip install git+https://github.com/moldyn/msmhelper.git
```


## Usage
This package is mainly based on `numpy` and `numba` for all computational complex tasks.
## Usage
```python
import msmhelper as mh
...
```

## Roadmap:
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

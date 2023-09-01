<div align="center">
  <img class="darkmode" style="width: 400px;" src="https://github.com/moldyn/msmhelper/blob/main/docs/logo_large_dark.svg?raw=true#gh-dark-mode-only" />
  <img class="lightmode" style="width: 400px;" src="https://github.com/moldyn/msmhelper/blob/main/docs/logo_large_light.svg?raw=true#gh-light-mode-only" />

  <p>
    <a href="https://joss.theoj.org/papers/0c2a2cd2ca10b5c0bdca4d0234eb94fd">
        <img src="https://joss.theoj.org/papers/0c2a2cd2ca10b5c0bdca4d0234eb94fd/status.svg" /></a>
    <a href="https://github.com/wemake-services/wemake-python-styleguide" alt="wemake-python-styleguide">
        <img src="https://img.shields.io/badge/style-wemake-000000.svg" /></a>
    <a href="https://pypi.org/project/msmhelper" alt="PyPI">
        <img src="https://img.shields.io/pypi/v/msmhelper" /></a>
    <a href="https://anaconda.org/conda-forge/msmhelper" alt="conda version">
        <img src="https://img.shields.io/conda/vn/conda-forge/msmhelper" /></a>
    <a href="https://pepy.tech/project/msmhelper" alt="Downloads">
        <img src="https://static.pepy.tech/badge/msmhelper" /></a>
    <a href="https://github.com/moldyn/msmhelper/actions/workflows/pytest.yml" alt="GitHub Workflow Status">
        <img src="https://img.shields.io/github/actions/workflow/status/moldyn/msmhelper/pytest.yml?branch=main"></a>
    <a href="https://codecov.io/gh/moldyn/msmhelper" alt="Code coverage">
        <img src="https://codecov.io/gh/moldyn/msmhelper/branch/main/graph/badge.svg?token=Ce2eW5JICI" /></a>
    <a href="https://github.com/moldyn/msmhelper/actions/workflows/codeql.yml" alt="CodeQL">
        <img src="https://github.com/moldyn/msmhelper/actions/workflows/codeql.yml/badge.svg?branch=main" /></a>
    <a href="https://img.shields.io/pypi/pyversions/msmhelper" alt="PyPI - Python Version">
        <img src="https://img.shields.io/pypi/pyversions/msmhelper" /></a>
    <a href="https://moldyn.github.io/msmhelper" alt="Docs">
        <img src="https://img.shields.io/badge/mkdocs-Documentation-brightgreen" /></a>
    <a href="https://github.com/moldyn/msmhelper/blob/main/LICENSE" alt="License">
        <img src="https://img.shields.io/github/license/moldyn/msmhelper" /></a>
  </p>

  <p>
    <a href="https://moldyn.github.io/msmhelper">Docs</a> •
    <a href="#features">Features</a> •
    <a href="#installation">Installation</a> •
    <a href="https://moldyn.github.io/msmhelper/faq">FAQ</a>
  </p>
</div>

# msmhelper

This is a package with helper functions to work with discrete state trajectories and Markov state models. In contrast to [pyemma](https://github.com/markovmodel/PyEMMA) and [msmbuilder](https://github.com/msmbuilder/msmbuilder), it focuses on Markov state modeling based on an already existing state trajectory. Therefore, neither dimensionality reduction methods nor clustering methods are included. For a methodological overview, we recommend [Sittel and Stock](https://doi.org/10.1063/1.5049637).

This package is published in:
> **msmhelper: A Python Package for Markov State Modeling of Protein Dynamics**,  
> D. Nagel, and G. Stock,  
> *J. Open Source Soft.* **2023** 8 (85), 5339,  
> doi: [10.21105/joss.05339](https://doi.org/10.21105/joss.05339)

We kindly ask you to cite this article in case you use this software package for published works.

## Features
- Simple usage with sleek function-based API
- High performance due to [numba](https://numba.pydata.org/)-optimized source code, checkout the [benchmark comparing to PyEMMA](https://moldyn.github.io/msmhelper/benchmark)
- [Documentation](https://moldyn.github.io/msmhelper) including tutorials
- Powerful command-line interface (CLI) to create publication-ready figures
- Supports Python 3.8-3.11

## Implemented Key Functionalities
- Hummer-Szabo projection of optimal dimensionality reduction by [Hummer and Szabo 2014](https://doi.org/10.1021/jp508375q)
- Dynamical coring by [Nagel et al. 2019](https://doi.org/10.1063/1.5081767)
- Fast extraction of pathways and MSM-based prediction of pathways based on the definition of [Nagel et al. 2020](https://pubs.acs.org/doi/10.1021/acs.jctc.0c00774)
- Fast calculation of waiting times based on both, state trajectories and MSMs
- Blazing fast [Chapman-Kolmogorov](https://www.wikiwand.com/en/Chapman%E2%80%93Kolmogorov_equation) test implementation
- Entropy-based similarity measure to compare different state discretizations, this method will be published soon in Nagel 2023
- Contact representation by [Nagel et al. 2023](https://arxiv.org/abs/2303.03814) for a compact structural representation of the states
- Command-line interface providing both, visualization and analysis methods
- Provide (non-reversible) transition matrix of all states (corresponds in pyemma to `connectivity='none', 'all'` which will (probably) [never be implemented](https://github.com/markovmodel/PyEMMA/blob/5315b8699eff2941e84577932921f694dca76f59/pyemma/msm/estimators/_msm_estimator_base.py#L110))

## Getting started
### Installation
The package is called `msmhelper` and is available via [PyPI](https://pypi.org/project/msmhelper) or [conda](https://anaconda.org/conda-forge/msmhelper). To install it, simply call:
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

### Documentation and Tutorials

The [documentation](https://moldyn.github.io/msmhelper) serves as a comprehensive resource, offering a broad range of information such as general guidelines, API code references, and command line tool details. It also includes a Frequently Asked Questions (FAQ) section and outlines the procedures for contributing to the project. 
Moreover, a suite of [tutorials](https://moldyn.github.io/msmhelper/tutorials/) is available, covering all the primary functionalities of the package. These tutorials are provided in the form of Jupyter notebooks. You can easily obtain these notebooks either directly from the [docs/tutorials](https://github.com/moldyn/msmhelper/tree/main/docs/tutorials) directory on our GitHub repository or by clicking the download buttons available on each tutorial page within the documentation.

If you prefer, you can compile the documentation on your local machine by executing the following commands:

```bash
# install all additional dependencies
python -m pip install msmhelper[docs]
# build the docs inside the site directory
python -m mkdocs build
```

### Shell Completion
Using the `bash`, `zsh` or `fish` shell click provides an easy way to
provide shell completion, checkout the
[docs](https://click.palletsprojects.com/en/8.1.x/shell-completion).
In the case of bash you need to add following line to your `~/.bashrc`
```bash
eval "$(_MSMHELPER_COMPLETE=bash_source msmhelper)"
```
In general one can call the module directly by its entry point `$ msmhelper`
or by calling the module `$ python -m msmhelper`. The latter method is
preferred to ensure using the desired python environment. For enabling
the shell completion, the entry point needs to be used.

### Usage
This package offers either a [command line interface](https://moldyn.github.io/msmhelper/reference/cli) to run standalone analysis and to create commonly-used figures, or its much more powerful [API](https://moldyn.github.io/msmhelper/tutorials/msmhelper) can be used to embedded it into an existing Python workflow. Check out the documentation for an overview over all modules and some example workflows, and for some examples see the [following section](#hummer-szabo-projection).
```python
import msmhelper as mh

# open text files
traj = mh.openmicrostates(filename, limitsfile)
# create markov state model
tmat, states = mh.estimate_markov_model(traj, lagtime=1)
...
```

### Hummer-Szabo Projection
In the following we show some sample figures produced directly with the command line tools. For more information on that, there is a [tutorial](https://moldyn.github.io/msmhelper/tutorials/hummerszabo) explaining the methods more in depth. In general we can see, that applying the HS-projection removes most projection artifacts based on coarse-graining many microstates into a few macrostates.

| Method | MSM | Hummer-Szabo MSM |
| :---: | :---: | :---: |
| Implied Timescales | [![Implied Timescales](https://moldyn.github.io/msmhelper/assets/8state_macrotraj.impl.jpg)](reference/cli/#msmhelper-implied-timescales) | [![Implied Timescales](https://moldyn.github.io/msmhelper/assets/8state_macrotraj.sh.impl.jpg)](reference/cli/#msmhelper-implied-timescales) |
| Chapman-Kolmogorov test | [![Chapman-Kolmogorov Test](https://moldyn.github.io/msmhelper/assets/8state_macrotraj.cktest.state1-4.jpg)](reference/cli/#msmhelper-ck-test) | [![Chapman-Kolmogorov Test](https://moldyn.github.io/msmhelper/assets/8state_macrotraj.sh.cktest.state1-4.jpg)](reference/cli/#msmhelper-ck-test) |
| Waiting Time Distributions | [![waiting time distribution](https://moldyn.github.io/msmhelper/assets/8state_macrotraj.wtd.jpg)](reference/cli/#msmhelper-waiting-time-dist) | [![waiting time distribution](https://moldyn.github.io/msmhelper/assets/8state_macrotraj.sh.wtd.jpg)](reference/cli/#msmhelper-waiting-time-dist) |
| Waiting Times | [![waiting times](https://moldyn.github.io/msmhelper/assets/8state_macrotraj.wts.jpg)](reference/cli/#msmhelper-waiting-times) | [![waiting times](https://moldyn.github.io/msmhelper/assets/8state_macrotraj.sh.wts.jpg)](reference/cli/#msmhelper-waiting-times) |
| Contact Representation | [![contact representation](https://moldyn.github.io/msmhelper/assets/hp35.contactRep.state1-12.jpg)](reference/cli/#msmhelper-contact-rep) | |

For more examples checkout the [tutorials](https://moldyn.github.io/msmhelper/tutorials).

## Roadmap
- Add [Buchete-Hummer test](https://doi.org/10.1021/jp0761665) as alternative for the Chapman-Kolmogorov test.
- Add a numba implementation of a parallelized autocorrelation function estimation.
- Use static type hints together with [beartype](https://github.com/beartype/beartype)

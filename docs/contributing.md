# Welcome to the `msmhelper` Contributing Guide

This guide will give you an overview of the contribution workflow from opening an issue and creating a PR. To get an overview of the project, read the [module overview][msmhelper].

## Issues

### Create a new issue

If you spot a bug, want to request a new functionality, or have a question on how to use the module, please [search if an issue already exists](https://github.com/moldyn/msmhelper/issues). If a related issue does not exist, feel free to [open a new issue](https://github.com/moldyn/msmhelper/issues/new/choose).

### Solve an issue

If you want to contribute and do not how, feel free to scan through the [existing issues](https://github.com/moldyn/msmhelper/issues).

## Create a new pull request
### Create a fork

If you want to request a change, you first have to [fork the repository](https://github.com/moldyn/msmhelper/fork).

### Setup a development environment

=== "bash + conda"

    ``` bash
    conda create -n msmhelper -c conda-forge python=3.10
    conda activate msmhelper
    python -m pip install -e .[all]
    ```

=== "bash + venv"

    ``` bash
    python -m venv ./msmhelper
    source ./msmhelper/bin/activate
    python -m pip install -e .[all]
    ```

=== "zsh + conda"

    ``` zsh
    conda create -n msmhelper -c conda-forge python=3.10
    conda activate msmhelper
    python -m pip install -e .\[all]
    ```

=== "zsh + venv"

    ``` zsh
    python -m venv ./msmhelper
    source ./msmhelper/bin/activate
    python -m pip install -e .\[all]
    ```

### Make changes and run tests

Apply your changes and check if you followed the coding style (PEP8) by running
```bash
python -m flake8 --config flake8-CI.cfg
```
All errors pointing to `./build/` can be neglected.

If you add a new function/method/class please ensure that you add a test function, as well. Running the test simply by
```bash
python -m pytest --no-cov
export NUMBA_DISABLE_JIT=1 && python -m pytest
```
Ensure that the coverage does not decrease.

### Open a pull request

Now you are ready to open a pull request and please do not forget to add a description.

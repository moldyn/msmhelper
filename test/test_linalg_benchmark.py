# -*- coding: utf-8 -*-
"""Benchmarking Linalg.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

"""
import numpy as np
import pytest

import msmhelper as mh


@pytest.fixture
def tmat1():
    """Define transition matrix."""
    np.random.seed(137)
    tmat, _ = mh.estimate_markov_model(
        mh.StateTraj(np.random.randint(low=0, high=10, size=int(1e5))),
        lagtime=1,
    )
    return tmat


@pytest.fixture
def tmat2():
    """Define transition matrix."""
    np.random.seed(137)
    tmat, _ = mh.estimate_markov_model(
        mh.StateTraj(np.random.randint(low=0, high=100, size=int(1e5))),
        lagtime=1,
    )
    return tmat


@pytest.fixture
def tmat3():
    """Define transition matrix."""
    np.random.seed(137)
    tmat, _ = mh.estimate_markov_model(
        mh.StateTraj(np.random.randint(low=0, high=1000, size=int(1e5))),
        lagtime=1,
    )
    return tmat


# ~~~ TESTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def test_linalg_eigvals_10states(tmat1, benchmark):
    """Benchmark msmhelper with StateTraj class."""
    benchmark(mh.linalg.eigvals, tmat1)


def test_linalg_eigvals_100states(tmat2, benchmark):
    """Benchmark msmhelper with StateTraj class."""
    benchmark(mh.linalg.eigvals, tmat2)


def test_linalg_eigvals_1000states(tmat3, benchmark):
    """Benchmark msmhelper with StateTraj class."""
    benchmark(mh.linalg.eigvals, tmat3)


def test_linalg_eig_10states(tmat1, benchmark):
    """Benchmark msmhelper with StateTraj class."""
    benchmark(mh.linalg.eig, tmat1)


def test_linalg_eig_100states(tmat2, benchmark):
    """Benchmark msmhelper with StateTraj class."""
    benchmark(mh.linalg.eig, tmat2)


def test_linalg_eig_1000states(tmat3, benchmark):
    """Benchmark msmhelper with StateTraj class."""
    benchmark(mh.linalg.eig, tmat3)

# -*- coding: utf-8 -*-
"""Benchmarking tests.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

"""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pytest

import msmhelper as mh


@pytest.fixture
def traj1():
    """Define traj 1."""
    np.random.seed(137)
    return mh.StateTraj(np.random.randint(low=1, high=11, size=int(1e5)))


@pytest.fixture
def traj2():
    """Define traj 2."""
    np.random.seed(337)
    return mh.StateTraj(np.random.randint(low=1, high=11, size=int(1e5)))


# ~~~ TESTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def test_compare_discretization(traj1, traj2, benchmark):
    """Benchmark msmhelper with pure numpy."""
    # execute for compiling numba
    mh.compare._compare_discretization(traj1, traj2, method='symmetric')
    benchmark(
        mh.compare._compare_discretization,
        traj1,
        traj2,
        method='symmetric',
    )

# -*- coding: utf-8 -*-
"""Benchmarking tests.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

"""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pytest
from pyemma import msm as emsm

import msmhelper as mh


@pytest.fixture
def state_traj():
    """Define state trajectory."""
    np.random.seed(137)
    return mh.format_state_traj(np.random.randint(10, size=int(1e6)))


@pytest.fixture
def lag_time():
    """Define lag time."""
    return 1


# ~~~ TESTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def test_msm_msmhelper(state_traj, lag_time, benchmark):
    """Test row normalization."""
    benchmark(mh.estimate_markov_model, state_traj, lag_time=lag_time)


def test_msm_pyemma(state_traj, lag_time, benchmark):
    """Test row normalization."""
    benchmark(
        emsm.estimate_markov_model,
        state_traj,
        lag_time,
        reversible=False,
    )


def test_msm_pyemma_reversible(state_traj, lag_time, benchmark):
    """Test row normalization."""
    benchmark(
        emsm.estimate_markov_model,
        state_traj,
        lag_time,
        reversible=True,
    )

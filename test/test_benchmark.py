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
def lagtime():
    """Define lag time."""
    return 1


# ~~~ TESTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def test_msm_msmhelper(state_traj, lagtime, benchmark):
    """Test row normalization."""
    assert mh.tests.is_index_traj(state_traj)
    nstates = len(mh.tools.unique(state_traj))
    benchmark(
        mh.msm._estimate_markov_model,
        state_traj,
        lagtime=lagtime,
        nstates=nstates,
    )


def test_msm_pyemma(state_traj, lagtime, benchmark):
    """Test row normalization."""
    benchmark(
        emsm.estimate_markov_model,
        state_traj,
        lagtime,
        reversible=False,
    )


def test_msm_pyemma_reversible(state_traj, lagtime, benchmark):
    """Test row normalization."""
    benchmark(
        emsm.estimate_markov_model,
        state_traj,
        lagtime,
        reversible=True,
    )


def test_is_index_traj(state_traj, benchmark):
    """Test row normalization."""
    benchmark(mh.tests.is_index_traj, state_traj)

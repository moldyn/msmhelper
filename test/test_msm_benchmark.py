# -*- coding: utf-8 -*-
"""Benchmarking tests.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

"""
import numpy as np
import pytest

import msmhelper as mh


@pytest.fixture
def state_traj():
    """Define state trajectory."""
    np.random.seed(137)
    return mh.StateTraj(np.random.randint(low=1, high=11, size=int(1e6)))


@pytest.fixture
def lagtime():
    """Define lag time."""
    return 1


def test_msm_msmhelper_statetraj(state_traj, lagtime, benchmark):
    """Benchmark msmhelper with StateTraj class."""
    benchmark(
        mh.msm.estimate_markov_model,
        state_traj,
        lagtime=lagtime,
    )


def test_msm_msmhelper_list(state_traj, lagtime, benchmark):
    """Benchmark msmhelper without StateTraj class."""
    state_traj = state_traj.state_trajs
    benchmark(
        mh.msm.estimate_markov_model,
        state_traj,
        lagtime=lagtime,
    )


def test_msm_pyemma(state_traj, lagtime, benchmark):
    """Benchmark pyemma without reversibility."""
    from pyemma import msm as emsm

    state_traj = state_traj.state_trajs
    benchmark(
        emsm.estimate_markov_model,
        state_traj,
        lagtime,
        reversible=False,
    )


def test_msm_pyemma_reversible(state_traj, lagtime, benchmark):
    """Benchmark pyemma with reversibility."""
    from pyemma import msm as emsm

    state_traj = state_traj.state_trajs
    benchmark(
        emsm.estimate_markov_model,
        state_traj,
        lagtime,
        reversible=True,
    )


def test_is_index_traj(state_traj, benchmark):
    """Test row normalization."""
    state_traj = state_traj.state_trajs
    benchmark(mh.tests.is_index_traj, state_traj)


def test_msm_wt(benchmark):
    np.random.seed(42)
    N = 50
    trajs = mh.StateTraj(np.random.randint(N, size=100))
    benchmark(
        mh.estimate_msm_waiting_times,
        trajs=trajs,
        lagtime=1,
        start=trajs.states.min(),
        final=trajs.states.max(),
        steps=int(1e8),
    )


def test_mcmc(benchmark):
    N = 50
    trajs = mh.StateTraj(np.random.randint(N, size=100))
    benchmark(
        mh.propagate_MCMC,
        trajs, 1, steps=int(1e7),
    )

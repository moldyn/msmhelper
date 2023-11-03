# -*- coding: utf-8 -*-
"""Tests for the filtering submodule.

BSD 3-Clause License
Copyright (c) 2019-2023, Daniel Nagel
All rights reserved.

"""
import numpy as np
import pytest
from msmhelper import msm
from msmhelper.utils import datasets

NSTEPS = 1000000
DECIMAL = 2


def test_nagel20_4state():
    """Test nagel20_4state mcmc."""
    traj_mcmc = datasets.nagel20_4state(NSTEPS)

    np.testing.assert_array_almost_equal(
        datasets.nagel20_4state.tmat,
        msm.estimate_markov_model(trajs=traj_mcmc, lagtime=1)[0],
        decimal=DECIMAL,
    )


def test_nagel20_6state():
    """Test nagel20_6state mcmc."""
    traj_mcmc = datasets.nagel20_6state(NSTEPS)

    np.testing.assert_array_almost_equal(
        datasets.nagel20_6state.tmat,
        msm.estimate_markov_model(trajs=traj_mcmc, lagtime=1)[0],
        decimal=DECIMAL,
    )


@pytest.mark.parametrize('nstates', [2, 2, 3, 4, 5])
def test_propagate_tmat(nstates):
    tmat = np.random.uniform(size=(nstates, nstates))
    tmat = msm.row_normalize_matrix(tmat)

    traj_mcmc = datasets.propagate_tmat(tmat, NSTEPS)
    np.testing.assert_array_almost_equal(
        tmat,
        msm.estimate_markov_model(trajs=traj_mcmc, lagtime=1)[0],
        decimal=DECIMAL,
    )

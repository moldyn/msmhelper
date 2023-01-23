# -*- coding: utf-8 -*-
"""Tests for the msm module.

BSD 3-Clause License
Copyright (c) 2019-2023, Daniel Nagel
All rights reserved.

"""
import numba
import numpy as np
import pytest

from msmhelper.msm import msm
from msmhelper.statetraj import StateTraj


@pytest.mark.parametrize('mat, matref', [
    ([[1, 1], [3, 1]], [[0.5, 0.5], [0.75, 0.25]])])
def test_row_normalize_matrix(mat, matref):
    """Test row normalization."""
    # cast mat to ndarray
    mat = np.array(mat)
    mat = msm.row_normalize_matrix(mat)
    np.testing.assert_array_equal(mat, matref)

    # set first row to 0
    mat[0] = 0
    np.testing.assert_array_equal(mat[0], msm.row_normalize_matrix(mat)[0])


@pytest.mark.parametrize('traj, lagtime, Tref, nstates', [
    ([1, 1, 1, 1, 1, 2, 2, 1, 2, 0, 2, 2, 0], 1,
     [[0, 0, 1], [0, 4, 2], [2, 1, 2]], 3)])
def test__generate_transition_count_matrix(traj, lagtime, Tref, nstates):
    """Test transition count matrix estimate."""
    # convert traj to numba # noqa: SC100
    if numba.config.DISABLE_JIT:
        traj = [traj]
    else:
        traj = numba.typed.List([np.array(traj)])

    T = msm._generate_transition_count_matrix(traj, lagtime, nstates)
    np.testing.assert_array_equal(T, Tref)


@pytest.mark.parametrize('traj, lagtime, Tref, statesref', [
    ([1, 1, 1, 1, 1, 2, 2, 1, 2, 0, 2, 2, 0], 1,
     [[0., 0., 1.], [0., 2 / 3, 1 / 3], [0.4, 0.2, 0.4]], [0, 1, 2]),
    ([3, 3, 3, 3, 3, 2, 2, 3, 2, 0, 2, 2, 0], 1,
     [[0., 1., 0.], [0.4, 0.4, 0.2], [0., 1 / 3, 2 / 3]], [0, 2, 3]),
])
def test_estimate_markov_model(traj, lagtime, Tref, statesref):
    """Test estimate markov model."""
    T, states = msm.estimate_markov_model(traj, lagtime)
    np.testing.assert_array_equal(T, Tref)
    np.testing.assert_array_equal(states, statesref)


@pytest.mark.parametrize('traj, lagtime, Tref, statesref', [
    ([1, 1, 1, 1, 1, 2, 2, 1, 2, 0, 2, 2, 0], 1,
     [[0., 0., 1.], [0., 2 / 3, 1 / 3], [0.4, 0.2, 0.4]], [0, 1, 2]),
])
def test__estimate_markov_model(traj, lagtime, Tref, statesref):
    """Test estimate markov model."""
    traj = StateTraj(traj)
    T, states = msm._estimate_markov_model(traj.trajs, lagtime, traj.nstates)
    np.testing.assert_array_equal(T, Tref)
    np.testing.assert_array_equal(states, statesref)


@pytest.mark.parametrize('tmat, peqref, kwargs, error', [
    (
        np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # nevver left
            [0.0, 0.8, 0.1, 0.1, 0.0, 0.0],
            [0.0, 0.1, 0.5, 0.4, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # nevver left
            [0.0, 0.2, 0.0, 0.0, 0.8, 0.0],  # never entered
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # never visited
        ]),
        [0, 0.6, 0.4, 0, 0, 0],
        {},
        None,
    ),
    (
        np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # nevver left
            [0.0, 0.8, 0.1, 0.1, 0.0, 0.0],
            [0.0, 0.1, 0.5, 0.4, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # nevver left
            [0.0, 0.2, 0.0, 0.0, 0.8, 0.0],  # never entered
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # never visited
        ]),
        None,
        {'allow_non_ergodic': False},
        ValueError,
    ),
    (
        np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]]),
        np.ones(3) / 3,
        {},
        None,
    ),
])
def test_equilibrium_population(tmat, peqref, kwargs, error):
    """Test peq."""
    if error is None:
        peq = msm.equilibrium_population(tmat, **kwargs)
        np.testing.assert_array_almost_equal(peq, peqref)
    else:
        with pytest.raises(error):
            msm.equilibrium_population(tmat, **kwargs)

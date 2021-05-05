# -*- coding: utf-8 -*-
"""Tests for the msm module.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

Author: Daniel Nagel
        Georg Diez

"""
import numba
import numpy as np
import pytest

import msmhelper
from msmhelper import msm
from msmhelper.statetraj import StateTraj


@pytest.mark.parametrize('mat, matref', [
    ([[1, 1], [3, 1]], [[0.5, 0.5], [0.75, 0.25]])])
def test__row_normalize_matrix(mat, matref):
    """Test row normalization."""
    # cast mat to ndarray
    mat = np.array(mat)
    mat = msm._row_normalize_matrix(mat)
    np.testing.assert_array_equal(mat, matref)

    # set first row to 0
    mat[0] = 0
    np.testing.assert_array_equal(mat[0], msm._row_normalize_matrix(mat)[0])


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
    T, states = msmhelper.estimate_markov_model(traj, lagtime)
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


@pytest.mark.parametrize('transmat, lagtime, result', [
    (
        [[0.8, 0.2, 0.0], [0.2, 0.78, 0.02], [0.0, 0.2, 0.8]],
        2,
        -2 / np.log([4 / 5, 29 / 50]),
    ),
    (
        [[0.1, 0.9], [0.8, 0.2]],
        1,
        [np.nan],
    ),
])
def test__implied_timescales(transmat, lagtime, result):
    """Test implied timescale."""
    impl = msm._implied_timescales(transmat, lagtime)
    np.testing.assert_array_almost_equal(impl, result)


@pytest.mark.parametrize('trajs, lagtimes, result', [
    ([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1], [1, 2],
     [-1 / np.log([2 / 3]), -2 / np.log([7 / 15])])])
def test_implied_timescales(trajs, lagtimes, result):
    """Test estimate markov model."""
    impl = msmhelper.implied_timescales(trajs, lagtimes)
    np.testing.assert_array_almost_equal(impl, result)

    with pytest.raises(TypeError):
        impl = msmhelper.implied_timescales(trajs, [-1, 2])

    with pytest.raises(TypeError):
        impl = msmhelper.implied_timescales(trajs, [1, 2.3])

    with pytest.raises(TypeError):
        impl = msmhelper.implied_timescales(trajs, [1, 2.3], reversible=True)

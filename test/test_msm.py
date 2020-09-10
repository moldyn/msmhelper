# -*- coding: utf-8 -*-
"""
Tests for the msm module.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

Author: Daniel Nagel
        Georg Diez

"""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numba
import numpy as np
import pytest

import msmhelper
from msmhelper import msm
from msmhelper.statetraj import StateTraj


# ~~~ TESTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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


@pytest.mark.parametrize('trajs, lagtime, Tref, statesref', [
    ([1, 1, 1, 1, 1, 2, 2, 1, 2, 0, 2, 2, 0], 1,
     [[0., 0., 1.], [0., 2 / 3, 1 / 3], [0.4, 0.2, 0.4]], [0, 1, 2])])
def test_build_MSM(trajs, lagtime, Tref, statesref):
    """Test estimate markov model."""
    for traj in [trajs, StateTraj(trajs)]:
        # non reversible
        T, states = msmhelper.build_MSM(traj, lagtime, reversible=False)
        np.testing.assert_array_almost_equal(T, Tref)
        np.testing.assert_array_equal(states, statesref)

        #  reversible
        T, states = msmhelper.build_MSM(traj, lagtime, reversible=True)
        np.testing.assert_array_almost_equal(T, Tref)
        np.testing.assert_array_equal(states, statesref)


@pytest.mark.parametrize('matrix, eigenvaluesref, eigenvectorsref', [
    (np.matrix([[1, 6, -1], [2, -1, -2], [1, 0, -1]]), np.array([3, 0, -4]),
     [np.array([-2, -3, 2]) / np.sqrt(17),
      np.array([-1, -6, 13]) / np.sqrt(206),
      np.array([-1, 2, 1]) / np.sqrt(6)])])
def test_left_eigenvectors(matrix, eigenvaluesref, eigenvectorsref):
    """Test left eigenvectors estimate."""
    eigenvalues, eigenvectors = msmhelper.left_eigenvectors(matrix)
    np.testing.assert_array_almost_equal(eigenvalues, eigenvaluesref)
    np.testing.assert_array_almost_equal(
        np.abs(eigenvectors),
        np.abs(eigenvectorsref),
    )

    with pytest.raises(TypeError):
        msmhelper.left_eigenvectors(matrix[0])


@pytest.mark.parametrize('transmat, lagtime, result', [
    ([[0.8, 0.2, 0.0], [0.2, 0.78, 0.02], [0.0, 0.2, 0.8]], 2,
     -2 / np.log([4 / 5, 29 / 50]))])
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


@pytest.mark.parametrize('trajs, lagtimes, tmax', [
    ([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1], [1, 2], 5),
])
def test_chapman_kolmogorov_test(trajs, lagtimes, tmax):
    """Test Chapman Kolmogorov test."""
    # check float as lag times
    with pytest.raises(TypeError):
        _ = msmhelper.ck_test(trajs, lagtimes=[1.2], tmax=tmax)

    # check negative lag times
    with pytest.raises(TypeError):
        _ = msmhelper.ck_test(trajs, lagtimes=[-1], tmax=tmax)

    # check 2d lag times
    with pytest.raises(TypeError):
        _ = msmhelper.ck_test(trajs, lagtimes=[lagtimes], tmax=tmax)

    # check maximal time negative
    with pytest.raises(TypeError):
        _ = msmhelper.ck_test(trajs, lagtimes=lagtimes, tmax=-1)

    # check maximal time negative
    with pytest.raises(TypeError):
        _ = msmhelper.ck_test(trajs, lagtimes=lagtimes, tmax=5.7)

    # check if all keys exists
    cktest = msmhelper.ck_test(trajs, lagtimes=lagtimes, tmax=tmax)
    assert 'md' in cktest
    for lagtime in lagtimes:
        assert lagtime in cktest


@pytest.mark.parametrize('trajs, lagtime, tmax, result', [(
    StateTraj([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1]), 1, 2,
    {
        'ck': np.array([[0.8, 0.68], [0.8, 0.68]]),
        'time': np.array([1, 2]),
        'is_ergodic': True,
    },
)])
def test__chapman_kolmogorov_test(trajs, lagtime, tmax, result):
    """Test Chapman Kolmogorov test."""
    cktest = msm._chapman_kolmogorov_test(trajs, lagtime, tmax)

    np.testing.assert_array_equal(cktest.keys(), result.keys())
    for key in cktest.keys():
        np.testing.assert_array_almost_equal(cktest[key], result[key])

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
import numpy as np
import numba
import pytest

import msmhelper
from msmhelper import msm


# ~~~ TESTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@pytest.mark.parametrize('mat, matref', [
    ([[1, 1], [3, 1]], [[0.5, 0.5], [0.75, 0.25]])])
def test__row_normalize_matrix(mat, matref):
    """Test row normalization."""
    # cast mat to ndarray
    mat = np.array(mat)
    mat = msm._row_normalize_matrix(mat)
    for i, row in enumerate(mat):
        assert (row == matref[i]).all()

    with pytest.raises(ValueError):
        # set first row to 0
        mat = np.array(mat)
        mat[0] = 0
        msm._row_normalize_matrix(mat)


@pytest.mark.parametrize('traj, lag_time, Tref, nstates', [
    ([1, 1, 1, 1, 1, 2, 2, 1, 2, 0, 2, 2, 0], 1,
     [[0, 0, 1], [0, 4, 2], [2, 1, 2]], 3)])
def test__generate_transition_count_matrix(traj, lag_time, Tref, nstates):
    """Test transition count matrix estimate."""
    # convert traj to numba
    if numba.config.DISABLE_JIT:
        traj = [traj]
    else:
        traj = numba.typed.List([np.array(traj)])

    T = msm._generate_transition_count_matrix(traj, lag_time, nstates)
    for i, row in enumerate(T):
        assert (row == Tref[i]).all()


@pytest.mark.parametrize('traj, lag_time, Tref', [
    ([1, 1, 1, 1, 1, 2, 2, 1, 2, 0, 2, 2, 0], 1,
     [[0., 0., 1.], [0., 2 / 3, 1 / 3], [0.4, 0.2, 0.4]])])
def test_estimate_markov_model(traj, lag_time, Tref):
    """Test estimate markov model."""
    T = msmhelper.estimate_markov_model(traj, lag_time)
    for i, row in enumerate(T):
        assert (row == Tref[i]).all()


@pytest.mark.parametrize('traj, lag_time, Tref', [
    ([1, 1, 1, 1, 1, 2, 2, 1, 2, 0, 2, 2, 0], 1,
     [[0., 0., 1.], [0., 2 / 3, 1 / 3], [0.4, 0.2, 0.4]])])
def test_build_MSM(traj, lag_time, Tref):
    """Test estimate markov model."""
    # non reversible
    T = msmhelper.build_MSM(traj, lag_time, reversible=False)
    for i, row in enumerate(T):
        assert (row == Tref[i]).all()
    #  reversible
    T = msmhelper.build_MSM(traj, lag_time, reversible=True)
    for i, row in enumerate(T):
        assert (row - Tref[i] < 1e-5).all()


@pytest.mark.parametrize('matrix, eigenvaluesref, eigenvectorsref', [
    (np.matrix([[1, 6, -1], [2, -1, -2], [1, 0, -1]]), np.array([3, 0, -4]),
     [np.array([-2, -3, 2]) / np.sqrt(17),
      np.array([-1, -6, 13]) / np.sqrt(206),
      np.array([-1, 2, 1]) / np.sqrt(6)])])
def test_left_eigenvectors(matrix, eigenvaluesref, eigenvectorsref):
    """Test left eigenvectors estimate."""
    eigenvalues, eigenvectors = msmhelper.left_eigenvectors(matrix)
    assert (eigenvalues - eigenvaluesref < 1e-9).all()
    for i, row in enumerate(eigenvectors):
        assert (np.abs(row) - np.abs(eigenvectorsref[i]) < 1e-9).all()

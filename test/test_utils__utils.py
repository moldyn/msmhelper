# -*- coding: utf-8 -*-
"""Tests for the _utils module.

BSD 3-Clause License
Copyright (c) 2019-2023, Daniel Nagel
All rights reserved.

"""
import numpy as np
import pytest
import msmhelper
from msmhelper.utils import _utils
from msmhelper.statetraj import StateTraj


@pytest.mark.parametrize('data, expected, val_old, val_new', [
    ([1, 1, 1, 3, 2, 2], [2, 2, 2, 3, 1, 1], [1, 2, 3], [2, 1, 3]),
    ([1, -1, 1, 3, 2, 2], [2, -1, 2, 3, 1, 1], [1, 2], [2, 1]),
    ([1, 1, 1, 3, 2, 2], [3, 3, 3, 2, 1, 1], [1, 2, 3], [3, 1, 2]),
    (np.array([1, 1, 1, 3, 2, 2]), [3, 3, 3, 2, 1, 1], [1, 2, 3], [3, 1, 2]),
    ([np.array([1, 1, -5, 3, 2])], [[3, 3, -5, 2, 1]], [1, 2, 3], [3, 1, 2]),
    ([[1, 1, 1], [3, 2, 2]], [[2, 2, 2], [3, 1, 1]], [1, 2, 3], [2, 1, 3])])
def test_shift_data(data, expected, val_old, val_new):
    """Test shift_data."""
    data_shifted = msmhelper.shift_data(data, val_old, val_new)
    for i, row in enumerate(data_shifted):
        assert (row == expected[i]).all()

    with pytest.raises(TypeError):
        msmhelper.shift_data(data, val_old, val_new, dtype=np.float64)


@pytest.mark.parametrize('data, expected, perm_expected', [
    ([1, 3, 3, 3, 2, 2], [3, 1, 1, 1, 2, 2], [3, 2, 1]),
    ([1, -5, -5, 7, 7, 7], [3, 2, 2, 1, 1, 1], [7, -5, 1])])
def test_rename_by_population(data, expected, perm_expected):
    """Test rename_by_population."""
    assert (msmhelper.rename_by_population(data) == expected).all()
    result, perm = msmhelper.rename_by_population(
        data, return_permutation=True,
    )
    assert (result == expected).all()
    assert (perm == perm_expected).all()


@pytest.mark.parametrize('data, expected, perm_expected', [
    ([1, 3, 3, 3, 2, 2], [0, 2, 2, 2, 1, 1], [1, 2, 3]),
    ([1, -5, -5, 7, 7, 7], [1, 0, 0, 2, 2, 2], [-5, 1, 7])])
def test_rename_by_index(data, expected, perm_expected):
    """Test rename_by_index."""
    assert (msmhelper.rename_by_index(data) == expected).all()
    result, perm = msmhelper.rename_by_index(
        data, return_permutation=True,
    )
    assert (result == expected).all()
    assert (perm == perm_expected).all()


@pytest.mark.parametrize('data, expected, pop', [
    ([1, 3, 3, 3, 2, 2], [1, 2, 3], [1, 2, 3]),
    ([[1, -5, -5, 7, 7, 7], [1, 1]], [-5, 1, 7], [2, 3, 3])])
def test_unique(data, expected, pop):
    """Test unique."""
    assert (msmhelper.unique(data) == expected).all()
    result, perm = msmhelper.unique(data, return_counts=True)
    assert (result == expected).all()
    assert (perm == pop).all()


@pytest.mark.parametrize('data, expected, window', [
    ([1, 1, 1, 2, 2, 2, 3, 3, 3], [1., 1., 1., 2., 2., 2., 3., 3., 3.], 1),
    ([1, 1, 1, 2, 2, 2, 3, 3, 3], [.5, 1., 1., 1.5, 2., 2., 2.5, 3., 3.], 2)])
def test_runningmean(data, expected, window):
    """Test runningmean."""
    assert (_utils.runningmean(data, window) == expected).all()


@pytest.mark.parametrize('traj', [([1, 1, 1, 1, 1, 2, 2, 1, 2, 0, 2, 2, 0])])
def test_format_state_traj(traj):
    """Test formating state trajectory."""
    # as list of floats
    with pytest.raises(TypeError):
        _utils.format_state_traj([float(s) for s in traj])

    # as list of integers
    assert (_utils.format_state_traj(traj)[0] == traj).all()

    # as list of lists
    assert (_utils.format_state_traj([traj])[0] == traj).all()

    # as ndarray
    traj = np.array(traj)
    assert (_utils.format_state_traj(traj)[0] == traj).all()

    # as 2d ndarray
    traj = np.atleast_2d(traj)
    assert (_utils.format_state_traj(traj)[0] == traj).all()

    # as ndarray of floats
    with pytest.raises(TypeError):
        _utils.format_state_traj(traj.astype(np.float64))

    # as list of ndarrays
    assert (_utils.format_state_traj([traj])[0] == traj).all()


@pytest.mark.parametrize('data', [
    ([[1, 1], [1, 2], [2, 1], [2, 3], [3, 3]])])
def test_swapcols(data):
    """Test swapcols."""
    # for same indices
    data_swap = _utils.swapcols(data, (0, 1), (0, 1))
    for row in range(len(data)):
        assert (data[row] == data_swap[row]).all()

    data_swap = _utils.swapcols(data, (0, 1), (1, 0))
    for row in range(len(data)):
        assert (data[row][0] == data_swap[row][1]).all()
        assert (data[row][1] == data_swap[row][0]).all()

    # wrong shape
    with pytest.raises(ValueError):
        _utils.swapcols(data, (0, 1), (1))


def test__asindex():
    """Test asindex."""
    idx = np.arange(5)
    assert (idx == _utils._asindex(idx)).all()

    # check if integer is casted to array
    assert (np.array(5) == _utils._asindex(5)).all()

    # wrong dimensionality
    with pytest.raises(ValueError):
        _utils._asindex([idx])


@pytest.mark.parametrize('traj', [([1, 1, 1, 1, 1, 2, 2, 1, 2, 0, 2, 2, 0])])
def test__check_state_traj(traj):
    """Test if state trajectory is formatted."""
    traj = _utils.format_state_traj(traj)

    assert _utils._check_state_traj(traj)
    assert _utils._check_state_traj(StateTraj(traj))

    with pytest.raises(TypeError):
        _utils._check_state_traj([traj[0].astype(np.float32)])

    with pytest.raises(TypeError):
        _utils._check_state_traj(traj[0])


@pytest.mark.parametrize('mat, power', [
    (np.arange(16, dtype=np.float64).reshape(4, 4), 2),
])
def test_matrix_power(mat, power):
    """Test matrix power."""
    np.testing.assert_array_equal(
        _utils.matrix_power(mat, power),
        np.linalg.matrix_power(mat, power),
    )


@pytest.mark.parametrize('array, val, pos', [
    (np.arange(10), 3, 3), (np.arange(10), 10, -1),
])
def test_find_first(array, val, pos):
    """Test find_first."""
    assert (_utils.find_first(val, array) == pos)

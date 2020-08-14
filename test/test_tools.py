# -*- coding: utf-8 -*-
"""
Tests for the tools module.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

Author: Daniel Nagel

"""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from importlib import reload

import numpy as np
import pytest

import __main__ as main
import msmhelper
from msmhelper import tools


class change_main__file__:
    """Emulate an ipython usage."""

    def __enter__(self):
        """Delete file name and reload module."""
        self.filename = main.__file__
        del main.__file__
        main.msmhelper = reload(msmhelper)

    def __exit__(self, typ, val, traceback):
        """Reset file name and reload module."""
        main.__file__ = self.filename
        main.msmhelper = reload(msmhelper)


# ~~~ TESTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        msmhelper.shift_data(data, val_old, val_new, dtype=np.float)


@pytest.mark.parametrize('data, expected, perm_expected', [
    ([1, 3, 3, 3, 2, 2], [3, 1, 1, 1, 2, 2], [3, 2, 1]),
    ([1, -5, -5, 7, 7, 7], [3, 2, 2, 1, 1, 1], [7, -5, 1])])
def test_rename_by_population(data, expected, perm_expected):
    """Test rename_by_population."""
    assert (msmhelper.rename_by_population(data) == expected).all()
    result, perm = msmhelper.rename_by_population(data,
                                                  return_permutation=True)
    assert (result == expected).all()
    assert (perm == perm_expected).all()


@pytest.mark.parametrize('data, expected, window', [
    ([1, 1, 1, 2, 2, 2, 3, 3, 3], [1., 1., 1., 2., 2., 2., 3., 3., 3.], 1),
    ([1, 1, 1, 2, 2, 2, 3, 3, 3], [.5, 1., 1., 1.5, 2., 2., 2.5, 3., 3.], 2)])
def test_runningmean(data, expected, window):
    """Test runningmean."""
    assert (msmhelper.runningmean(data, window) == expected).all()


@pytest.mark.parametrize('traj', [([1, 1, 1, 1, 1, 2, 2, 1, 2, 0, 2, 2, 0])])
def test_format_state_traj(traj):
    """Test formating state trajectory."""
    # as list of floats
    with pytest.raises(TypeError):
        tools.format_state_traj([float(s) for s in traj])

    # as list of integers
    assert (tools.format_state_traj(traj)[0] == traj).all()

    # as list of lists
    assert (tools.format_state_traj([traj])[0] == traj).all()

    # as ndarray
    traj = np.array(traj)
    assert (tools.format_state_traj(traj)[0] == traj).all()

    # as 2d ndarray
    traj = np.atleast_2d(traj)
    assert (tools.format_state_traj(traj)[0] == traj).all()

    # as ndarray of floats
    with pytest.raises(TypeError):
        tools.format_state_traj(traj.astype(np.float))

    # as list of ndarrays
    assert (tools.format_state_traj([traj])[0] == traj).all()


@pytest.mark.parametrize('data', [
    ([[1, 1], [1, 2], [2, 1], [2, 3], [3, 3]])])
def test_swapcols(data):
    """Test swapcols."""
    # for same indices
    data_swap = tools.swapcols(data, (0, 1), (0, 1))
    for row in range(len(data)):
        assert(data[row] == data_swap[row]).all()

    data_swap = tools.swapcols(data, (0, 1), (1, 0))
    for row in range(len(data)):
        assert(data[row][0] == data_swap[row][1]).all()
        assert(data[row][1] == data_swap[row][0]).all()

    # wrong shape
    with pytest.raises(ValueError):
        tools.swapcols(data, (0, 1), (1))


def test__asindex():
    """Test asindex."""
    idx = np.arange(5)
    assert (idx == tools._asindex(idx)).all()

    # check if integer is casted to array
    assert (np.array(5) == tools._asindex(5)).all()

    # wrong dimensionality
    with pytest.raises(ValueError):
        tools._asindex([idx])


def test__check_quadratic():
    """Test _check_quadratic."""
    mat = np.arange(9).reshape(3, 3)

    # check if no error is raised
    tools._check_quadratic(mat)

    # check for non quadratic matrices
    with pytest.raises(ValueError):
        tools._check_quadratic([2, 1])

    # check for scalar
    with pytest.raises(ValueError):
        tools._check_quadratic(1)

    # check for 3d matrices
    with pytest.raises(ValueError):
        tools._check_quadratic(np.arange(8).reshape(2, 2, 2))


def test_get_runtime_user_information():
    """Check for console usage."""
    assert tools.get_runtime_user_information()['script_name'] != 'console'
    with change_main__file__():
        assert tools.get_runtime_user_information()['script_name'] == 'console'

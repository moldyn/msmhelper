"""
Tests for the tools module.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

Author: Daniel Nagel

"""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pytest

import msmhelper
from msmhelper import tools


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
def test__format_state_trajectory(traj):
    """Test formating state trajectory."""
    # as list of floats
    with pytest.raises(TypeError):
        tools._format_state_trajectory([float(s) for s in traj])

    # as list of integers
    assert (tools._format_state_trajectory(traj)[0] == traj).all()

    # as list of lists
    assert (tools._format_state_trajectory([traj])[0] == traj).all()

    # as ndarray
    traj = np.array(traj)
    assert (tools._format_state_trajectory(traj)[0] == traj).all()

    # as ndarry of floats
    with pytest.raises(TypeError):
        tools._format_state_trajectory(traj.astype(np.float))

    # as list of ndarrays
    assert (tools._format_state_trajectory([traj])[0] == traj).all()

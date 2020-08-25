# -*- coding: utf-8 -*-
"""Tests for the tests module.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

"""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pytest

from msmhelper import tests, tools
from msmhelper.statetraj import StateTraj


# ~~~ TESTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@pytest.mark.parametrize('mat', [(np.arange(9).reshape(3, 3))])
def test_is_quadratic(mat):
    """Test is_quadratic."""
    # check if no error is raised
    assert tests.is_quadratic(mat)
    # check for non quadratic matrices
    assert not tests.is_quadratic([2, 1])
    # check for scalar
    assert not tests.is_quadratic(1)
    # check for 3d matrices
    assert not tests.is_quadratic(np.arange(8).reshape(2, 2, 2))


@pytest.mark.parametrize('traj', [([1, 1, 1, 1, 1, 2, 2, 1, 2, 0, 2, 2, 0])])
def test_is_state_traj(traj):
    """Test formating state trajectory."""
    traj = tools.format_state_traj(traj)

    assert tests.is_state_traj(traj)
    assert not tests.is_state_traj([traj[0].astype(np.float32)])


@pytest.mark.parametrize('traj', [([1, 1, 1, 1, 1, 2, 2, 1, 2, 0, 2, 2, 0])])
def test_is_index_traj(traj):
    """Test formating state trajectory."""
    traj = tools.format_state_traj(traj)

    assert tests.is_index_traj(traj)
    assert tests.is_index_traj(StateTraj(traj))
    traj.append(np.array([4, 5]))
    assert not tests.is_index_traj(traj)
    assert not tests.is_index_traj([5])

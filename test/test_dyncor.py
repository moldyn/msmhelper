# -*- coding: utf-8 -*-
"""Tests for the dynamical correction module.

BSD 3-Clause License
Copyright (c) 2019-2022, Daniel Nagel
All rights reserved.

Author: Daniel Nagel

"""
import numba
import numpy as np
import pytest

from msmhelper import dyncor


@pytest.mark.parametrize('idx, traj, lagtime, iterative, result', [
    (0, [1, 1, 2, 2, 2, 2, 2, 1, 2, 0, 2, 2, 0], 2, False, True),
    (0, [1, 1, 2, 2, 2, 2, 2, 1, 2, 0, 2, 2, 0], 3, False, False),
    (0, [1, 1, 2, 2, 2, 2, 2, 1, 2, 0, 2, 2, 0], 8, False, False),
    (0, [1, 1, 2, 2, 2, 2, 2, 1, 2, 0, 2, 2, 0], 8, True, True),
    (2, [1, 1, 2, 2, 2, 2, 2, 1, 2, 0, 2, 2, 0], 3, False, True),
    (0, [1, 1, 2, 2, 2, 2, 2, 1, 2, 0, 2, 2, 0], 20, False, False),
])
def test__remains_in_core(idx, traj, lagtime, iterative, result):
    """Test finding first core."""
    # convert traj to numba # noqa: SC100
    if numba.config.DISABLE_JIT:
        traj = traj
    else:
        traj = numba.typed.List(traj)

    assert dyncor._remains_in_core(
        idx=idx,
        traj=traj,
        lagtime=lagtime,
        iterative=iterative
    ) == result


@pytest.mark.parametrize('traj, lagtime, result', [
    ([1, 1, 2, 2, 2, 2, 2, 1, 2, 0, 2, 2, 0], 2, 1),
    ([1, 1, 2, 2, 2, 2, 2, 1, 2, 0, 2, 2, 0], 3, 2),
    ([1, 1, 2, 2, 2, 2, 2, 1, 2, 0, 2, 2, 0], 6, -1),
])
def test__find_first_core(traj, lagtime, result):
    """Test finding first core."""
    # convert traj to numba # noqa: SC100
    if numba.config.DISABLE_JIT:
        traj = traj
    else:
        traj = numba.typed.List(traj)

    assert dyncor._find_first_core(traj, lagtime) == result


@pytest.mark.parametrize('traj, lagtime, iterative, result, error', [
    (
        [1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 3, 3],
        2,
        True,
        [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3],
        None,
    ),
    (
        [1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 3, 3],
        3,
        True,
        [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        None,
    ),
    (
        [1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 3, 3],
        3,
        False,
        [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
        None,
    ),
    (
        [1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 3, 3],
        4,
        False,
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        None,
    ),
    (
        [1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 3, 3],
        6,
        True,
        None,
        dyncor.LagtimeError,
    ),
])
def test__dynamical_coring_single_traj(
        traj, lagtime, iterative, result, error,
):
    """Test dynamical coring of single trajectory."""
    # convert traj to numba # noqa: SC100
    if numba.config.DISABLE_JIT:
        traj = traj
    else:
        traj = numba.typed.List(traj)

    if error is None:
        cored_traj = dyncor._dynamical_coring_single_traj(
            traj=traj,
            lagtime=lagtime,
            iterative=iterative,
        )
        np.testing.assert_array_almost_equal(cored_traj, result)
    else:
        with pytest.raises(error):
            dyncor._dynamical_coring_single_traj(
                traj=traj,
                lagtime=lagtime,
                iterative=iterative,
            )

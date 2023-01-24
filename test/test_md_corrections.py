# -*- coding: utf-8 -*-
"""Tests for the dynamical correction module.

BSD 3-Clause License
Copyright (c) 2019-2023, Daniel Nagel
All rights reserved.

"""
import numba
import numpy as np
import pytest

from msmhelper.md import corrections
from msmhelper.statetraj import LumpedStateTraj, StateTraj


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
    if not numba.config.DISABLE_JIT:
        traj = numba.typed.List(traj)

    assert corrections._remains_in_core(
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
    if not numba.config.DISABLE_JIT:
        traj = numba.typed.List(traj)

    assert corrections._find_first_core(traj, lagtime) == result


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
        corrections.LagtimeError,
    ),
])
def test__dynamical_coring_single_traj(
        traj, lagtime, iterative, result, error,
):
    """Test dynamical coring of single trajectory."""
    # convert traj to numba # noqa: SC100
    if not numba.config.DISABLE_JIT:
        traj = numba.typed.List(traj)

    if error is None:
        cored_traj = corrections._dynamical_coring_single_traj(
            traj=traj,
            lagtime=lagtime,
            iterative=iterative,
        )
        np.testing.assert_array_almost_equal(cored_traj, result)
    else:
        with pytest.raises(error):
            corrections._dynamical_coring_single_traj(
                traj=traj,
                lagtime=lagtime,
                iterative=iterative,
            )


@pytest.mark.parametrize('trajs, lagtime, kwargs, result, error', [
    (
        StateTraj([1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 2]),
        2,
        {},
        StateTraj([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]),
        None,
    ),
    (
        StateTraj([1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 2]),
        3,
        {},
        StateTraj([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]),
        None,
    ),
    (
        StateTraj([1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 2]),
        3,
        {'iterative': True},
        StateTraj([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]),
        None,
    ),
    (
        StateTraj([1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 2]),
        3,
        {'iterative': False},
        StateTraj([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]),
        None,
    ),
    (
        StateTraj([1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 2]),
        1,
        {},
        StateTraj([1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 2]),
        None,
    ),
    (
        StateTraj([1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 2]),
        0,
        {},
        None,
        ValueError,
    ),
    (
        LumpedStateTraj(
            [1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 2],
            [1, 1, 3, 2, 3, 4, 2, 3, 2, 2, 4],
        ),
        2,
        {},
        None,
        NotImplementedError,
    ),
])
def test_dynamical_coring(trajs, lagtime, kwargs, result, error):
    """Test dynamical coring."""
    if error is None:
        cored_trajs = corrections.dynamical_coring(
            trajs=trajs, lagtime=lagtime, **kwargs,
        )
        assert len(cored_trajs) == len(result)
        for cored_traj, traj in zip(cored_trajs, result):
            np.testing.assert_array_almost_equal(cored_traj, traj)
    else:
        with pytest.raises(error):
            corrections.dynamical_coring(
                trajs=trajs, lagtime=lagtime, **kwargs,
            )

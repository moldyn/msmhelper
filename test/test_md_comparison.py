# -*- coding: utf-8 -*-
"""Tests for the compare module.

BSD 3-Clause License
Copyright (c) 2019-2023, Daniel Nagel
All rights reserved.

"""
import numba
import numpy as np
import pytest

from msmhelper.md import comparison
from msmhelper.statetraj import StateTraj


@pytest.mark.parametrize('arr1, arr2, result', [
    ([0, 1, 2, 3], [4, 5, 6, 7], 0),
    ([0, 1, 2, 3], [1, 2, 3, 4], 3),
    ([0, 1, 2, 3], [0, 1, 2, 3], 4),
])
def test__intersect(arr1, arr2, result):
    """Test intersect method."""
    if not numba.config.DISABLE_JIT:
        arr1 = numba.typed.List(arr1)
        arr2 = numba.typed.List(arr2)

    np.testing.assert_almost_equal(
        comparison._intersect(arr1, arr2),
        result,
    )


@pytest.mark.parametrize('arr1, arr2, result', [(
    [np.array([0, 1, 2, 3, 4]), np.array([5, 6, 7])],
    [np.array([0, 1]), np.array([2, 3]), np.array([4, 5, 6, 7])],
    np.array([[2., 2., 1.], [0., 0., 3.]]),
)])
def test__intersect_array(arr1, arr2, result):
    """Test intersect method."""
    if not numba.config.DISABLE_JIT:
        arr1 = numba.typed.List(arr1)
        arr2 = numba.typed.List(arr2)

    np.testing.assert_array_almost_equal(
        comparison._intersect_array(arr1, arr2), result,
    )


@pytest.mark.parametrize('traj1, traj2, kwargs, result', [
    (
        [0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 2, 2, 2],
        {'method': 'directed'}, 1.0,
    ),
    (
        [0, 0, 0, 0, 0, 1, 1, 1], [1, 1, 2, 2, 2, 3, 3, 3],
        {'method': 'directed'}, 1.0,
    ),
    (
        [0, 0, 1, 1, 1, 2, 2, 2], [0, 0, 0, 0, 0, 1, 1, 1],
        {'method': 'directed'}, 0.7,
    ),
    (
        [0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2, 2, 2],
        {'method': 'symmetric'}, 0.90625,
    ),
    (
        [0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2, 2, 2],
        {'method': 'directed'}, 0.8125,
    ),
    (
        [0, 0, 1, 1, 2, 2, 2, 2], [0, 0, 0, 0, 0, 1, 1, 1],
        {'method': 'directed'}, 0.6,
    ),
])
def test__compare_discretization(traj1, traj2, kwargs, result):
    """Test compare diescretitzation."""
    traj1, traj2 = StateTraj(traj1), StateTraj(traj2)
    np.testing.assert_almost_equal(
        comparison._compare_discretization(traj1, traj2, **kwargs),
        result,
    )

    with pytest.raises(ValueError):
        comparison._compare_discretization(traj1, traj2, method='NotAMethod')


@pytest.mark.parametrize('traj1, traj2, kwargs, error', [
    ([0, 0, 0, 0, 1], [0, 1, 1, 1, 1], {}, None),
    ([0, 0, 0, 0, 1], [0, 1, 1, 1, 1, 1], {}, ValueError),
    ([0, 0, 0, 0, 1], [0, 1, 1, 1], {}, ValueError),
    ([0, 0, 0, 0], [0, 1, 1, 1], {}, ValueError),
    ([0, 0, 0, 0, 1], [1, 1, 1], {}, ValueError),
    ([0, 0, 0, 0], [1, 1, 1], {}, ValueError),
    ([0, 0, 0, 0, 1], [0, 1, 1, 1, 1], {'method': 'symmetric'}, None),
    ([0, 0, 0, 0, 1], [0, 1, 1, 1, 1], {'method': 'directed'}, None),
    ([0, 0, 0, 0, 1], [0, 1, 1, 1, 1], {'method': 'forward'}, ValueError),
])
def test_compare_discretization(traj1, traj2, kwargs, error):
    """Test compare diescretitzation."""
    if error is None:
        comparison.compare_discretization(traj1, traj2, **kwargs)
    else:
        with pytest.raises(error):
            comparison.compare_discretization(traj1, traj2, **kwargs)

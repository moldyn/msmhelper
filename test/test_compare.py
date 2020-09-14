# -*- coding: utf-8 -*-
"""Tests for the compare module.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

"""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pytest

from msmhelper import compare


# ~~~ TESTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@pytest.mark.parametrize('traj1, traj2, kwargs, error', [
    ([0, 0, 0, 0], [1, 1, 1, 1], {}, None),
    ([0, 0, 0, 0], [1, 1, 1, 1, 1], {}, ValueError),
    ([0, 0, 0, 0], [1, 1, 1], {}, ValueError),
    ([0, 0, 0, 0], [1, 1, 1, 1], {'method': 'symmetric'}, None),
    ([0, 0, 0, 0], [1, 1, 1, 1], {'method': 'directed'}, None),
    ([0, 0, 0, 0], [1, 1, 1, 1], {'method': 'forward'}, ValueError),
])
def test_compare_discretization(traj1, traj2, kwargs, error):
    """Test compare diescretitzation."""
    if error is None:
        compare.compare_discretization(traj1, traj2, **kwargs)
    else:
        with pytest.raises(error):
            compare.compare_discretization(traj1, traj2, **kwargs)


@pytest.mark.parametrize('traj1, traj2, kwargs, result', [
    ([0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 2, 2, 2], {}, 1),
    (
        [0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 2, 2, 2],
        {'method': 'directed'}, 1.0,
    ),
    (
        [0, 0, 1, 1, 1, 2, 2, 2], [0, 0, 0, 0, 0, 1, 1, 1],
        {'method': 'directed'}, 0.7,
    ),
    ([0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2, 2, 2], {}, 0.90625),
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
    assert compare.compare_discretization(traj1, traj2, **kwargs) == result


@pytest.mark.parametrize('arr1, arr2, result', [
    ([0, 1, 2, 3], [4, 5, 6, 7], 0),
    ([0, 1, 2, 3], [4, 2, 3, 1], 0.75),
    ([0, 1, 2, 3], [0, 1, 2, 3], 1),
])
def test__intersect(arr1, arr2, result):
    """Test intersect method."""
    assert compare._intersect(arr1, arr2) == result


@pytest.mark.parametrize('arr1, arr2, result', [(
    {0: np.array([0, 1, 2, 3, 4]), 1: np.array([5, 6, 7])},
    {0: np.array([0, 1]), 1: np.array([2, 3]), 2: np.array([4, 5, 6, 7])},
    np.array([[0.4, 0.4, 0.2], [0., 0., 1.]]),
)])
def test__intersect_array(arr1, arr2, result):
    """Test intersect method."""
    np.testing.assert_array_almost_equal(
        compare._intersect_array(arr1, arr2), result,
    )

# -*- coding: utf-8 -*-
"""Tests for the filtering submodule.

BSD 3-Clause License
Copyright (c) 2019-2023, Daniel Nagel
All rights reserved.

"""
import numpy as np
import pytest
from msmhelper.utils import filtering


@pytest.mark.parametrize('data, expected, window, error', [
    (
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1., 1., 1., 2., 2., 2., 3., 3., 3.],
        1,
        None,
    ),
    (
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [.5, 1., 1., 1.5, 2., 2., 2.5, 3., 3.],
        2,
        None
    ),
    (
        np.random.normal(size=(20, 2)),
        None,
        1,
        ValueError,
    )
])
def test_runningmean(data, expected, window, error):
    """Test runningmean."""
    if error is None:
        np.testing.assert_array_almost_equal(
            filtering.runningmean(data, window),
            expected
        )
    else:
        with pytest.raises(error):
            filtering.runningmean(data, window)


@pytest.mark.parametrize('data, expected, sigma, error', [
    (
        np.arange(10),
        np.arange(10),
        0.1,
        None,
    ),
    (
        np.vstack([np.arange(10), np.zeros(10)]).T,
        np.vstack([
            filtering.gaussian_filter(np.arange(10), 2), np.zeros(10),
        ]).T,
        2,
        None,
    ),
    (
        np.arange(10).reshape(-1, 2),
        [
            [1.53241702, 2.53241702],
            [2.65493269, 3.65493269],
            [4.00000000, 5.00000000],
            [5.34506731, 6.34506731],
            [6.46758298, 7.46758298],
        ],
        2,
        None,
    ),
    (
        np.random.normal(size=(20, 5, 3)),
        None,
        1,
        ValueError,
    )
])
def test_gaussian_filter(data, expected, sigma, error):
    """Test runningmean."""
    if error is None:
        np.testing.assert_array_almost_equal(
            filtering.gaussian_filter(data, sigma),
            expected,
        )
    else:
        with pytest.raises(error):
            filtering.gaussian_filter(data, sigma)

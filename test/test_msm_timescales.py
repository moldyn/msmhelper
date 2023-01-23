# -*- coding: utf-8 -*-
"""Tests for the msm module.

BSD 3-Clause License
Copyright (c) 2019-2023, Daniel Nagel
All rights reserved.

"""
import numba
import numpy as np
import pytest

from msmhelper.msm import timescales
from msmhelper.statetraj import StateTraj


@pytest.mark.parametrize('transmat, lagtime, result', [
    (
        [[0.8, 0.2, 0.0], [0.2, 0.78, 0.02], [0.0, 0.2, 0.8]],
        2,
        -2 / np.log([4 / 5, 29 / 50]),
    ),
    (
        [[0.1, 0.9], [0.8, 0.2]],
        1,
        [np.nan],
    ),
])
def test__implied_timescales(transmat, lagtime, result):
    """Test implied timescale."""
    ntimescales = len(result)
    impl = timescales._implied_timescales(
        transmat, lagtime, ntimescales=ntimescales,
    )
    np.testing.assert_array_almost_equal(impl, result)


@pytest.mark.parametrize('trajs, lagtimes, kwargs, result, error', [
    (
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 2],
        {},
        [-1 / np.log([2 / 3]), -2 / np.log([7 / 15])],
        None,
    ),
    (
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 2],
        {'ntimescales': 1},
        [-1 / np.log([2 / 3]), -2 / np.log([7 / 15])],
        None,
    ),
    (
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
        [-1, 2],
        {},
        None,
        TypeError,
    ),
    (
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 2.3],
        {},
        None,
        TypeError,
    ),
    (
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 2],
        {'reversible': True},
        None,
        NotImplementedError,
    ),
])
def test_implied_timescales(trajs, lagtimes, kwargs, result, error):
    """Test implied timescale."""
    if error is None:
        impl = timescales.implied_timescales(trajs, lagtimes, **kwargs)
        np.testing.assert_array_almost_equal(impl, result)
    else:
        with pytest.raises(error):
            timescales.implied_timescales(trajs, lagtimes, **kwargs)

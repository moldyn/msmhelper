# -*- coding: utf-8 -*-
"""Tests for the msm module.

BSD 3-Clause License
Copyright (c) 2019-2023, Daniel Nagel
All rights reserved.

"""
import random

import numpy as np
import pytest

from msmhelper import LumpedStateTraj
from msmhelper.msm import timescales


@pytest.fixture
def rand():
    random.seed(43)
    np.random.seed(43)


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


@pytest.mark.parametrize('traj, lagtime, result, error', [
    (
        [1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2],
        1,
        (
            [[0.8, 1.0], [0.6, 1.0]],
            [[1, 0], [0, 1]],
        ),
        None,
    ),
    (
        LumpedStateTraj(
            [3, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 2, 3],
            [4, 1, 1, 1, 3, 2, 3, 2, 1, 1, 1, 3, 2, 2, 4],
        ),
        1,
        None,
        ValueError,
    ),
])
def test__get_cummat(traj, lagtime, result, error):
    """Test cumulative matrix."""
    if error is None:
        cummat, perm = timescales._get_cummat(traj, lagtime)
        np.testing.assert_array_almost_equal(cummat, result[0])
        np.testing.assert_array_almost_equal(perm, result[1])
    else:
        with pytest.raises(error):
            cummat, perm = timescales._get_cummat(traj, lagtime)


@pytest.mark.parametrize('cummat, idx_from, result', [
    (
        (
            np.array([[0.8, 1.0], [0.6, 1.0]]),
            np.array([[1, 0], [0, 1]]),
        ),
        0,
        1,
    ),
    (
        (
            np.array([[0.8, 1.0], [0.6, 1.0]]),
            np.array([[1, 0], [0, 1]]),
        ),
        1,
        0,
    ),
])
def test__propagate_MCMC_step(cummat, idx_from, result, rand):
    """Test MCMC propagation step."""
    idx_next = timescales._propagate_MCMC_step(cummat, idx_from)
    assert idx_next == result


@pytest.mark.parametrize('cummat, start, steps, result', [
    (
        (
            np.array([[0.8, 1.0], [0.6, 1.0]]),
            np.array([[1, 0], [0, 1]]),
        ),
        0,
        4,
        [0, 1, 1, 0],
    ),
])
def test__propagate_MCMC(cummat, start, steps, result, rand):
    """Test MCMC propagation step."""
    mcmc = timescales._propagate_MCMC(cummat, start, steps)
    np.testing.assert_array_almost_equal(mcmc, result)


@pytest.mark.parametrize('trajs, lagtime, steps, start, error', [
    (
        [1, 1, 1, 2, 2, 1, 1],
        1,
        10,
        1,
        None,
    ),
    (
        [1, 1, 1, 2, 2, 1, 1],
        1,
        10,
        -1,
        None,
    ),
    (
        [1, 1, 1, 2, 2, 1, 1],
        1,
        10,
        3,
        ValueError,
    ),
])
def test_propagate_MCMC(trajs, lagtime, steps, start, error, rand):
    """Test MCMC propagation."""
    if error is None:
        mcmc = timescales.propagate_MCMC(trajs, lagtime, steps, start)

        # due to different random numbers on local machine and github
        # workflows compare only basic stats
        np.testing.assert_array_almost_equal(
            np.unique(mcmc), np.unique(trajs),
        )
    else:
        with pytest.raises(error):
            mcmc = timescales.propagate_MCMC(trajs, lagtime, steps, start)

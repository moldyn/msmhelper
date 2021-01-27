# -*- coding: utf-8 -*-
"""Tests for the benchmark module.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

"""
import numpy as np
import pytest

from msmhelper import benchmark
from msmhelper.statetraj import StateTraj


@pytest.mark.parametrize('trajs, lagtimes, tmax', [
    ([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1], [1, 2], 5),
])
def test_chapman_kolmogorov_test(trajs, lagtimes, tmax):
    """Test Chapman Kolmogorov test."""
    # check float as lag times
    with pytest.raises(TypeError):
        _ = benchmark.ck_test(trajs, lagtimes=[1.2], tmax=tmax)

    # check negative lag times
    with pytest.raises(TypeError):
        _ = benchmark.ck_test(trajs, lagtimes=[-1], tmax=tmax)

    # check 2d lag times
    with pytest.raises(TypeError):
        _ = benchmark.ck_test(trajs, lagtimes=[lagtimes], tmax=tmax)

    # check maximal time negative
    with pytest.raises(TypeError):
        _ = benchmark.ck_test(trajs, lagtimes=lagtimes, tmax=-1)

    # check maximal time float
    with pytest.raises(TypeError):
        _ = benchmark.ck_test(trajs, lagtimes=lagtimes, tmax=5.7)

    # check if all keys exists
    cktest = benchmark.ck_test(trajs, lagtimes=lagtimes, tmax=tmax)
    assert 'md' in cktest
    for lagtime in lagtimes:
        assert lagtime in cktest


@pytest.mark.parametrize('trajs, lagtime, tmax, result', [(
    StateTraj([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1]), 1, 2,
    {
        'ck': np.array([[0.8, 0.68], [0.8, 0.68]]),
        'time': np.array([1, 2]),
        'is_ergodic': True,
        'is_fuzzy_ergodic': True,
    },
)])
def test__chapman_kolmogorov_test(trajs, lagtime, tmax, result):
    """Test Chapman Kolmogorov test."""
    cktest = benchmark._chapman_kolmogorov_test(trajs, lagtime, tmax)

    np.testing.assert_array_equal(cktest.keys(), result.keys())
    for key in cktest.keys():
        np.testing.assert_array_almost_equal(cktest[key], result[key])


@pytest.mark.parametrize('trajs, lagtime, tmax', [
    ([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1], 2, 5),
])
def test_buchete_hummer_test(trajs, lagtime, tmax):
    """Test Buchete Hummer test."""
    # check float as lag times
    with pytest.raises(TypeError):
        _ = benchmark.bh_test(trajs, lagtime=[1.2], tmax=tmax)

    # check negative lag times
    with pytest.raises(TypeError):
        _ = benchmark.bh_test(trajs, lagtime=-1, tmax=tmax)

    # check maximal time negative
    with pytest.raises(TypeError):
        _ = benchmark.bh_test(trajs, lagtime=lagtime, tmax=-1)

    # check maximal time float
    with pytest.raises(TypeError):
        _ = benchmark.bh_test(trajs, lagtime=lagtime, tmax=5.7)

    # check if all keys exists
    bhtest = benchmark.bh_test(trajs, lagtime=lagtime, tmax=tmax)
    assert 'md' in bhtest
    assert lagtime in bhtest

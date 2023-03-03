# -*- coding: utf-8 -*-
"""Tests for the tests submodule of msm.

BSD 3-Clause License
Copyright (c) 2019-2023, Daniel Nagel
All rights reserved.

"""
import numpy as np
import pytest

from msmhelper.msm import tests
from msmhelper.statetraj import StateTraj


@pytest.mark.parametrize('trajs, lagtimes, tmax', [
    ([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1], [1, 2], 5),
])
def test_chapman_kolmogorov_test(trajs, lagtimes, tmax):
    """Test Chapman Kolmogorov test."""
    # check float as lag times
    with pytest.raises(TypeError):
        _ = tests.ck_test(trajs, lagtimes=[1.2], tmax=tmax)

    # check negative lag times
    with pytest.raises(TypeError):
        _ = tests.ck_test(trajs, lagtimes=[-1], tmax=tmax)

    # check 2d lag times
    with pytest.raises(TypeError):
        _ = tests.ck_test(trajs, lagtimes=[lagtimes], tmax=tmax)

    # check maximal time negative
    with pytest.raises(TypeError):
        _ = tests.ck_test(trajs, lagtimes=lagtimes, tmax=-1)

    # check maximal time float
    with pytest.raises(TypeError):
        _ = tests.ck_test(trajs, lagtimes=lagtimes, tmax=5.7)

    # check if all keys exists
    cktest = tests.ck_test(trajs, lagtimes=lagtimes, tmax=tmax)
    assert 'md' in cktest
    for lagtime in lagtimes:
        assert lagtime in cktest


@pytest.mark.parametrize('trajs, lagtime, tmax, result', [(
    StateTraj([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1]), 1, 2,
    {
        'ck': {0: np.array([0.8, 0.68]), 1: np.array([0.8, 0.68])},
        'time': np.array([1, 2]),
        'is_ergodic': True,
        'is_fuzzy_ergodic': True,
    },
)])
def test__chapman_kolmogorov_test(trajs, lagtime, tmax, result):
    """Test Chapman Kolmogorov test."""
    cktest = tests._chapman_kolmogorov_test(trajs, lagtime, tmax)

    np.testing.assert_array_equal(cktest.keys(), result.keys())
    for key in cktest.keys():
        if key == 'ck':
            for state in trajs.states:
                np.testing.assert_array_almost_equal(
                    cktest[key][state], result[key][state]
                )
        else:
            np.testing.assert_array_almost_equal(cktest[key], result[key])

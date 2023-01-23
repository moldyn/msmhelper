# -*- coding: utf-8 -*-
"""Tests for the md module.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

"""
import numpy as np
import pytest

import msmhelper


@pytest.mark.parametrize('traj, start, final, reftimes', [
    ([1, 2, 1, 2, 1, 1, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4], 1, 3, [6, 4]),
    ([1, 2, 1, 2, 1, 1, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4], [1, 2], [3, 4], [6, 4]),
])
def test_estimate_waiting_times(traj, start, final, reftimes):
    """Test estimate waiting time."""
    times = msmhelper.md.estimate_waiting_times(traj, start, final)
    np.testing.assert_array_equal(times, reftimes)

    # check overlapping start and final states
    with pytest.raises(ValueError):
        msmhelper.md.estimate_waiting_times(traj, start, start)

    # check state not contained in traj
    with pytest.raises(ValueError):
        msmhelper.md.estimate_waiting_times(traj, start, np.max(traj) + 1)


@pytest.mark.parametrize('traj, start, final, refpaths', [
    (
        [1, 2, 1, 2, 1, 1, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4],
        1,
        3,
        {(1, 3): [6], (1, 4, 3): [4]},
    ),
    (
        [1, 2, 1, 2, 1, 1, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4],
        [1, 2],
        [3, 4],
        {(1, 3): [6], (1, 4): [4]},
    ),
    (
        [1, 2, 3, 4, 5, 2, 6],
        1,
        6,
        {(1, 2, 6): [6]},
    ),
])
def test_estimate_paths(traj, start, final, refpaths):
    """Test estimate waiting time."""
    paths = msmhelper.md.estimate_paths(traj, start, final)
    assert paths == refpaths

    # check overlapping start and final states
    with pytest.raises(ValueError):
        msmhelper.md.estimate_paths(traj, start, start)

    # check state not contained in traj
    with pytest.raises(ValueError):
        msmhelper.md.estimate_paths(traj, start, np.max(traj) + 1)

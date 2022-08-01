# -*- coding: utf-8 -*-
"""Tests for the iotext module.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

"""
import os.path

import numpy as np
import pytest

from msmhelper import iotext

# Current directory
HERE = os.path.dirname(__file__)


@pytest.fixture
def limits_file():
    """Define limits file."""
    return os.path.join(HERE, 'limits.dat')


@pytest.mark.parametrize('traj_file, kwargs', [
    (os.path.join(HERE, 'data.dat'), {'comment': '#'}),
    (os.path.join(HERE, 'data.dat'), {'usecols': (1, 0)}),
    (os.path.join(HERE, 'traj1.dat'), {}),
    (os.path.join(HERE, 'traj1.dat'), {'comment': ['#']}),
    (os.path.join(HERE, 'traj2.dat'), {'comment': ['#', '@']}),
])
def test_opentxt(traj_file, kwargs):
    """Test that a file is opened correctly."""
    expected = np.array([[1, 2], [1, 3], [1, 2], [1, 1], [1, 2], [2, 1],
                         [2, 5], [1, 4], [2, 3], [3, 2], [2, 1], [2, 2],
                         [3, 3]])
    data = iotext.opentxt(traj_file, **kwargs)
    if len(data.shape) == 1:
        assert (data == expected[:, 0]).all()
    else:
        if 'usecols' in kwargs:
            expected[:, (0, 1)] = expected[:, kwargs['usecols']]
        for i, row in enumerate(data):
            assert (row == expected[i]).all()


@pytest.mark.parametrize('traj_file, header', [
    (os.path.join(HERE, 'traj1.dat'), None),
    (os.path.join(HERE, 'traj1.dat'), 'Test header comment.'),
])
def test_savetxt(traj_file, header, tmpdir):
    """Test that a file is opened correctly."""
    expected = [1, 1, 1, 1, 1, 2, 2, 1, 2, 3, 2, 2, 3]
    output = tmpdir.join('test_traj.txt')
    iotext.savetxt(output, expected, header=header, fmt='%.0f')
    assert (iotext.opentxt(output) == expected).all()


@pytest.mark.parametrize('wrong_limit', [os.path.join(HERE, 'data.dat')])
def test_open_limits(limits_file, wrong_limit):
    """Test that the limits are loaded correctly."""
    assert (iotext.open_limits(13, limits_file) == [5, 10, 13]).all()
    assert (iotext.open_limits(13) == [13]).all()
    with pytest.raises(iotext.FileError):
        iotext.open_limits(13, wrong_limit)
    with pytest.raises(ValueError):
        iotext.open_limits(14, limits_file)


@pytest.mark.parametrize('traj_file, kwargs', [
    (os.path.join(HERE, 'traj1.dat'), {}),
    (os.path.join(HERE, 'traj1.dat'), {'dtype': np.int32}),
    (os.path.join(HERE, 'traj1.dat'), {'dtype': np.float64}),
    (os.path.join(HERE, 'data.dat'), {'no_traj': True}),
])
def test_openmicrostates(limits_file, traj_file, kwargs):
    """Test that the trajectory is split correctly."""
    if 'dtype' in kwargs and kwargs['dtype'] == np.float64:
        with pytest.raises(TypeError):
            iotext.openmicrostates(
                limits_file=limits_file,
                file_name=traj_file,
                **kwargs,
            )
    elif 'no_traj' in kwargs:
        with pytest.raises(iotext.FileError):
            iotext.openmicrostates(
                limits_file=limits_file,
                file_name=traj_file,
            )
    else:
        # check with limits file
        expected = [[1, 1, 1, 1, 1], [2, 2, 1, 2, 3], [2, 2, 3]]
        trajs = iotext.openmicrostates(
            limits_file=limits_file,
            file_name=traj_file,
            **kwargs,
        )
        for i, traj in enumerate(trajs):
            assert (traj == expected[i]).all()


@pytest.mark.parametrize('traj_file, expected', [
    (
        os.path.join(HERE, 'traj1.dat'),
        [[1, 1, 1, 1, 1], [2, 2, 1, 2, 3], [2, 2, 3]],
    ),
    (
        os.path.join(HERE, 'data.dat'),
        [
            [[1, 2], [1, 3], [1, 2], [1, 1], [1, 2]],
            [[2, 1], [2, 5], [1, 4], [2, 3], [3, 2]],
            [[2, 1], [2, 2], [3, 3]],
        ],
    ),
])
def test_opentxt_limits(limits_file, traj_file, expected):
    """Test that the trajectory is split correctly."""
    # check with limits file
    trajs = iotext.opentxt_limits(
        limits_file=limits_file,
        file_name=traj_file,
    )
    for i, traj in enumerate(trajs):
        assert (traj == expected[i]).all()

    # check without limits file
    if len(trajs[0].shape) == 1:
        expected = [[1, 1, 1, 1, 1, 2, 2, 1, 2, 3, 2, 2, 3]]
        trajs = iotext.opentxt_limits(file_name=traj_file)
        for i, traj in enumerate(trajs):
            assert (traj == expected[i]).all()

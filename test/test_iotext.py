"""Tests for the reader.feed module"""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os.path
import pytest
import msmhelper

# Current directory
HERE = os.path.dirname(__file__)


@pytest.fixture
def limits_file():
    """Define limits file."""
    return os.path.join(HERE, "limits.dat")


@pytest.fixture
def traj_file():
    """Define example trajectory file."""
    return os.path.join(HERE, "traj.dat")


# ~~~ TESTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def test_opentxt(traj_file):
    """Test that a file is opened correctly."""
    expected = [1, 1, 1, 1, 1, 2, 2, 1, 2, 3, 2, 2, 3]
    assert (msmhelper.opentxt(traj_file) == expected).all()


def test_open_limits(limits_file):
    """Test that the limits are loaded correctly."""
    assert (msmhelper.open_limits(13, limits_file) == [5, 10, 13]).all()
    assert (msmhelper.open_limits(13) == [13]).all()


def test_opentxt_limits(limits_file, traj_file):
    """Test that the trajectory is split correctly."""
    # check with limits file
    expected = [[1, 1, 1, 1, 1], [2, 2, 1, 2, 3], [2, 2, 3]]
    trajs = msmhelper.opentxt_limits(limits_file=limits_file,
                                     file_name=traj_file)
    for i, traj in enumerate(trajs):
        assert (traj == expected[i]).all()

    # check without limits file
    expected = [[1, 1, 1, 1, 1, 2, 2, 1, 2, 3, 2, 2, 3]]
    trajs = msmhelper.opentxt_limits(file_name=traj_file)
    for i, traj in enumerate(trajs):
        assert (traj == expected[i]).all()

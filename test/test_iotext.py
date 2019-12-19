"""Tests for the iotext module."""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os.path
import pytest
import msmhelper
import numpy as np

# Current directory
HERE = os.path.dirname(__file__)


@pytest.fixture
def limits_file():
    """Define limits file."""
    return os.path.join(HERE, "limits.dat")


# ~~~ TESTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@pytest.mark.parametrize("traj_file, comment", [
    (os.path.join(HERE, "data.dat"), '#'),
    (os.path.join(HERE, "traj1.dat"), '#'),
    (os.path.join(HERE, "traj1.dat"), ['#']),
    (os.path.join(HERE, "traj2.dat"), ['#', '@'])])
def test_opentxt(traj_file, comment):
    """Test that a file is opened correctly."""
    expected = [1, 1, 1, 1, 1, 2, 2, 1, 2, 3, 2, 2, 3]
    data = msmhelper.opentxt(traj_file, comment=comment)
    if len(data.shape) > 1:
        data = data[:, 0]
    assert (data == expected).all()


@pytest.mark.parametrize("traj_file, header", [
    (os.path.join(HERE, "traj1.dat"), None),
    (os.path.join(HERE, "traj1.dat"), "Test header comment.")])
def test_savetxt(traj_file, header, tmpdir):
    """Test that a file is opened correctly."""
    expected = [1, 1, 1, 1, 1, 2, 2, 1, 2, 3, 2, 2, 3]
    output = tmpdir.join('test_traj.txt')
    msmhelper.savetxt(output, expected, header=header, fmt='%.0f')
    assert (msmhelper.opentxt(output) == expected).all()


def test_open_limits(limits_file):
    """Test that the limits are loaded correctly."""
    assert (msmhelper.open_limits(13, limits_file) == [5, 10, 13]).all()
    assert (msmhelper.open_limits(13) == [13]).all()


@pytest.mark.parametrize("traj_file, kwargs", [
    (os.path.join(HERE, "traj1.dat"), {}),
    (os.path.join(HERE, "traj1.dat"), {'dtype': np.integer}),
    (os.path.join(HERE, "traj1.dat"), {'dtype': np.float})])
def test_opentxt_limits(limits_file, traj_file, kwargs):
    """Test that the trajectory is split correctly."""
    if 'dtype' in kwargs and kwargs['dtype'] == np.float:
        with pytest.raises(AssertionError):
            msmhelper.opentxt_limits(limits_file=limits_file,
                                     file_name=traj_file, **kwargs)
    else:
        # check with limits file
        expected = [[1, 1, 1, 1, 1], [2, 2, 1, 2, 3], [2, 2, 3]]
        trajs = msmhelper.opentxt_limits(limits_file=limits_file,
                                         file_name=traj_file, **kwargs)
        for i, traj in enumerate(trajs):
            assert (traj == expected[i]).all()

        # check without limits file
        expected = [[1, 1, 1, 1, 1, 2, 2, 1, 2, 3, 2, 2, 3]]
        trajs = msmhelper.opentxt_limits(file_name=traj_file)
        for i, traj in enumerate(trajs):
            assert (traj == expected[i]).all()

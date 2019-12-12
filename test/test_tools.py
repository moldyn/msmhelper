"""Tests for the iotext module."""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import pytest
import msmhelper


# ~~~ TESTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@pytest.mark.parametrize("data, expected, val_old, val_new", [
    ([1, 1, 1, 3, 2, 2], [2, 2, 2, 3, 1, 1], [1, 2, 3], [2, 1, 3]),
    ([1, 1, 1, 3, 2, 2], [2, 2, 2, 3, 1, 1], [1, 2], [2, 1]),
    ([1, 1, 1, 3, 2, 2], [3, 3, 3, 2, 1, 1], [1, 2, 3], [3, 1, 2]),
    ([[1, 1, 1], [3, 2, 2]], [[2, 2, 2], [3, 1, 1]], [1, 2, 3], [2, 1, 3])])
def test_opentxt(data, expected, val_old, val_new):
    """Test that a file is opened correctly."""
    trajs = msmhelper.shift_data(data, val_old, val_new)
    for i, traj in enumerate(trajs):
        assert (traj == expected[i]).all()

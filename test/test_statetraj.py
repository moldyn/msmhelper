# -*- coding: utf-8 -*-
"""Tests for the statetraj module.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

"""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pytest
from numpy import array, int32

from msmhelper.statetraj import LumpedStateTraj, StateTraj


@pytest.fixture
def index_traj():
    """Define index trajectory."""
    traj = array(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 1, 0, 2, 1, 2, 2, 2],
        dtype=int32,
    )
    return StateTraj(traj)


@pytest.fixture
def indextraj():
    """Define index trajectory."""
    return array(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 1, 0, 2, 1, 2, 2, 2],
        dtype=int32,
    )


@pytest.fixture
def state_traj():
    """Define state trajectory."""
    traj = array(
        [0, 0, 0, 3, 3, 3, 2, 2, 2, 3, 3, 3, 0, 2, 3, 2, 2, 2],
        dtype=int32,
    )
    return StateTraj(traj)


@pytest.fixture
def statetraj():
    """Define state trajectory."""
    return array(
        [0, 0, 0, 3, 3, 3, 2, 2, 2, 3, 3, 3, 0, 2, 3, 2, 2, 2],
        dtype=int32,
    )


@pytest.fixture
def macrotraj():
    """Define state trajectory."""
    return array(
        [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2],
        dtype=int32,
    )


@pytest.fixture
def macro_traj():
    """Define index trajectory."""
    traj = array(
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1, 3, 2, 3, 3, 3],
        dtype=int32,
    )
    macrotraj = array(
        [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2],
        dtype=int32,
    )
    return LumpedStateTraj(macrotrajs=macrotraj, microtrajs=traj)


# ~~~ TESTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def test_StateTraj_constructor(statetraj):
    """Test construction of object."""
    traj = StateTraj(statetraj)
    np.testing.assert_array_equal(
        statetraj,
        traj.state_trajs[0],
    )

    # check if passing StateTraj returns object
    assert traj is StateTraj(traj)
    # check if passing LumpedStateTraj returns object
    lumpedTraj = LumpedStateTraj(statetraj, statetraj)
    assert lumpedTraj is StateTraj(lumpedTraj)


def test_LumpedStateTraj_constructor(macrotraj, statetraj):
    """Test construction of object."""
    traj = LumpedStateTraj(macrotraj, statetraj)
    np.testing.assert_array_equal(
        macrotraj,
        traj.state_trajs[0],
    )

    np.testing.assert_array_equal(
        statetraj,
        traj.microstate_trajs[0],
    )

    # check if passing LumpedStateTraj returns object
    assert traj is LumpedStateTraj(traj)
    assert traj is StateTraj(traj)

    with pytest.raises(TypeError):
        LumpedStateTraj(macrotraj)


def test_nstates(state_traj):
    """Test nstates property."""
    assert state_traj.nstates == len(np.unique(state_traj[0]))

    with pytest.raises(AttributeError):
        state_traj.nstates = 5


def test_nframes(state_traj):
    """Test nframes property."""
    assert state_traj.nframes == len(state_traj[0])

    with pytest.raises(AttributeError):
        state_traj.nframes = 5


def test_ntrajs(state_traj):
    """Test ntrajs property."""
    assert state_traj.ntrajs == len(state_traj.trajs)

    with pytest.raises(AttributeError):
        state_traj.ntrajs = 5


def test_index_trajs(state_traj, index_traj):
    """Test index trajs property."""
    np.testing.assert_array_equal(
        index_traj.trajs,
        index_traj.state_trajs,
    )
    np.testing.assert_array_equal(
        state_traj.trajs,
        state_traj.index_trajs,
    )

    with pytest.raises(AttributeError):
        state_traj.trajs = 5
    with pytest.raises(AttributeError):
        state_traj.index_trajs = 5


def test_trajs_flatten(state_traj, index_traj):
    """Test flatten index trajectory."""
    np.testing.assert_array_equal(
        index_traj.trajs[0],
        index_traj.trajs_flatten,
    )
    np.testing.assert_array_equal(
        state_traj.trajs[0],
        state_traj.trajs_flatten,
    )


def test_state_trajs_flatten(state_traj, index_traj):
    """Test flatten state trajectory."""
    np.testing.assert_array_equal(
        index_traj.state_trajs[0],
        index_traj.state_trajs_flatten,
    )
    np.testing.assert_array_equal(
        state_traj.state_trajs[0],
        state_traj.state_trajs_flatten,
    )


def test___eq__(state_traj, index_traj):
    """Test eq method."""
    for traj in [state_traj, index_traj]:
        assert StateTraj(traj.state_trajs) == traj

    assert state_traj != index_traj
    assert state_traj != 5


def test___repr__(state_traj, index_traj):
    """Test repr method."""
    for traj in [state_traj, index_traj]:
        assert eval(traj.__repr__()) == traj  # noqa: S307


def test___str__(state_traj, index_traj):
    """Test str method."""
    for traj in [state_traj, index_traj]:
        assert traj.__str__().startswith('[')


def test_as_list(state_traj, index_traj):
    """Test iterating over object."""
    for traj in [state_traj, index_traj]:
        for trajectory in traj:
            assert StateTraj(trajectory) == traj

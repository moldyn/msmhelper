# -*- coding: utf-8 -*-
"""Tests for the statetraj module.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

"""
import numpy as np
import pytest

from msmhelper.statetraj import LumpedStateTraj, StateTraj


@pytest.fixture
def index_traj():
    """Define index trajectory."""
    traj = np.array(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 1, 0, 2, 1, 2, 2, 2],
        dtype=np.int32,
    )
    return StateTraj(traj)


@pytest.fixture
def indextraj():
    """Define index trajectory."""
    return np.array(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 1, 0, 2, 1, 2, 2, 2],
        dtype=np.int32,
    )


@pytest.fixture
def state_traj():
    """Define state trajectory."""
    traj = np.array(
        [0, 0, 0, 3, 3, 3, 2, 2, 2, 3, 3, 3, 0, 2, 3, 2, 2, 2],
        dtype=np.int32,
    )
    return StateTraj(traj)


@pytest.fixture
def statetraj():
    """Define state trajectory."""
    return np.array(
        [0, 0, 0, 3, 3, 3, 2, 2, 2, 3, 3, 3, 0, 2, 3, 2, 2, 2],
        dtype=np.int32,
    )


@pytest.fixture
def macrotraj():
    """Define state trajectory."""
    return np.array(
        [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2],
        dtype=np.int32,
    )


@pytest.fixture
def macro_indextraj():
    """Define state trajectory."""
    return np.array(
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        dtype=np.int32,
    )


@pytest.fixture
def macro_traj():
    """Define index trajectory."""
    traj = np.array(
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1, 3, 2, 3, 3, 3],
        dtype=np.int32,
    )
    macrotraj = np.array(
        [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2],
        dtype=np.int32,
    )
    return LumpedStateTraj(macrotrajs=macrotraj, microtrajs=traj)


def test_StateTraj_constructor(statetraj):
    """Test construction of object."""
    traj = StateTraj(statetraj)
    np.testing.assert_array_equal(
        statetraj,
        traj.trajs[0],
    )

    # check if passing StateTraj returns object
    assert traj is StateTraj(traj)
    # check if passing LumpedStateTraj returns object
    lumpedTraj = LumpedStateTraj(statetraj, statetraj)
    assert lumpedTraj is StateTraj(lumpedTraj)

    # check that immutable
    assert traj._trajs[0] is not StateTraj(traj.trajs)._trajs[0]
    # check for index trajs
    assert (
        StateTraj(traj.index_trajs)._trajs[0] is not
        StateTraj(traj.index_trajs)._trajs[0]
    )


def test_LumpedStateTraj_constructor(macrotraj, statetraj):
    """Test construction of object."""
    traj = LumpedStateTraj(macrotraj, statetraj)
    np.testing.assert_array_equal(
        macrotraj,
        traj.trajs[0],
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


def test_nstates(state_traj, statetraj, macro_traj, macrotraj):
    """Test nstates property."""
    assert state_traj.nstates == len(np.unique(statetraj))
    assert macro_traj.nstates == len(np.unique(macrotraj))

    with pytest.raises(AttributeError):
        state_traj.nstates = 5


def test_states(state_traj):
    """Test immutability of states property."""
    assert state_traj.states is not state_traj.states

    with pytest.raises(AttributeError):
        state_traj.states = state_traj.states


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
        index_traj.index_trajs,
    )

    with pytest.raises(AttributeError):
        state_traj.trajs = 5
    with pytest.raises(AttributeError):
        state_traj.index_trajs = 5


def test_macroindex_trajs(macro_traj, indextraj, macro_indextraj):
    """Test index trajs property."""
    np.testing.assert_array_equal(
        macro_traj.index_trajs_flatten,
        macro_indextraj,
    )
    np.testing.assert_array_equal(
        macro_traj.microstate_index_trajs_flatten,
        indextraj,
    )


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


def test_index_trajs_flatten(state_traj, index_traj):
    """Test flatten index trajectory."""
    np.testing.assert_array_equal(
        index_traj.index_trajs[0],
        index_traj.index_trajs_flatten,
    )
    np.testing.assert_array_equal(
        state_traj.index_trajs[0],
        state_traj.index_trajs_flatten,
    )


def test_microstate_trajs(macrotraj, statetraj, indextraj):
    """Test flatten trajectory."""
    macro_traj = LumpedStateTraj(macrotraj, statetraj)
    np.testing.assert_array_equal(
        statetraj,
        macro_traj.microstate_trajs[0],
    )
    np.testing.assert_array_equal(
        statetraj,
        macro_traj.microstate_trajs_flatten,
    )
    np.testing.assert_array_equal(
        macrotraj,
        macro_traj.trajs[0],
    )
    np.testing.assert_array_equal(
        macrotraj,
        macro_traj.trajs_flatten,
    )

    # check that state_trajs cannot be set
    with pytest.raises(AttributeError):
        macro_traj.trajs = 5

    # check for index trajs
    macro_traj = LumpedStateTraj(macrotraj, indextraj)
    np.testing.assert_array_equal(
        indextraj,
        macro_traj.microstate_trajs[0],
    )
    np.testing.assert_array_equal(
        indextraj,
        macro_traj.microstate_trajs_flatten,
    )
    np.testing.assert_array_equal(
        indextraj,
        macro_traj.microstate_index_trajs[0],
    )
    np.testing.assert_array_equal(
        indextraj,
        macro_traj.microstate_index_trajs_flatten,
    )


def test___eq__(state_traj, index_traj):
    """Test eq method."""
    for traj in [state_traj, index_traj]:
        assert StateTraj(traj.trajs) == traj

    assert state_traj != index_traj
    assert state_traj != 5


def test_LumpedStateTraj__eq__(macro_traj):
    """Test eq method."""
    assert LumpedStateTraj(
        macro_traj.trajs, macro_traj.microstate_trajs,
    ) == macro_traj

    assert macro_traj != LumpedStateTraj([0, 0], [1, 0])
    assert macro_traj != 5


def test___repr__(state_traj, index_traj, macro_traj):
    """Test repr method."""
    # used implicitly for repr evaluation
    array = np.array  # noqa: F841
    int32 = np.int32  # noqa: F841
    for traj in [state_traj, index_traj, macro_traj]:
        assert eval(traj.__repr__()) == traj  # noqa: S307


def test___str__(state_traj, index_traj, macro_traj):
    """Test str method."""
    for traj in [state_traj, index_traj, macro_traj]:
        assert traj.__str__().startswith('[')
        assert traj.__str__().endswith(']')


def test_as_list(state_traj, index_traj):
    """Test iterating over object."""
    for traj in [state_traj, index_traj]:
        for trajectory in traj:
            assert StateTraj(trajectory) == traj


def test_LumpedStateTraj_as_list(statetraj, indextraj, macrotraj):
    """Test iterating over object."""
    for traj in [statetraj, indextraj, macrotraj]:
        macro_traj = LumpedStateTraj(traj, traj)
        for trajectory in macro_traj:
            assert LumpedStateTraj(trajectory, trajectory) == macro_traj

        np.testing.assert_array_equal(macro_traj[0], traj)


def test_LumpedStateTraj_estimate_markov_model(macro_traj):
    """Check MSM estimation from microstate matrix."""
    tlag = 1
    states_ref = np.array([1, 2])
    tmat_ref = np.array([[0.50177725, 0.49822275], [0.06872038, 0.93127962]])

    tmat, states = macro_traj.estimate_markov_model(tlag)
    np.testing.assert_array_almost_equal(tmat, tmat_ref)
    np.testing.assert_array_equal(states, states_ref)

    with pytest.raises(TypeError):
        macrotraj = macro_traj.trajs_flatten
        microtraj = macro_traj.microstate_trajs_flatten
        microtraj[-1] = np.max(microtraj) + 1
        trap_traj = LumpedStateTraj(macrotraj, microtraj)
        trap_traj.estimate_markov_model(tlag)

    # enforce T_ij >= 0
    macro_traj = LumpedStateTraj(
        macro_traj.trajs,
        macro_traj.microstate_trajs,
        positive=True,
    )

    tmat, states = macro_traj.estimate_markov_model(tlag)
    assert np.all(tmat >= 0)
    np.testing.assert_array_almost_equal(tmat, tmat_ref)
    np.testing.assert_array_equal(states, states_ref)


@pytest.mark.parametrize('traj, state, idx, error', [
    ([1, 2, 4], 1, 0, None),
    ([1, 2, 4], 0, None, ValueError),
    ([1, 8, 3], 8, 2, None),
    ([1, 8, 3, 1], 3, 1, None),
])
def test_state_to_idx(traj, state, idx, error):
    """Check state to idx conversion."""
    statetraj = StateTraj(traj)
    for straj in (
        statetraj,
        LumpedStateTraj(statetraj, np.arange(statetraj.nframes)),
    ):
        if error is not None:
            with pytest.raises(error):
                straj.state_to_idx(state)
        else:
            assert straj.state_to_idx(state) == idx

# -*- coding: utf-8 -*-
# BSD 3-Clause License
# Copyright (c) 2019-2023, Daniel Nagel
# All rights reserved.
"""Set of functions for analyzing the MD trajectory.

This submodule contains methods for estimating various timescales based on a
given state trajectory.

"""
from collections import defaultdict

import decorit
import numba
import numpy as np

from msmhelper.md.comparison import _intersect as intersect
from msmhelper.statetraj import StateTraj


@decorit.alias('estimate_wt')
def estimate_waiting_times(trajs, start, final):
    """Estimates waiting times between stated states.

    The stated states (from/to) will be treated as a basin. The function
    calculates all transitions from first entering the start-basin until first
    reaching the final-basin.

    Parameters
    ----------
    trajs : statetraj or list or ndarray or list of ndarray
        State trajectory/trajectories. The states should start from zero and
        need to be integers.
    start : int or list of
        States to start counting.
    final : int or list of
        States to start counting.

    Returns
    -------
    wt : ndarray
        List of waiting times, given in frames.

    """
    # check correct input format
    trajs = StateTraj(trajs)

    states_start, states_final = np.unique(start), np.unique(final)

    if intersect(states_start, states_final):
        raise ValueError('States `start` and `final` do overlap.')

    # check that all states exist in trajectory
    for states in (states_start, states_final):
        if intersect(states, trajs.states) != len(states):
            raise ValueError(
                'Selected states does not exist in state trajectory.',
            )

    # do not convert for pytest coverage
    if numba.config.DISABLE_JIT:
        return _estimate_waiting_times(trajs, states_start, states_final)
    return _estimate_waiting_times(  # pragma: no cover
        numba.typed.List(trajs),
        numba.typed.List(states_start),
        numba.typed.List(states_final),
    )


@numba.njit
def _estimate_waiting_times(trajs, states_start, states_final):
    """Estimate waiting time between subsets of states."""
    times = []
    for traj in trajs:
        times.extend(
            _estimate_waiting_times_singletraj(
                traj, states_start, states_final,
            ),
        )
    return np.array(times)


@numba.njit
def _estimate_waiting_times_singletraj(traj, states_start, states_final):
    """Estimate waiting time between subsets of states for a single traj."""
    paths_idx = _estimate_events_singletraj(traj, states_start, states_final)
    return [idx_end - idx_start for idx_start, idx_end in paths_idx]


def estimate_paths(trajs, start, final):
    """Estimates paths and waiting times between stated states.

    The stated states (from/to) will be treated as a basin. The function
    calculates all transitions from first entering the start-basin until first
    reaching the final-basin. The results will be listed by the corresponding
    pathways, where loops are removed occuring first.

    Parameters
    ----------
    trajs : statetraj or list or ndarray or list of ndarray
        State trajectory/trajectories. The states should start from zero and
        need to be integers.
    start : int or list of
        States to start counting.
    final : int or list of
        States to start counting.

    Returns
    -------
    paths : dict
        Dictionary containing the the paths as keys and and an array holding
        the times of all paths as value.

    """
    # check correct input format
    trajs = StateTraj(trajs)

    states_start, states_final = np.unique(start), np.unique(final)

    if intersect(states_start, states_final):
        raise ValueError('States `start` and `final` do overlap.')

    # check that all states exist in trajectory
    for states in (states_start, states_final):
        if intersect(states, trajs.states) != len(states):
            raise ValueError(
                'Selected states does not exist in state trajectory.',
            )

    # do not convert for pytest coverage
    if numba.config.DISABLE_JIT:  # pragma: no cover
        path_tuples = _estimate_paths(trajs, states_start, states_final)
    else:
        path_tuples = _estimate_paths(  # pragma: no cover
            numba.typed.List(trajs),
            numba.typed.List(states_start),
            numba.typed.List(states_final),
        )
    paths = defaultdict(list)
    for path, time in path_tuples:
        paths[tuple(path)].append(time)

    return paths


@numba.njit
def _estimate_paths(trajs, states_start, states_final):
    """Estimate waiting time between subsets of states."""
    paths = []
    for traj in trajs:
        paths.extend(
            _estimate_paths_singletraj(
                traj, states_start, states_final,
            ),
        )
    return paths


@numba.njit
def _estimate_events_singletraj(traj, states_start, states_final):
    """Estimate waiting time between subsets of states for a single traj."""
    paths_idx = []
    propagates_forwards = False
    idx_start = 0
    for idx in range(len(traj)):  # noqa: WPS518
        state = traj[idx]

        if not propagates_forwards and state in states_start:
            propagates_forwards = True
            idx_start = idx
        elif propagates_forwards and state in states_final:
            propagates_forwards = False
            paths_idx.append((idx_start, idx))

    return paths_idx


@numba.njit
def _estimate_paths_singletraj(traj, states_start, states_final):
    """Estimate waiting time between subsets of states for a single traj."""
    paths_idx = _estimate_events_singletraj(traj, states_start, states_final)

    paths = []
    for idx_start, idx_end in paths_idx:
        for state in traj[idx_start: idx_end + 1]:  # noqa: WPS518
            if state in states_start:
                path = [int(define_type_var) for define_type_var in range(0)]
            elif state in path:
                path = path[: path.index(state)]
            path.append(state)

        paths.append((path, idx_end - idx_start))
    return paths

# -*- coding: utf-8 -*-
"""Set of functions for analyzing the MD trajectory.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

"""
import numba
import numpy as np

from msmhelper.compare import _intersect as intersect
from msmhelper.decorators import shortcut
from msmhelper.statetraj import StateTraj


@shortcut('estimate_wt')
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
                'Selected states does not exist in state trajectoty.',
            )

    # do not convert for pytest coverage
    if numba.config.DISABLE_JIT:
        return _estimate_waiting_times(trajs, states_start, states_final)
    return _estimate_waiting_times(
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
    times = []
    propagates_forwards = False
    idx_start = 0
    for idx in range(len(traj)):  # noqa: WPS518
        state = traj[idx]

        if not propagates_forwards and state in states_start:
            propagates_forwards = True
            idx_start = idx
        elif propagates_forwards and state in states_final:
            propagates_forwards = False
            times.append(idx - idx_start)

    return times

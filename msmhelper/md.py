# -*- coding: utf-8 -*-
"""Set of functions for analyzing the MD trajectory.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

"""
import random
from collections import defaultdict

import decorit
import numba
import numpy as np

from msmhelper.compare import _intersect as intersect
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
                'Selected states does not exist in state trajectoty.',
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
    paths : ndarray
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


@decorit.alias('estimate_msm_wt')
def estimate_msm_waiting_times(
    *,
    trajs,
    lagtime,
    start,
    final,
    steps,
    return_list=False,
):
    """Estimates waiting times between stated states.

    The stated states (from/to) will be treated as a basin. The function
    calculates all transitions from first entering the start-basin until first
    reaching the final-basin.

    Parameters
    ----------
    trajs : statetraj or list or ndarray or list of ndarray
        State trajectory/trajectories. The states should start from zero and
        need to be integers.

    lagtime : int
        Lag time for estimating the markov model given in [frames].

    start : int or list of
        States to start counting.

    final : int or list of
        States to start counting.

    steps : int
        Number of MCMC propagation steps of MCMC run.

    return_list : bool
        If true a list of all events is returned, else a dictionary is
        returned.

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

    # convert states to idx
    idxs_start = np.array(
        [trajs.state_to_idx(state) for state in states_start],
    )
    idxs_final = np.array(
        [trajs.state_to_idx(state) for state in states_final],
    )
    start = np.random.choice(idxs_final)

    # do not convert for pytest coverage
    if not numba.config.DISABLE_JIT:
        idxs_start = numba.typed.List(idxs_start)
        idxs_final = numba.typed.List(idxs_final)

    # estimate cummulative transition matrix
    cummat = _get_cummat(trajs=trajs, lagtime=lagtime)

    wts = _estimate_msm_waiting_times(
        cummat=cummat,
        start=start,
        states_from=idxs_start,
        states_to=idxs_final,
        steps=steps,
    )
    # multiply wts by lagtime
    if return_list:
        return np.repeat(
            list(wts.keys()), list(wts.values()),
        ) * lagtime
    return {wt * lagtime: count for wt, count in wts.items()}


@numba.njit
def _propagate_MCMC_step(cummat, idx_from):
    """Propagate a single step Markov chain Monte Carlo."""
    rand = random.random()  # noqa: S311
    cummat_perm, state_perm = cummat
    cummat_perm, state_perm = cummat_perm[idx_from], state_perm[idx_from]

    for idx, cummat_idx in enumerate(cummat_perm):
        # strict less to ensure that rand=0 does not jump along unconnected
        # states with Tij=0.
        if rand < cummat_idx:
            return state_perm[idx]
    # this should never be reached, but needed for numba to ensure int return
    return len(cummat_perm) - 1


@numba.njit
def _estimate_msm_waiting_times(
    cummat, start, states_from, states_to, steps,
):
    wts = {}

    idx_start = 0
    propagates_forwards = False
    state = start
    for idx in range(steps):
        state = _propagate_MCMC_step(cummat=cummat, idx_from=state)

        if not propagates_forwards and state in states_from:
            propagates_forwards = True
            idx_start = idx
        elif propagates_forwards and state in states_to:
            propagates_forwards = False
            wt = idx - idx_start
            if wt in wts:
                wts[wt] += 1
            else:
                wts[wt] = 1

    return wts


def propagate_MCMC(
    trajs,
    lagtime,
    steps,
    start=-1,
):
    """Propagate MCMC trajectory.

    Parameters
    ----------
    trajs : statetraj or list or ndarray or list of ndarray
        State trajectory/trajectories. The states should start from zero and
        need to be integers.

    lagtime : int
        Lag time for estimating the markov model given in [frames].

    steps : int
        Number of MCMC propagation steps.

    start : int or list of, optional
        State to start propagating. Default (-1) is random state.

    Returns
    -------
    mcmc : ndarray
        MCMC trajecory.

    """
    # check correct input format
    trajs = StateTraj(trajs)

    # check that all states exist in trajectory
    if start == -1:
        start = np.random.choice(trajs.states)
    elif start not in trajs.states:
        raise ValueError(
            'Selected starting state does not exist in state trajectoty.',
        )

    # convert states to idx
    idx_start = trajs.state_to_idx(start)

    # estimate permuted cummulative transition matrix
    cummat = _get_cummat(trajs=trajs, lagtime=lagtime)

    # do not convert for pytest coverage
    return _propagate_MCMC(  # pragma: no cover
        cummat=cummat,
        start=idx_start,
        steps=steps,
    )


@numba.njit
def _propagate_MCMC(cummat, start, steps):
    mcmc = np.empty(steps, dtype=np.int32)

    state = start
    mcmc[0] = state
    for idx in range(steps - 1):
        state = _propagate_MCMC_step(cummat=cummat, idx_from=state)
        mcmc[idx + 1] = state

    return mcmc


def _get_cummat(trajs, lagtime):
    # estimate cummulative transition matrix
    msm, _ = StateTraj(trajs).estimate_markov_model(lagtime)

    cummat_perm = np.empty_like(msm)
    state_perm = np.empty_like(msm, dtype=np.int64)

    for idx, row in enumerate(msm):
        idx_sort = np.argsort(row)[::-1]

        cummat_perm[idx] = np.cumsum(row[idx_sort])
        state_perm[idx] = idx_sort

    cummat_perm[:, -1] = 1  # enforce that probability sums up to 1
    return cummat_perm, state_perm

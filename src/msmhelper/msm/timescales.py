# -*- coding: utf-8 -*-
# BSD 3-Clause License
# Copyright (c) 2019-2023, Daniel Nagel
# All rights reserved.
"""# Set of functions for analyzing the MD trajectory.

This submodule contains methods for estimating various timescales based on a
Markov model.

"""
import random

import decorit
import numba
import numpy as np
from matplotlib.cbook import boxplot_stats

from msmhelper.md.comparison import _intersect as intersect
from msmhelper.md import estimate_paths as md_estimate_paths
from msmhelper.md import estimate_waiting_times as md_estimate_wt
from msmhelper.msm.utils import linalg
from msmhelper.utils import shift_data
from msmhelper.statetraj import StateTraj


def implied_timescales(trajs, lagtimes, ntimescales=None, reversible=False):
    r"""Calculate the implied timescales.

    Calculate the implied timescales, which are defined by

    $$t_i = - \frac{t_\text{lag}}{\log\lambda_i}$$

    the $i$-th eigenvalue $\lambda_i$.

    !!! note
        It is not checked if for higher lagtimes the dimensionality changes.

    Parameters
    ----------
    trajs : StateTraj or list or ndarray or list of ndarray
        State trajectory/trajectories. The states should start from zero and
        need to be integers.
    lagtimes : list or ndarray int
        Lagtimes for estimating the markov model given in [frames].
        This is not implemented yet!
    ntimescales : int, optional
        Number of returned lagtimes.
    reversible : bool
        If reversibility should be enforced for the markov state model.

    Returns
    -------
    ts : ndarray
        Matrix containing the implied Timescales.

    """
    # format input
    trajs = StateTraj(trajs)
    lagtimes = np.atleast_1d(lagtimes)

    # check that lag times are array of integers
    if not np.issubdtype(lagtimes.dtype, np.integer):
        raise TypeError(
            'Lagtimes needs to be integers but are {0}'.format(lagtimes.dtype),
        )
    if not (lagtimes > 0).all():
        raise TypeError('Lagtimes needs to be positive integers')
    if reversible:
        raise NotImplementedError(
            'Reversible matrices are not anymore supported.'
        )

    if ntimescales is None:
        ntimescales = trajs.nstates - 1

    # initialize result
    impl_timescales = np.zeros((len(lagtimes), ntimescales))

    for idx, lagtime in enumerate(lagtimes):
        transmat, _ = trajs.estimate_markov_model(lagtime)
        impl_timescales[idx] = _implied_timescales(
            transmat, lagtime, ntimescales=ntimescales,
        )

    return impl_timescales


def _implied_timescales(tmat, lagtime, ntimescales):
    """Calculate implied timescales.

    Parameters
    ----------
    tmat : ndarray
        Quadratic transition matrix.
    lagtime : int
        Lagtime for estimating the markov model given in [frames].
    ntimescales : int, optional
        Number of returned timescales.

    Returns
    -------
    timescales : ndarray
        Implied timescale given in frames.

    """
    tmat = np.asarray(tmat)

    eigenvalues = linalg.left_eigenvalues(tmat, nvals=ntimescales + 1)
    # for negative eigenvalues no timescale is defined
    eigenvalues[eigenvalues < 0] = np.nan
    return np.ma.divide(- lagtime, np.log(eigenvalues[1:]))


def _estimate_times(
    *,
    trajs,
    lagtime,
    start,
    final,
    steps,
    estimator,
    return_list=False,
):
    """Estimates times between stated states.

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
    estimator : function
        Estimator to propagate mcmc and return times.
    return_list : bool
        If true a list of all events is returned, else the probability density
        together with the edges is returned.

    Returns
    -------
    ts : ndarray
        Density probability of the time distribution. If `return_list=True`,
        return a sorted (!) list containing all times.
    edges : ndarray
        Array containing the edges corresponding to the probability, given in
        frames. Only for `return_list=False`.


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

    # convert states to idx
    idxs_start = np.array(
        [trajs.state_to_idx(state) for state in states_start],
    )
    idxs_final = np.array(
        [trajs.state_to_idx(state) for state in states_final],
    )
    start = np.random.choice(idxs_final)

    # do not convert for pytest coverage
    if not numba.config.DISABLE_JIT:  # pragma: no cover
        idxs_start = numba.typed.List(idxs_start)
        idxs_final = numba.typed.List(idxs_final)

    # estimate cumulative transition matrix
    cummat = _get_cummat(trajs=trajs, lagtime=lagtime)

    ts = estimator(
        cummat=cummat,
        start=start,
        states_from=idxs_start,
        states_to=idxs_final,
        steps=steps,
    )
    # multiply wts by lagtime
    if return_list:
        return np.repeat(
            list(ts.keys()), list(ts.values()),
        ) * lagtime

    maxtime = max(ts.keys())
    pts = np.zeros(maxtime + 1)
    for time, count in ts.items():
        pts[time] = count
    return (
        pts / (pts.sum() * lagtime),
        np.arange(len(pts) + 1) * lagtime,
    )


@decorit.alias('estimate_wt')
def estimate_waiting_times(
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
        If true a list of all events is returned, else the probability density
        together with the edges is returned.

    Returns
    -------
    ts : ndarray
        Density probability of the time distribution. If `return_list=True`,
        return a sorted (!) list containing all times.
    edges : ndarray
        Array containing the edges corresponding to the probability, given in
        frames. Only for `return_list=False`.

    """
    return _estimate_times(
        trajs=trajs,
        lagtime=lagtime,
        start=start,
        final=final,
        steps=steps,
        estimator=_estimate_waiting_times,
        return_list=return_list,
    )


@decorit.alias('estimate_tt')
def estimate_transition_times(
    *,
    trajs,
    lagtime,
    start,
    final,
    steps,
    return_list=False,
):
    """Estimates transition times between stated states.

    The stated states (from/to) will be treated as a basin. The function
    calculates all transitions from leaving the start-basin until first
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
        If true a list of all events is returned, else the probability density
        together with the edges is returned.

    Returns
    -------
    ts : ndarray
        Density probability of the time distribution. If `return_list=True`,
        return a sorted (!) list containing all times.
    edges : ndarray
        Array containing the edges corresponding to the probability, given in
        frames. Only for `return_list=False`.

    """
    return _estimate_times(
        trajs=trajs,
        lagtime=lagtime,
        start=start,
        final=final,
        steps=steps,
        estimator=_estimate_transition_times,
        return_list=return_list,
    )


@numba.njit
def _propagate_MCMC_step(cummat, idx_from):
    """Propagate a single step Markov chain Monte Carlo."""
    rand = random.random()  # noqa: S311
    cummat_perm, state_perm = cummat
    cummat_from, state_from = cummat_perm[idx_from], state_perm[idx_from]

    for idx, cummat_idx in enumerate(cummat_from):
        # strict less to ensure that rand=0 does not jump along unconnected
        # states with Tij=0.
        if rand < cummat_idx:
            return state_from[idx]
    # this should never be reached, but needed for numba to ensure int return
    return state_from[np.argmax(cummat_from)]  # pragma: no cover


@numba.njit
def _estimate_waiting_times(
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


@numba.njit
def _estimate_transition_times(
    cummat, start, states_from, states_to, steps,
):
    tpts = {}

    idx_start = 0
    propagates_forwards = False
    state = start
    for idx in range(steps):
        state = _propagate_MCMC_step(cummat=cummat, idx_from=state)

        if state in states_from:
            propagates_forwards = True
            idx_start = idx
        elif propagates_forwards and state in states_to:
            propagates_forwards = False
            wt = idx - idx_start
            if wt in tpts:
                tpts[wt] += 1
            else:
                tpts[wt] = 1

    return tpts


def estimate_paths(
    *,
    trajs,
    lagtime,
    start,
    final,
    steps,
):
    """Estimates paths and waiting times between stated states.

    The stated states (from/to) will be treated as a basin. The function
    estimates transitions from first entering the start-basin until first
    reaching the final-basin. The results will be listed by the corresponding
    pathways, where loops are removed occuring first.

    !!! note
        This function is a simple wrapper and in contrast to
        [estimate_wt][msmhelper.msm.estimate_waiting_times] it stores the whole
        MCMC trajectory in memory. Hence, it memory-hungry.

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

    return md_estimate_paths(
        propagate_MCMC(trajs, lagtime, steps),
        start,
        final,
    )


def propagate_MCMC(
    trajs,
    lagtime,
    steps,
    start=-1,
):
    """Propagate Monte Carlo Markov chain.

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
        Monte Carlo Markov chain state trajectory.

    """
    # check correct input format
    trajs = StateTraj(trajs)

    # check that all states exist in trajectory
    if start == -1:
        start = np.random.choice(trajs.states)
    elif start not in trajs.states:
        raise ValueError(
            'Selected starting state does not exist in state trajectory.',
        )

    # convert states to idx
    idx_start = trajs.state_to_idx(start)

    # estimate permuted cumulative transition matrix
    cummat = _get_cummat(trajs=trajs, lagtime=lagtime)

    # do not convert for pytest coverage
    return shift_data(
        _propagate_MCMC(  # pragma: no cover
            cummat=cummat,
            start=idx_start,
            steps=steps,
        ),
        np.arange(trajs.nstates),
        trajs.states,
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
    # estimate cumulative transition matrix
    msm, _ = StateTraj(trajs).estimate_markov_model(lagtime)

    if np.any(msm < 0):
        raise ValueError('An entry of T_ij is less than 0!')

    cummat_perm = np.empty_like(msm)
    state_perm = np.empty_like(msm, dtype=np.int64)

    for idx, row in enumerate(msm):
        idx_sort = np.argsort(row)[::-1]

        cummat_perm[idx] = np.cumsum(row[idx_sort])
        state_perm[idx] = idx_sort

    cummat_perm[:, -1] = 1  # enforce that probability sums up to 1
    return cummat_perm, state_perm


def _estimate_stats(coord):
    """Return boxplot stats of data."""
    Q1, Q2, Q3 = np.quantile(
        coord,
        [0.25, 0.5, 0.75],
    )
    IQR = Q3 - Q1
    return (
        coord.min(),
        max(Q1 - IQR, coord.min()),
        Q1,
        Q2,
        Q3,
        min(Q3 + IQR, coord.max()),
        coord.max()
    )


@decorit.alias('estimate_wtd')
def estimate_waiting_time_dist(
    trajs,
    max_lagtime,
    start,
    final,
    steps,
    n_lagtimes=50,
):
    """Estimate waiting time distribution.

    Parameters
    ----------
    trajs : statetraj or list or ndarray or list of ndarray
        State trajectory/trajectories. The states should start from zero and
        need to be integers.
    max_lagtime : int
        Maximal lag time for estimating the markov model given in [frames].
    start : int or list of
        States to start counting.
    final : int or list of
        States to start counting.
    steps : int
        Number of MCMC propagation steps of MCMC run.

    Returns
    -------
    wtd : dict
        Dictionary containing waiting time distribution.

    """
    lagtimes = np.unique(
        np.linspace(1, max_lagtime, num=n_lagtimes, dtype=int),
    )

    # get stats
    wtd = {
        lagtime: boxplot_stats(
            estimate_waiting_times(
                trajs=trajs,
                lagtime=lagtime,
                start=start,
                final=final,
                steps=steps,
                return_list=True,
            ),
        )[0]
        for lagtime in lagtimes
    }

    # include MD
    wtd['MD'] = boxplot_stats(
        md_estimate_wt(
            trajs=trajs,
            start=start,
            final=final,
        ),
    )
    return wtd

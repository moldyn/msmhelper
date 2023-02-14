# -*- coding: utf-8 -*-
# BSD 3-Clause License
# Copyright (c) 2019-2023, Daniel Nagel
# All rights reserved.
"""Set of datasets to use for the tutorials."""
import numpy as np

from msmhelper.msm.timescales import _propagate_MCMC
from .tests import is_transition_matrix
from ._utils import shift_data


def propagate_tmat(tmat, nsteps, start=None):
    """Markov chain Monte Carlo propagation of transition matrix for nsteps.

    Parameters
    ----------
    tmat : ndarray
        Transition matrix to propagate.
    nsteps : int
        Number of steps to propagate.
    start : int
        Index where to start. If `None` a random number will be used.

    Returns
    -------
    traj : ndarray
        Markov chain Monte Carlo state trajectory of given tmat.

    """
    if not is_transition_matrix(tmat):
        raise ValueError('tmat needs to be a row-normalized matrix.')

    n_states = len(tmat)
    cummat = np.cumsum(tmat, axis=1)
    cummat[:, -1] = 1  # enforce exact normalization
    cummat_perm = np.tile(np.arange(n_states), (n_states, 1))

    if start is None:
        start = np.random.randint(n_states)

    return _propagate_MCMC(
        cummat=(cummat, cummat_perm),
        start=start,
        steps=nsteps,
    )


def _hummer15_nstate(*, n_states, rate_k, rate_h, nsteps, macrotraj=False):
    """N-state model inspired by Hummer and Szabo 15.

    Gerhard Hummer and Attila Szabo
    The Journal of Physical Chemistry B 2015 119 (29), 9029-9037
    DOI: [10.1021/jp508375q](https://pubs.acs.org/doi/10.1021/jp508375q)

    Parameters
    ----------
    rate_k : float
        Rate between state 1<->2, 3<->4, etc.
    rate_h : float
        Rate between state 2<->3, 4<->5, etc.
    nsteps : int
        Number of steps to propagate.
    macrotraj : bool, optional
        If `True` return macrotraj where state (1,2), (3,4), etc. are lumped.

    Returns
    -------
    traj : ndarray
        Markov chain Monte Carlo state trajectory.

    """
    tmat = np.zeros((n_states, n_states))
    for state in range(n_states):
        for neighbor, rate in (
            (state - 1, rate_k if state % 2 else rate_h),
            (state + 1, rate_h if state % 2 else rate_k),
        ):
            if 0 <= neighbor < n_states:
                tmat[state, neighbor] = rate

    # set diagonal
    tmat[np.diag_indices_from(tmat)] = 1 - np.sum(tmat, axis=1)

    states = np.tile(
        np.arange(1, n_states // 2 + 1),
        (2, 1),
    ).T.flatten() if macrotraj else np.arange(1, n_states + 1)

    return shift_data(
        propagate_tmat(tmat, nsteps),
        np.arange(n_states),
        states,
    )


def hummer15_4state(rate_k, rate_h, nsteps, macrotraj=False):
    """Four state model taken from Hummer and Szabo 15.

    Gerhard Hummer and Attila Szabo
    The Journal of Physical Chemistry B 2015 119 (29), 9029-9037
    DOI: [10.1021/jp508375q](https://pubs.acs.org/doi/10.1021/jp508375q)

    Parameters
    ----------
    rate_k : float
        Rate between state 1<->2 and 3<->4.
    rate_h : float
        Rate between state 2<->3.
    nsteps : int
        Number of steps to propagate.
    macrotraj : bool, optional
        If `True` return 2-state macrotraj where state 1,2 and 3,4 are lumped.

    Returns
    -------
    traj : ndarray
        Markov chain Monte Carlo state trajectory.

    """
    return _hummer15_nstate(
        n_states=4,
        rate_k=rate_k,
        rate_h=rate_h,
        nsteps=nsteps,
        macrotraj=macrotraj,
    )


def hummer15_8state(rate_k, rate_h, nsteps, macrotraj=False):
    """Eight state model inspired by Hummer and Szabo 15.

    Gerhard Hummer and Attila Szabo
    The Journal of Physical Chemistry B 2015 119 (29), 9029-9037
    DOI: [10.1021/jp508375q](https://pubs.acs.org/doi/10.1021/jp508375q)

    Parameters
    ----------
    rate_k : float
        Rate between state 1<->2, 3<->4, 5<->6, 7<->8.
    rate_h : float
        Rate between state 2<->3, 4<->5, 6<->7.
    nsteps : int
        Number of steps to propagate.
    macrotraj : bool, optional
        If `True` return 4-state macrotraj where state (1,2), (3,4), (5, 6),
        and (7,8) are lumped.

    Returns
    -------
    traj : ndarray
        Markov chain Monte Carlo state trajectory.

    """
    return _hummer15_nstate(
        n_states=8,
        rate_k=rate_k,
        rate_h=rate_h,
        nsteps=nsteps,
        macrotraj=macrotraj,
    )

# -*- coding: utf-8 -*-
# BSD 3-Clause License
# Copyright (c) 2019-2023, Daniel Nagel
# All rights reserved.
"""Set of datasets to use for the tutorials."""
import functools

import numpy as np

from msmhelper.msm.msm import row_normalize_matrix
from msmhelper.msm.timescales import _propagate_MCMC
from .tests import is_transition_matrix
from ._utils import shift_data


def _decorate_tmat(tmat):
    """Add tmat as attribute of function."""
    def decorator_tmat(func):
        @functools.wraps(func)
        def wrapper_tmat(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper_tmat.tmat = tmat
        return wrapper_tmat

    return decorator_tmat


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
    cummat = np.cumsum(  # enforce exact normalization
        row_normalize_matrix(tmat),
        axis=1,
    )
    cummat_perm = np.tile(np.arange(n_states), (n_states, 1))

    if start is None:
        start = np.random.randint(n_states)

    return _propagate_MCMC(
        cummat=(cummat, cummat_perm),
        start=start,
        steps=nsteps,
    )


def _hummer15_nstate(
    *, n_states, rate_k, rate_h, nsteps, return_macrotraj=False
):
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
    return_macrotraj : bool, optional
        If `True` return a macrotraj where state (1,2), (3,4), etc. are lumped
        as well.

    Returns
    -------
    traj : ndarray
        Markov chain Monte Carlo state trajectory.
    macrotraj : ndarray
        Markov chain Monte Carlo macrostate trajectory if `macrotraj=True`.

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

    microstates = np.arange(1, n_states + 1)
    microtraj = shift_data(
        propagate_tmat(tmat, nsteps),
        np.arange(n_states),
        microstates,
    )

    if return_macrotraj:
        macrostates = np.tile(
            np.arange(1, n_states // 2 + 1), (2, 1),
        ).T.flatten()
        return (
            microtraj,
            shift_data(
                microtraj,
                microstates,
                macrostates,
            ),
        )
    return microtraj


def hummer15_4state(rate_k, rate_h, nsteps, return_macrotraj=False):
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
    return_macrotraj : bool, optional
        If `True` return a macrotraj where state (1,2) and (3,4) are lumped
        as well.

    Returns
    -------
    traj : ndarray
        Markov chain Monte Carlo state trajectory.
    macrotraj : ndarray
        Markov chain Monte Carlo macrostate trajectory if `macrotraj=True`.

    """
    return _hummer15_nstate(
        n_states=4,
        rate_k=rate_k,
        rate_h=rate_h,
        nsteps=nsteps,
        return_macrotraj=return_macrotraj,
    )


def hummer15_8state(rate_k, rate_h, nsteps, return_macrotraj=False):
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
    return_macrotraj : bool, optional
        If `True` return a macrotraj where state (1,2), (3,4), (5,6), and (7,8)
        are lumped as well.

    Returns
    -------
    traj : ndarray
        Markov chain Monte Carlo state trajectory.
    macrotraj : ndarray
        Markov chain Monte Carlo macrostate trajectory if `macrotraj=True`.

    """
    return _hummer15_nstate(
        n_states=8,
        rate_k=rate_k,
        rate_h=rate_h,
        nsteps=nsteps,
        return_macrotraj=return_macrotraj,
    )


_NAGEL20_4STATE_TMAT = np.array([
    [0.92, 0.04, 0.04, 0.00],
    [0.10, 0.70, 0.10, 0.10],
    [0.10, 0.10, 0.70, 0.10],
    [0.00, 0.04, 0.04, 0.92],
])


@_decorate_tmat(_NAGEL20_4STATE_TMAT)
def nagel20_4state(nsteps):
    """Four state model taken from Nagel et al. 20.

    Daniel Nagel, Anna Weber, and Gerhard Stock
    Journal of Chemical Theory and Computation 2020 16 (12), 7874-7882
    DOI: [10.1021/acs.jctc.0c00774](https://doi.org/10.1021/acs.jctc.0c00774)

    Parameters
    ----------
    nsteps : int
        Number of steps to propagate.

    Returns
    -------
    traj : ndarray
        Markov chain Monte Carlo state trajectory.

    """
    return propagate_tmat(_NAGEL20_4STATE_TMAT, nsteps) + 1


_NAGEL20_6STATE_TMAT = np.array([
    [0.90, 0.03, 0.07, 0.00, 0.00, 0.00],
    [0.05, 0.63, 0.15, 0.07, 0.10, 0.00],
    [0.08, 0.04, 0.60, 0.14, 0.14, 0.00],
    [0.00, 0.12, 0.15, 0.50, 0.15, 0.08],
    [0.00, 0.04, 0.16, 0.10, 0.60, 0.10],
    [0.00, 0.00, 0.00, 0.08, 0.02, 0.90],
])


@_decorate_tmat(_NAGEL20_6STATE_TMAT)
def nagel20_6state(nsteps):
    """Six state model taken from Nagel et al. 20.

    Daniel Nagel, Anna Weber, and Gerhard Stock
    Journal of Chemical Theory and Computation 2020 16 (12), 7874-7882
    DOI: [10.1021/acs.jctc.0c00774](https://doi.org/10.1021/acs.jctc.0c00774)

    Parameters
    ----------
    nsteps : int
        Number of steps to propagate.

    Returns
    -------
    traj : ndarray
        Markov chain Monte Carlo state trajectory.

    """
    return propagate_tmat(_NAGEL20_6STATE_TMAT, nsteps) + 1

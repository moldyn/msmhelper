# -*- coding: utf-8 -*-
"""Create Markov State Model.

This submodule contains all methods related to estimate the Markov state model.

"""
# BSD 3-Clause License
# Copyright (c) 2019-2023, Daniel Nagel
# All rights reserved.

import decorit
import numba
import numpy as np

from msmhelper.utils import tests
from msmhelper.msm.utils import linalg
from msmhelper.statetraj import StateTraj


def estimate_markov_model(trajs, lagtime):
    """Estimates Markov State Model.

    This method estimates the MSM based on the transition count matrix.

    Parameters
    ----------
    trajs : StateTraj or list or ndarray or list of ndarray
        State trajectory/trajectories used to estimate the MSM.
    lagtime : int
        Lag time for estimating the markov model given in [frames].

    Returns
    -------
    T : ndarray
        Transition probability matrix $T_{ij}$, containing the transition
        probability transition from state $i\to j$.
    states : ndarray
        Array holding states corresponding to the columns of $T_{ij}$.

    """
    trajs = StateTraj(trajs)
    return trajs.estimate_markov_model(lagtime)


def _estimate_markov_model(trajs, lagtime, nstates, perm=None):
    """Estimate MSM based on the transition count matrix."""
    # convert trajs to numba list # noqa: SC100
    if not numba.config.DISABLE_JIT:  # pragma: no cover
        trajs = numba.typed.List(trajs)

    if perm is None:
        perm = np.arange(nstates)

    Tcount = _generate_transition_count_matrix(trajs, lagtime, nstates)
    return row_normalize_matrix(Tcount), perm


@numba.njit
def _generate_transition_count_matrix(trajs, lagtime, nstates):
    """Generate a simple transition count matrix from multiple trajectories."""
    # initialize matrix
    T_count = np.zeros((nstates, nstates), dtype=np.int64)

    for traj in trajs:
        for stateFrom, stateTo in zip(traj[:-lagtime], traj[lagtime:]):
            T_count[stateFrom, stateTo] += 1

    return T_count


@numba.njit
def row_normalize_matrix(mat):
    """Row normalize the given 2d matrix.

    Parameters
    ----------
    mat : ndarray
        Matrix to be row normalized.

    Returns
    -------
    mat : ndarray
        Normalized matrix.

    """
    row_sum = np.sum(mat, axis=1)
    if not row_sum.all():
        row_sum[row_sum == 0] = 1

    # due to missing np.newaxis row_sum[:, np.newaxis] becomes # noqa: SC100
    return mat / row_sum.reshape(mat.shape[0], 1)


@decorit.alias('peq')
def equilibrium_population(tmat, allow_non_ergodic=True):
    """Calculate equilibirum population.

    If there are non ergodic states, their population is set to zero.

    Parameters
    ----------
    tmat : ndarray
        Quadratic transition matrix, needs to be ergodic.
    allow_non_ergodic : bool
        If True only the largest ergodic subset will be used. Otherwise it will
        throw an error if not ergodic.

    Returns
    -------
    peq : ndarray
        Equilibrium population of input matrix.

    """
    tmat = np.asarray(tmat)
    is_ergodic = tests.is_ergodic(tmat)
    if not allow_non_ergodic and not is_ergodic:
        raise ValueError('tmat needs to be ergodic transition matrix.')

    # calculate ev for ergodic subset
    if is_ergodic:
        _, eigenvectors = linalg.left_eigenvectors(tmat, nvals=1)
        eigenvectors = eigenvectors[0]
    else:
        mask = tests.ergodic_mask(tmat)
        _, evs_mask = linalg.left_eigenvectors(
            row_normalize_matrix(
                tmat[np.ix_(mask, mask)],
            ),
            nvals=1,
        )

        eigenvectors = np.zeros(len(tmat), dtype=tmat.dtype)
        eigenvectors[mask] = evs_mask[0]

    return eigenvectors / np.sum(eigenvectors)

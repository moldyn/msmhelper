# -*- coding: utf-8 -*-
"""Create Markov State Model.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

Authors: Daniel Nagel
         Georg Diez

"""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numba
import numpy as np
from pyemma import msm as emsm

from msmhelper import tools


# ~~~ FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def build_MSM(trajs, lag_time, reversible=False, **kwargs):
    """Wrapps pyemma.msm.estimate_markov_model.

    Based on the choice of reversibility it either calls pyemma for a
    reversible matrix or it creates a transition count matrix.

    Parameters
    ----------
    trajs : list or ndarray or list of ndarray
        State trajectory/trajectories. The states should start from zero and
        need to be integers.

    lag_time : int
        Lag time for estimating the markov model given in [frames].

    reversible : bool, optional
        If `True` it will uses pyemma.msm.estimate_markov_model which does not
        guarantee that the matrix is of full dimension. In case of `False` or
        if not statedm the local function based on a simple transitition count
        matrix will be used instead.

    kwargs
        For passing values to `pyemma.msm.estimate_markov_model`.

    Returns
    -------
    transmat : ndarray
        Transition rate matrix.

    """
    if reversible:
        MSM = emsm.estimate_markov_model(trajs, lag_time, **kwargs)
        transmat = MSM.transition_matrix
    else:
        transmat = estimate_markov_model(trajs, lag_time)

    return transmat


def estimate_markov_model(trajs, lag_time):
    """Estimates Markov State Model.

    This method estimates the MSM based on the transition count matrix.
    .. todo::
        - allow states to be unequal to indices
        - return active set

    Parameters
    ----------
    trajs : list or ndarray or list of ndarray
        State trajectory/trajectories. The states should start from zero and
        need to be integers.

    lag_time : int
        Lag time for estimating the markov model given in [frames].

    Returns
    -------
    T : ndarray
        Transition rate matrix.

    """
    # format input
    trajs = tools.format_state_traj(trajs)

    # get number of states
    nstates = np.unique(np.concatenate(trajs)).shape[0]

    # convert trajs to numba list
    if not numba.config.DISABLE_JIT:
        trajs = numba.typed.List(trajs)

    T_count = _generate_transition_count_matrix(trajs, lag_time, nstates)
    return _row_normalize_matrix(T_count)


@numba.njit
def _generate_transition_count_matrix(trajs, lag_time, nstates):
    """Generate a simple transition count matrix from multiple trajectories."""
    # initialize matrix
    T_count = np.zeros((nstates, nstates), dtype=np.int64)

    for traj in trajs:
        for stateFrom, stateTo in zip(traj[:-lag_time], traj[lag_time:]):
            T_count[stateFrom, stateTo] += 1


    return T_count


@numba.njit
def _row_normalize_matrix(matrix):
    """Row normalize the given 2d matrix."""
    row_sum = np.sum(matrix, axis=1)
    if not row_sum.all():
        raise ValueError('Row sum of 0 can not be normalized.')

    # due to missing np.newaxis row_sum[:, np.newaxis] becomes
    return matrix / row_sum.reshape(matrix.shape[0], 1)


def left_eigenvectors(matrix):
    """Estimate left eigenvectors.

    Estimates the left eigenvectors and corresponding eigenvalues of a
    quadratic matrix.

    Parameters
    ----------
    matrix : ndarray
        Quadratic 2d matrix eigenvectors and eigenvalues or determined of.

    Returns
    -------
    eigenvalues : ndarray
        N eigenvalues sorted by their value (descending).

    eigenvectors : ndarray
        N eigenvectors sorted by descending eigenvalues.

    """
    matrix = np.asarray(matrix)
    tools._check_quadratic(matrix)  # noqa: WPS437

    # Transpose matrix and therefore determine eigenvalues and left
    # eigenvectors
    matrix_T = np.matrix.transpose(matrix)
    eigenvalues, eigenvectors = np.linalg.eig(matrix_T)

    # Transpose eigenvectors, since v[:,i] is eigenvector
    eigenvectors = eigenvectors.T

    # Sort them by descending eigenvalues
    idx_eigenvalues = eigenvalues.argsort()[::-1]
    eigenvalues_sorted = eigenvalues[idx_eigenvalues]
    eigenvectors_sorted = eigenvectors[idx_eigenvalues]

    return eigenvalues_sorted, eigenvectors_sorted


def _implied_timescales(transmat, lagtime):
    """
    Calculate implied timescales.

    .. todo::
        - Clearify usage. Better passing trajs to calculate matrix?

    Parameters
    ----------
    transmat : ndarray
        Quadratic transition matrix.

    lagtime: int
        Lagtime for estimating the markov model given in [frames].

    Returns
    -------
    timescales: ndarray
        Implied timescale given in frames.

    """
    transmat = np.asarray(transmat)
    tools._check_quadratic(transmat)  # noqa: WPS437

    eigenvalues, eigenvectors = left_eigenvectors(transmat)
    eigenvalues = np.abs(eigenvalues)  # avoid numerical errors
    return - lagtime / np.log(eigenvalues[1:])


def implied_timescales(trajs, lagtimes, reversible=False):
    """Calculate the implied timescales.

    Calculate the implied timescales for the given values.
    .. todo::
        - catch if for higher lagtimes the dimensionality changes

    Parameters
    ----------
    trajs : list or ndarray or list of ndarray
        State trajectory/trajectories. The states should start from zero and
        need to be integers.

    lagtimes : list or ndarray int
        Lagtimes for estimating the markov model given in [frames].

    reversible : bool
        If reversibility should be enforced for the markov state model.

    Returns
    -------
    T : ndarray
        Transition rate matrix.

    """
    # format input
    trajs = tools.format_state_traj(trajs)
    lagtimes = np.atleast_1d(lagtimes)

    # check that lagtimes are array of integers
    if not np.issubdtype(lagtimes.dtype, np.integer):
        raise TypeError(
            'Lagtimes needs to be integers but are {0}'.format(lagtimes.dtype),
        )
    if not (lagtimes > 0).all():
        raise TypeError('Lagtimes needs to be positive integers')

    # initialize result
    nstates = len(np.unique(np.concatenate(trajs)))
    impl_timescales = np.zeros((len(lagtimes), nstates - 1))

    for idx, lagtime in enumerate(lagtimes):
        transmat = build_MSM(trajs, lagtime, reversible=reversible)
        impl_timescales[idx] = _implied_timescales(transmat, lagtime)

    return impl_timescales

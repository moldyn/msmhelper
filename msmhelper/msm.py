"""
Create Markov State Model.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

Authors: Daniel Nagel
         Georg Diez

TODO:
    - create todo

"""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pyemma.msm

from msmhelper import tools


# ~~~ FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def build_MSM(trajs, lag_time, **kwargs):
    """
    Wrapps pyemma.msm.estimate_markov_model.

    Based on the choice of reversibility it either calls pyemma for a
    reversible matrix or it creates a transition count matrix.

    Parameters
    ----------
    trajs : list or ndarray or list of ndarray
        State trajectory/trajectories. The states should start from zero and
        need to be integers.

    lag_time : int
        Lag time for estimating the markov model given in [frames].

    reversible : bool
        If `True` it will uses pyemma.msm.estimate_markov_model which does not
        guarantee that the matrix is of full dimension. In case of `False` or
        if not statedm the local function based on a simple transitition count
        matrix will be used instead.

    See kwargs of both function.

    Returns
    -------
    T : ndarray
        Transition rate matrix.

    """
    if 'reversible' in kwargs and kwargs['reversible']:
        MSM = pyemma.msm.estimate_markov_model(trajs, lag_time, **kwargs)
        T = MSM.transition_matrix
    else:
        if 'reversible' in kwargs:
            del kwargs['reversible']
        T = estimate_markov_model(trajs, lag_time, **kwargs)

    return T


def estimate_markov_model(trajs, lag_time):
    """
    Estimates Markov State Model.

    This method estimates the MSM based on the transition count matrix.

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
    trajs = tools._format_state_trajectory(trajs)

    T_count = _generate_transition_count_matrix(trajs, lag_time)
    T_count_norm = _row_normalize_2d_matrix(T_count)
    return T_count_norm


def _generate_transition_count_matrix(trajs, lag_time: int):
    """Generate a simple transition count matrix from multiple trajectories."""
    # get number of states
    n_states = np.unique(np.concatenate(trajs)).shape[0]
    # initialize matrix
    T_count = np.zeros((n_states, n_states), dtype=int)

    for traj in trajs:
        for i in range(len(traj) - lag_time):  # due to sliding window
            T_count[traj[i], traj[i + lag_time]] += 1

    return T_count


def _row_normalize_2d_matrix(matrix):
    """Row normalize the given 2d matrix."""
    matrix_norm = np.copy(matrix).astype(dtype=np.float64)
    for i, row in enumerate(matrix):
        row_sum = np.sum(row)
        if not row_sum:
            raise ValueError('Row sum of 0 can not be normalized.')
        matrix_norm[i] = matrix_norm[i] / row_sum
    return matrix_norm


def left_eigenvectors(matrix):
    """
    Estimate left eigenvectors.

    Estimates the left eigenvectors and corresponding eigenvalues of a
    quadratic matrix.

    Parameters
    ----------
    matrix : n x n matrix

    Returns
    -------
    eigenvalues: ndarray
        N eigenvalues sorted by their value (descending).

    eigenvectors: ndarray
        N eigenvectors sorted by descending eigenvalues.

    """
    # Check whether matrix is quadratic.
    if np.shape(matrix)[0] != np.shape(matrix)[1]:
        raise ValueError('Matrix is not quadratic.')
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


def implied_timescales(matrix, lagtime):
    """
    Calculate implied timescales.

    Parameters
    ----------
    matrix : n x n matrix
        Transition matrix

    lagtime: int
        lagtime specified in the desired unit


    Returns
    -------
    timescales: ndarray
        N implied timescales in [unit]. The first entry corresponds to the
        stationary distribution.

    """
    eigenvalues, eigenvectors = left_eigenvectors(matrix)
    timescales = - (lagtime / np.log(eigenvalues))

    return timescales

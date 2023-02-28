# -*- coding: utf-8 -*-
# BSD 3-Clause License
# Copyright (c) 2019-2023, Daniel Nagel
# All rights reserved.
"""Set of helpful test functions."""
import decorit
import numpy as np

from msmhelper.utils import _utils
from msmhelper.statetraj import StateTraj


def is_quadratic(matrix):
    """Check if matrix is quadratic.

    Parameters
    ----------
    matrix : ndarray, list of lists
        Matrix which is checked if is 2d array.

    Returns
    -------
    is_quadratic : bool

    """
    # cast to 2d for easier error checking

    matrix = np.atleast_2d(matrix)
    shape = np.shape(matrix)

    # Check whether matrix is quadratic.
    if shape[0] != shape[1]:
        return False
    # check if scalar or tensor higher than 2d
    if shape[0] == 1 or matrix.ndim > 2:
        return False

    return True


def is_state_traj(trajs):
    """Check if state trajectory is correct formatted.

    Parameters
    ----------
    trajs : list of ndarray
        State trajectory/trajectories need to be lists of ndarrays of integers.

    Returns
    -------
    is_state_traj : bool

    """
    try:
        _utils._check_state_traj(trajs)  # noqa: WPS437
    except TypeError:
        return False
    else:
        return True


def is_index_traj(trajs):
    """Check if states can be used as indices.

    Parameters
    ----------
    trajs : list of ndarray
        State trajectory/trajectories need to be lists of ndarrays of integers.

    Returns
    -------
    is_index : bool

    """
    if isinstance(trajs, StateTraj):
        return True
    if is_state_traj(trajs):
        states = _utils.unique(trajs)
        return np.array_equal(states, np.arange(len(states)))
    return False


@decorit.alias('is_tmat')
def is_transition_matrix(matrix, atol=1e-8):
    """Check if transition matrix.

    Rows and cols of zeros (non-visited states) are accepted.

    Parameters
    ----------
    matrix : ndarray
        Transition matrix.
    atol : float, optional
        Absolute tolerance.

    Returns
    -------
    is_tmat : bool

    """
    matrix = np.atleast_2d(matrix)
    visited = np.logical_or(
        matrix.sum(axis=-1),
        matrix.sum(axis=0),
    )
    return (
        is_quadratic(matrix) and
        np.logical_or(
            np.abs(matrix.sum(axis=-1) - 1) <= atol,
            ~visited,
        ).all()
    )


def is_ergodic(matrix, atol=1e-8):
    """Check if matrix is ergodic.

    Parameters
    ----------
    matrix : ndarray
        Transition matrix.
    atol : float, optional
        Absolute tolerance.

    Returns
    -------
    is_ergodic : bool

    References
    ----------
    Wielandt, **Unzerlegbare, Nicht Negativen Matrizen.**
        *Mathematische Zeitschrift* Vol. 52, 1950, pp. 642–648.

    """
    if not is_transition_matrix(matrix):
        return False

    matrix = np.atleast_2d(matrix)

    nstates = len(matrix)
    exponent = (nstates - 1)**2 + 1

    matrix = _utils.matrix_power(matrix, exponent)
    return (matrix > atol).all()


def is_fuzzy_ergodic(matrix, atol=1e-8):
    """Check if matrix is ergodic, up to missing states or trap states.

    If there are two or more disjoint

    Parameters
    ----------
    matrix : ndarray
        Transition matrix.
    atol : float, optional
        Absolute tolerance.

    Returns
    -------
    is_fuzzy_ergodic : bool

    """
    if not is_transition_matrix(matrix):
        return False

    matrix = np.atleast_2d(matrix)
    row_col_sum = matrix.sum(axis=-1) + matrix.sum(axis=0)

    is_trap_state = np.logical_or(
        np.abs(row_col_sum - 2) <= atol,
        np.abs(row_col_sum) <= atol,
    )
    is_trap_state = np.logical_or(
        is_trap_state[:, np.newaxis],
        is_trap_state[np.newaxis, :],
    )

    nstates = len(matrix)
    exponent = (nstates - 1)**2 + 1
    matrix = _utils.matrix_power(matrix, exponent)

    return np.logical_or(matrix > 0, is_trap_state).all()


def ergodic_mask(matrix, atol=1e-8):
    """Create mask for filtering ergodic submatrix.

    Parameters
    ----------
    matrix : ndarray
        Transition matrix.
    atol : float, optional
        Absolute tolerance.

    Returns
    -------
    mask : bool ndarray

    """
    if not is_transition_matrix(matrix):
        raise ValueError("Input matrix needs to be of kind transition matrix.")

    matrix = np.atleast_2d(matrix)
    nstates = len(matrix)
    exponent = (nstates - 1)**2 + 1

    matrix = _utils.matrix_power(matrix, exponent) > atol
    matrix = np.logical_and(matrix, matrix.T)

    # find minimum row counts to identify largest connected block
    maxstates = np.max(matrix.sum(axis=-1))
    return matrix.sum(axis=-1) == maxstates

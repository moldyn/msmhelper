# -*- coding: utf-8 -*-
"""Set of helpful test functions.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

"""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np

from msmhelper import tools
from msmhelper.statetraj import StateTraj
from msmhelper.decorators import shortcut


# ~~~ FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        tools._check_state_traj(trajs)  # noqa: WPS437
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
        states = tools.unique(trajs)
        return np.array_equal(states, np.arange(len(states)))
    return False


@shortcut('is_tmat')
def is_transition_matrix(matrix):
    """Check if transition matrix.

    Parameters
    ----------
    matrix : ndarray
        Transition matrix.

    Returns
    -------
    is_tmat : bool

    """
    matrix = np.atleast_2d(matrix)
    nstates = matrix.shape[0]
    return (
        is_quadratic(matrix) and
        np.array_equal(matrix.sum(axis=-1), np.ones(nstates))
    )


def is_ergodic(matrix):
    """Check if matrix is ergodic.

    Taken from:
    Wielandt, H. "Unzerlegbare, Nicht Negativen Matrizen."
    Mathematische Zeitschrift. Vol. 52, 1950, pp. 642–648.

    Parameters
    ----------
    matrix : ndarray
        Transition matrix.

    Returns
    -------
    is_tmat : bool

    """
    if not is_transition_matrix(matrix):
        return False

    nstates = len(matrix)
    exponent = (nstates - 1)**2 + 1

    matrix = np.linalg.matrix_power(matrix, exponent)
    return (matrix > 0).all()

# -*- coding: utf-8 -*-
"""Set of helpful test functions.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

"""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np

from msmhelper import tools


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
        tools._check_state_traj(trajs)
    except TypeError:
        return False
    else:
        return True

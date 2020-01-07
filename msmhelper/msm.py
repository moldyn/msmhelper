"""
Create Markov State Model.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

Author: Daniel Nagel
        Georg Diez

TODO:
    - create todo

"""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pyemma.msm

from msmhelper import tools


# ~~~ FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def build_MSM(*args, **kwargs):
    """
    Wrapps pyemma.msm.estimate_markov_model.

    Based on the choice of reversibility it either calls pyemma for a
    reversible matrix or it creates a transition count matrix.

    Parameters
    ----------
    reversible : bool
        If `True` it will uses pyemma.msm.estimate_markov_model which does not
        guarantee that the matrix is of full dimension. In case of `False` or
        if not statedm the local function based on a simple transitition count
        matrix will be used instead.

    See args and kwargs of both function.

    Returns
    -------
    T : ndarray
        Transition rate matrix.

    """
    if 'reversible' in kwargs and kwargs['reversible']:
        MSM = pyemma.msm.estimate_markov_model(*args, **kwargs)
        T = MSM.transition_matrix
    else:
        if 'reversible' in kwargs:
            del kwargs['reversible']
        T = estimate_markov_model(*args, **kwargs)

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
        sum = np.sum(row)
        matrix_norm[i] = matrix_norm[i] / sum
    return matrix_norm

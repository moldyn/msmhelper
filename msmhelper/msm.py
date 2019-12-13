"""
Create Markov State Model.

BSD 3-Clause License
Copyright (c) 2019, Daniel Nagel
All rights reserved.

Author: Daniel Nagel
        Georg Diez

TODO:
    - create todo

"""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pyemma.msm as msm

# ~~~ FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def build_MSM(*args, **kwargs):
    """
    Wrapps pyemma.msm.estimate_markov_model.

    Based on the choice of reversibility it either calls pyemma for a
    reversible matrix or it creates a transition count matrix.

    Parameters
    ----------
    See args and kwargs of both function.

    Returns
    -------
    T : ndarray
        Transition rate matrix.

    """
    if 'reversible' in kwargs and kwargs['reversible']:
        MSMrev = msm.estimate_markov_model(*args, **kwargs)
        MSM = MSMrev.transition_matrix
    else:
        if 'reversible' in kwargs:
            del kwargs['reversible']
        MSM = estimate_markov_model(*args, **kwargs)

    return MSM


def estimate_markov_model(trajs, lag_time):
    """
    Estimates Markov State Model.

    This method estimates the MSM based on the transition count matrix.

    Parameters
    ----------
    trajs : ndarray or list of ndarray
        State trajectory/trajectories. The states should start from zero.

    lag_time : int
        Lag time for estimating the markov model given in [frames].

    Returns
    -------
    T : ndarray
        Transition rate matrix.

    """
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
        for i in range(len(traj)-lag_time):  # due to sliding window
            T_count[traj[i], traj[i+lag_time]] += 1

    return T_count


def _row_normalize_2d_matrix(matrix):
    """Row normalize the given 2d matrix."""
    matrix_norm = np.copy(matrix).astype(dtype=np.float64)
    for i, row in enumerate(matrix):
        sum = np.sum(row)
        matrix_norm[i] = matrix_norm[i]/sum
    return matrix_norm

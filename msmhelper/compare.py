# -*- coding: utf-8 -*-
"""Set of helpful functions for comparing markov state models.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

"""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import numba

from msmhelper.statetraj import StateTraj


# ~~~ FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compare_discretization(traj1, traj2, method='symmetric'):
    """Compare similarity of two state discretizations.

    This method compares the similarity of two state discretizations of the
    same dataset. There are two different methods, 'directed' gives a measure
    on how high is the probable to assign a frame correclty knowing the
    `traj1`. Hence splitting a state into many is not penalized, while merging
    multiple into a single state is. Selecting 'symmetric' it is check in both
    directions, so it checks for each state if it is possible to assigned it
    forward or backward. Hence, splitting and merging states is not penalized.

    Parameters
    ----------
    traj1 : StateTraj like
        First state discretization.
    traj2 : StateTraj like
        Second state discretization.
    method : ['symmetric', 'directed']
        Selecting similarity norm. 'symmetric' compares if each frame is
        forward or backward assignable, while 'directed' checks only if it is
        forard assignable.

    Returns
    -------
    similarity : float
        Similarity going from [0, 1], where 1 means identical and 0 no
        similarity at all.

    """
    # format input
    traj1, traj2 = StateTraj(traj1), StateTraj(traj2)
    if method not in {'symmetric', 'directed'}:
        raise ValueError(
            'Only methods "symmetric" and "directed" are supported',
        )

    # check if same length
    if traj1.nframes != traj2.nframes:
        raise ValueError(
            'Trajectories are of different length: ' +
            '{0} vs {1}'.format(traj1.nframes, traj2.nframes),
        )
    return _compare_discretization(traj1, traj2, method)


def _compare_discretization(traj1, traj2, method):
    """Compare similarity of two state discretizations."""
    traj1_flat, traj2_flat = traj1.trajs_flatten, traj2.trajs_flatten

    # get index numbers for each states
    idx_traj1 = {
        state: np.where(traj1_flat == state)[0]
        for state in range(traj1.nstates)
    }
    idx_traj2 = {
        state: np.where(traj2_flat == state)[0]
        for state in range(traj2.nstates)
    }

    intersect12 = _intersect_array(idx_traj1, idx_traj2)
    intersect21 = _intersect_array(idx_traj2, idx_traj1)

    similarity = 0
    for state1, state2 in zip(traj1_flat, traj2_flat):
        # use optimal assignment direction
        if method == 'symmetric':
            similarity += np.max([
                intersect12[state1, state2], intersect21[state2, state1],
            ])
        elif method == 'directed':
            similarity += intersect21[state2, state1]

    # normalize to be in range [0, 1]
    return similarity / traj1.nframes


def _intersect_array(idx1, idx2):
    """Intersect two dictionaries of arrays."""
    return np.frompyfunc(
        lambda state1, state2: _intersect(idx1[state1], idx2[state2]),
        2,  # no. of input arguments
        1,  # no. if output arguments
    ).outer(
        list(idx1.keys()), list(idx2.keys()),
    ).astype(np.float64)


@numba.njit
def _intersect(ar1, ar2):
    """Intersect unique sorted list ar1 with ar2."""
    idx1, idx2 = 0, 0
    len1, len2 = len(ar1), len(ar2)

    count = 0
    while idx1 < len1 and idx2 < len2:
        if ar1[idx1] == ar2[idx2]:
            count += 1
            idx1 += 1
            idx2 += 1
        elif ar1[idx1] > ar2[idx2]:
            idx2 += 1
        else:
            idx1 += 1

    return count / len1

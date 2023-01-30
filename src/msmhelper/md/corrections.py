# -*- coding: utf-8 -*-
"""Dynamical corrections.

BSD 3-Clause License
Copyright (c) 2022, Daniel Nagel
All rights reserved.

Authors: Daniel Nagel

"""
import numba

from msmhelper.statetraj import LumpedStateTraj, StateTraj


class LagtimeError(Exception):
    """An exception for the given lagtime was raised."""


def dynamical_coring(trajs, lagtime, iterative=True):
    """Fix spurious transitions with dynamical coring.

    Projecting high dimensional data onto low dimensional collective variables
    can result in spurious state transitions which can be correct for applying
    dynamical coring, for more details see Nagel et al. [^1].

    !!! note
        Applying dynamical coring on a [msmhelper.LumpedStateTraj][] is not
        supported. The reason is that while applying dynamical coring on the
        microstate level leads to different coarse-graining, applying it on the
        macrostate level the HS-Projection is not well defined anymore.

    [^1]: Nagel et al., **Dynamical coring of Markov state models**,
        *J. Chem. Phys.*, 150, 094111 (2019),
        doi:[10.1063/1.5081767](https://doi.org/10.1063/1.5081767)

    Parameters
    ----------
    trajs : StateTraj or list or ndarray or list of ndarray
        State trajectory/trajectories. The states should start from zero and
        need to be integers.
    lagtime : int
        Lagtime [frames] is the minimum time a trajectory is required to spend
        in a new state to be accepted.
    iterative : bool, optional
        If `True` dynamical coring is applied iteratively with increasing
        lagtimes, so lagtime=1, 2, ..., lagtimes.

    Returns
    -------
    trajs : StateTraj
        Dynamically corrected state trajectory.

    """
    trajs = StateTraj(trajs)

    if isinstance(trajs, LumpedStateTraj):
        raise NotImplementedError(
            'Applying dynamical coring on a LumpedStateTraj is not supported. '
            'The reason is that while applying dynamical coring on the '
            'microstate level leads to different coarse-graining, applying it '
            'on the macrostate level the HS-Projection is not well defined '
            'anymore.'
        )

    # convert trajs to numba list # noqa: SC100
    if numba.config.DISABLE_JIT:
        cored_trajs = trajs.trajs
    else:  # pragma: no cover
        cored_trajs = numba.typed.List(trajs.trajs)

    if lagtime <= 0:
        raise ValueError('The lagtime should be greater 0.')

    # if lagtime == 1 nothing changes
    if lagtime == 1:
        return trajs

    # catch if lagtime <=1
    return StateTraj(
        _dynamical_coring(cored_trajs, lagtime, iterative),
    )


@numba.njit
def _dynamical_coring(trajs, lagtime, iterative):
    """Apply dynamical coring."""
    lagtimes = (
        numba.typed.List(range(2, lagtime + 1))
    ) if iterative else (
        numba.typed.List([lagtime])
    )

    for tau in lagtimes:
        trajs = _dynamical_coring_single_lagtime(
            trajs, tau, iterative,
        )

    return trajs


@numba.njit
def _dynamical_coring_single_lagtime(trajs, lagtime, iterative):
    """Apply dynamical coring."""
    # initialize matrix
    return numba.typed.List([
        _dynamical_coring_single_traj(traj, lagtime, iterative)
        for traj in trajs
    ])


@numba.njit
def _dynamical_coring_single_traj(traj, lagtime, iterative):
    """Apply dynamical coring."""
    # initialize matrix
    core = _find_first_core(traj, lagtime)
    if core == -1:
        raise LagtimeError(
            'For the given lagtime no core can be found. '
            'Try decreasing the lagtime.'
        )

    cored_traj = traj.copy()
    for idx in range(len(traj)):  # noqa: WPS518
        if cored_traj[idx] == core:
            continue

        if _remains_in_core(idx, cored_traj, lagtime, iterative=iterative):
            core = cored_traj[idx]
        else:
            cored_traj[idx] = core
    return cored_traj


@numba.njit
def _remains_in_core(idx, traj, lagtime, iterative):
    """Remains at position idx in core for time lagtime."""
    # if remaining length is shorter than lagtime it cannot remain in the core
    if len(traj) + 1 <= idx + lagtime:
        return False

    # if applying dynamical coring iteratively, we know that every sequence is
    # at least of length lagtime - 1. Hence, it is sufficient to check for the
    # last frame.
    if iterative:
        return traj[idx] == traj[idx + lagtime - 1]
    for idxNext in range(idx + 1, idx + lagtime):
        if traj[idx] != traj[idxNext]:
            return False
    return True


@numba.njit
def _find_first_core(traj, lagtime):
    """Find first core in trajectory."""
    for idx in range(len(traj)):  # noqa: WPS518
        if _remains_in_core(idx, traj, lagtime, iterative=False):
            return traj[idx]
    return -1

# -*- coding: utf-8 -*-
"""Create Markov State Model.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

Authors: Daniel Nagel
         Georg Diez

"""
import decorit
import numba
import numpy as np

from msmhelper import linalg, tests
from msmhelper.statetraj import LumpedStateTraj, StateTraj


def estimate_markov_model(trajs, lagtime):
    """Estimates Markov State Model.

    This method estimates the MSM based on the transition count matrix.

    Parameters
    ----------
    trajs : statetraj or list or ndarray or list of ndarray
        State trajectory/trajectories. The states should start from zero and
        need to be integers.

    lagtime : int
        Lag time for estimating the markov model given in [frames].

    Returns
    -------
    T : ndarray
        Transition rate matrix.

    permutation : ndarray
        Array with corresponding states.

    """
    trajs = StateTraj(trajs)

    if isinstance(trajs, LumpedStateTraj):
        return _estimate_markov_model(
            trajs.trajs,
            lagtime,
            trajs.nmicrostates,
            trajs.microstates,
        )
    return _estimate_markov_model(
        trajs.trajs,
        lagtime,
        trajs.nstates,
        trajs.states,
    )


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


def _implied_timescales(tmat, lagtime, ntimescales):
    """
    Calculate implied timescales.

    .. todo::
        - Clearify usage. Better passing trajs to calculate matrix?

    Parameters
    ----------
    tmat : ndarray
        Quadratic transition matrix.

    lagtime: int
        Lagtime for estimating the markov model given in [frames].

    ntimescales : int, optional
        Number of returned timescales.

    Returns
    -------
    timescales: ndarray
        Implied timescale given in frames.

    """
    tmat = np.asarray(tmat)

    eigenvalues = linalg.left_eigenvalues(tmat, nvals=ntimescales + 1)
    # for negative eigenvalues no timescale is defined
    eigenvalues[eigenvalues < 0] = np.nan
    return np.ma.divide(- lagtime, np.log(eigenvalues[1:]))


def implied_timescales(trajs, lagtimes, ntimescales=None, reversible=False):
    """Calculate the implied timescales.

    Calculate the implied timescales for the given values.
    .. todo:: catch if for higher lagtimes the dimensionality changes

    Parameters
    ----------
    trajs : StateTraj or list or ndarray or list of ndarray
        State trajectory/trajectories. The states should start from zero and
        need to be integers.

    lagtimes : list or ndarray int
        Lagtimes for estimating the markov model given in [frames].
        This is not implemented yet!

    ntimescales : int, optional
        Number of returned lagtimes.

    reversible : bool
        If reversibility should be enforced for the markov state model.

    Returns
    -------
    T : ndarray
        Transition rate matrix.

    """
    # format input
    trajs = StateTraj(trajs)
    lagtimes = np.atleast_1d(lagtimes)

    # check that lag times are array of integers
    if not np.issubdtype(lagtimes.dtype, np.integer):
        raise TypeError(
            'Lagtimes needs to be integers but are {0}'.format(lagtimes.dtype),
        )
    if not (lagtimes > 0).all():
        raise TypeError('Lagtimes needs to be positive integers')
    if reversible:
        raise NotImplementedError(
            'Reversible matrices are not anymore supported.'
        )

    if ntimescales is None:
        ntimescales = trajs.nstates - 1

    # initialize result
    impl_timescales = np.zeros((len(lagtimes), ntimescales))

    for idx, lagtime in enumerate(lagtimes):
        transmat, _ = trajs.estimate_markov_model(lagtime)
        impl_timescales[idx] = _implied_timescales(
            transmat, lagtime, ntimescales=ntimescales,
        )

    return impl_timescales


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

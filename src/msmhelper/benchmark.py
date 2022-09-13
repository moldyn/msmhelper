# -*- coding: utf-8 -*-
"""Benchmark Markov State Model.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

"""
import decorit
import numba
import numpy as np

from msmhelper import linalg, msm, tests, tools
from msmhelper.statetraj import LumpedStateTraj, StateTraj


@decorit.alias('ck_test')
def chapman_kolmogorov_test(trajs, lagtimes, tmax):
    r"""Calculate the Chapman Kolmogorov equation.

    This method estimates the Chapman Kolmogorov equation
    $$T(\tau n) = T^n(\tau)\;.$$
    Projected onto the diagonal this is known as the Chapman Kolmogorov test.
    For more details see, e.g., the review Prinz et al.[^1].

    [^1]: Prinz et al.
        **Markov models of molecular kinetics: Generation and validation**,
        *J. Chem. Phys.*, 134, 174105 (2011),
        doi:[10.1063/1.3565032](https://doi.org/10.1063/1.3565032)

    Parameters
    ----------
    trajs : StateTraj or list or ndarray or list of ndarray
        State trajectory/trajectories. The states should start from zero and
        need to be integers.

    lagtimes : list or ndarray int
        Lagtimes for estimating the markov model given in [frames].

    tmax : int
        Longest time to evaluate the CK equation given in [frames].

    Returns
    -------
    cktest : dict
        Dictionary holding for each lagtime the ckequation and with 'md' the
        reference.

    """
    # format input
    trajs = StateTraj(trajs)
    lagtimes = np.atleast_1d(lagtimes)
    lagtimes = np.sort(lagtimes)

    # check that lag times are array of integers
    if not np.issubdtype(lagtimes.dtype, np.integer):
        raise TypeError(
            'Lagtimes needs to be integers but are {0}'.format(lagtimes.dtype),
        )
    if not (lagtimes > 0).all():
        raise TypeError('Lagtimes needs to be positive integers')

    if lagtimes.ndim != 1:
        raise TypeError(
            'Lagtimes needs to be maximal 1d, but {0}'.format(lagtimes),
        )

    if not isinstance(tmax, int) or tmax < 0:
        raise TypeError('tmax needs to be a positive integer')

    ckeqs = {}
    for lagtime in lagtimes:
        ckeqs[lagtime] = _chapman_kolmogorov_test(trajs, lagtime, tmax)
    ckeqs['md'] = _chapman_kolmogorov_test_md(
        trajs, tmin=lagtimes[0], tmax=tmax,
    )

    return ckeqs


def _chapman_kolmogorov_test(trajs, lagtime, tmax):
    r"""Calculate the Chapman Kolmogorov equation $T^n(\tau)$."""
    times = _calc_times(lagtime=lagtime, tmax=tmax)
    ntimes = len(times)
    ckeq = np.empty((trajs.nstates, ntimes))

    # estimate Markov model
    tmat, _ = trajs.estimate_markov_model(lagtime=lagtime)

    is_ergodic = tests.is_ergodic(tmat)
    is_fuzzy_ergodic = tests.is_fuzzy_ergodic(tmat)
    for idx in range(ntimes):
        tmatpow = tools.matrix_power(tmat, idx + 1)
        ckeq[:, idx] = np.diagonal(tmatpow)

    return {
        'ck': ckeq,
        'time': times,
        'is_ergodic': is_ergodic,
        'is_fuzzy_ergodic': is_fuzzy_ergodic,
    }


def _chapman_kolmogorov_test_md(trajs, tmin, tmax, steps=30):
    r"""Calculate the Chapman Kolmogorov equation $T(n\tau)$."""
    times = np.around(np.geomspace(
        start=tmin,
        stop=tmax,
        num=steps,
    )).astype(np.int64)
    # filter duplicated times (for small ranges)
    times = np.unique(times)
    ntimes = len(times)

    ckeq = np.empty((trajs.nstates, ntimes))
    is_ergodic = np.empty(ntimes, dtype=bool)
    is_fuzzy_ergodic = np.empty(ntimes, dtype=bool)

    # enforce using non-lumped trajectory
    if isinstance(trajs, LumpedStateTraj):
        macrotrajs = StateTraj(trajs.state_trajs)
    else:
        macrotrajs = trajs

    # estimate Markov model
    for idx, time in enumerate(times):
        tmat, _ = macrotrajs.estimate_markov_model(lagtime=time)
        ckeq[:, idx] = np.diagonal(tmat)
        is_ergodic[idx] = tests.is_ergodic(tmat)
        is_fuzzy_ergodic[idx] = tests.is_fuzzy_ergodic(tmat)

    return {
        'ck': ckeq,
        'time': times,
        'is_ergodic': is_ergodic,
        'is_fuzzy_ergodic': is_fuzzy_ergodic,
    }


@decorit.alias('bh_test')
def buchete_hummer_test(trajs, lagtime, tmax):
    r"""Calculate the Buchete Hummer test.

    This method estimates the Buchete Hummer autocorrelation test. Projecting
    the state trajectory onto the right eigenvectors of the row normalized
    transition matrix
    $$C_{lm} (t) = \langle \phi_l[s(\tau +t)] \phi_m[S(\tau)]\rangle$$
    where \(\phi_i\) is the \(i\)-th right eigenvector. Buchete and Hummer[^2]
    showed that for a Markovian system it obeys an exponentil decay,
    corresponds to
    $$C_{lm} (t) = \delta_{lm} \exp(-t / t_k)$$
    with the implied timescale \(t_k = - \tau_\text{lag} / \ln \lambda_k\).

    [^2]: Buchete and Hummer
        **Coarse master equations for peptide folding dynamics**,
        *J. Phys. Chem.*, 112, 6057-6069 (2008),
        doi:[10.1021/jp0761665](https://doi.org/10.1021/jp0761665)

    Parameters
    ----------
    trajs : StateTraj or list or ndarray or list of ndarray
        State trajectory/trajectories. The states should start from zero and
        need to be integers.

    lagtime : int
        Lagtimes for estimating the markov model given in [frames].

    tmax : int
        Longest time to evaluate the CK equation given in [frames].

    Returns
    -------
    bhtest : dict
        Dictionary holding for each lagtime the ckequation and with 'md' the
        reference.

    """
    # format input
    trajs = StateTraj(trajs)

    # check that lag times are array of integers
    if not isinstance(lagtime, int) or lagtime < 0:
        raise TypeError(
            'Lagtimes needs to be positive integers, but {0}.'.format(lagtime),
        )

    if not isinstance(tmax, int) or tmax < 0:
        raise TypeError(
            'tmax needs to be a positive integer, but {0}'.format(tmax),
        )

    return _buchete_hummer_test(trajs, lagtime, tmax)


def _project_states_onto_vector(trajs, vector):
    """Shift states to corresponding vector values.

    Parameters
    ----------
    trajs : StateTraj or ndarray or list or list of ndarrays
        1D data or a list of data.

    vector : ndarray or list
        Values which will be used instead of states.

    Returns
    -------
    array : ndarray
        Shifted data in same shape as input.

    """
    trajs = StateTraj(trajs)
    vector = np.atleast_1d(vector)

    # check data-type
    if len(vector) != trajs.nstates:
        raise TypeError(
            'Vector of wrong length. {0} '.format(len(vector)) +
            'vs. {0}'.format(trajs.nstates),
        )

    # flatten data
    limits = np.cumsum([len(traj) for traj in trajs.trajs])
    array = trajs.trajs_flatten

    # convert to np.array
    states = np.arange(trajs.nstates)

    # shift data
    conv = np.arange(trajs.nstates, dtype=np.float64)
    conv[states] = vector
    array = conv[array]

    # reshape and return
    return np.split(array, limits)[:-1]


def _buchete_hummer_test(trajs, lagtime, tmax):
    r"""Calculate the Buchete Hummer equation."""
    times = _calc_times(lagtime=lagtime, tmax=tmax)
    ntimes = len(times)
    bheq = np.empty((trajs.nstates - 1, ntimes))

    # estimate Markov model
    tmat, _ = msm.estimate_markov_model(trajs, lagtime=lagtime)

    # symmetrize matrix
    tmat = np.sqrt(tmat * tmat.T)

    # get orthogonal eigenvectors of symmetric matrix
    evals, evecs = linalg.right_eigenvectors(tmat)
    pi, *evecs = evecs  # first eigenvector is equilibrium population

    # stationary distribution
    pi = pi**2 / np.sum(pi**2)

    # normalize eigenvectors to be eigenvectors of the non symmetric transition
    # matrix
    evecs = np.array(evecs) / np.sqrt(pi[np.newaxis, :])

    for idx, evec in enumerate(evecs):
        trajs_proj = _project_states_onto_vector(trajs, evec)

        if not numba.config.DISABLE_JIT:  # pragma: no cover
            trajs_proj = numba.typed.List(trajs_proj)

        bheq[idx] = _autocorrelation(trajs_proj, times)

        # for idx_col, time in enumerate(times):
        #     # autocorrelation function
        #     c_nn, norm = 0, 0
        #     for traj_proj in trajs_proj:
        #         c_nn += np.sum(traj_proj[: -time] * traj_proj[time:])
        #         norm += len(traj_proj[time:])
#
        #     bheq[idx, idx_col] = c_nn / norm

    # calculate reference (MD)
    bheq_ref = np.empty((trajs.nstates - 1, ntimes))
    for idx, eigenval in enumerate(evals[1:]):
        bheq_ref[idx] = np.exp(times * np.log(eigenval) / lagtime)

    return {
        lagtime: {'bh': bheq, 'time': times},
        'md': {'bh': bheq_ref, 'time': times},
    }


@numba.njit
def _autocorrelation(trajs, times):
    """Generate a simple transition count matrix from multiple trajectories."""
    # initialize matrix
    bheq = np.zeros(len(times), dtype=np.float64)

    for idx, time in enumerate(times):
        # autocorrelation function
        c_nn, norm = 0, 0
        for traj in trajs:
            for val_from, val_to in zip(  # noqa: WPS519
                traj[: -time], traj[time:],
            ):
                c_nn += val_from * val_to
            norm += len(traj[time:])

        bheq[idx] = c_nn / norm

    return bheq


def _calc_times(lagtime, tmax):
    """Return time array."""
    steps = int(np.floor(tmax / lagtime))
    return lagtime * np.arange(1, steps + 1)

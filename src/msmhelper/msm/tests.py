# -*- coding: utf-8 -*-
"""Benchmark Markov State Model.

This submodule holds methods for validating Markov state models.

"""
# BSD 3-Clause License
# Copyright (c) 2019-2023, Daniel Nagel
# All rights reserved.
import decorit
import numpy as np

from msmhelper import utils
from msmhelper.statetraj import LumpedStateTraj, StateTraj


@decorit.alias('ck_test')
def chapman_kolmogorov_test(trajs, lagtimes, tmax):
    r"""Calculate the Chapman-Kolmogorov equation.

    This method evaluates both sides of the Chapman-Kolmogorov equation

    $$T(\tau n) = T^n(\tau)\;.$$

    So to compare the transition probability estimated based on the lag time
    $n\tau$ (referred as "MD") with the transition probability estimated based
    on the lag time $\tau$ and propagated $n$ times (referred as "MSM"), we can
    use the Chapman-Kolmogorov test. If the model is Markovian, both sides are
    identical, and the deviation indicates how Markovian the model is. The
    Chapman-Kolmogorov test is commonly projected onto the diagonal (so
    limiting to $T_{ii}$). For more details, see the review by Prinz et al.
    [^1].

    The returned dictionary can be visualized using
    [msmhelper.plot.plot_ck_test][]. An example can be found in the
    [tutorial](/msmhelper/tutorials/msm/#chapman-kolmogorov-test).

    [^1]: Prinz et al., **Markov models of molecular kinetics: Generation and
        validation**, *J. Chem. Phys.*, 134, 174105 (2011),
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
        Dictionary holding for each lagtime the CK equation and with 'md' the
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

    is_ergodic = utils.tests.is_ergodic(tmat)
    is_fuzzy_ergodic = utils.tests.is_fuzzy_ergodic(tmat)
    for idx in range(ntimes):
        tmatpow = utils.matrix_power(tmat, idx + 1)
        ckeq[:, idx] = np.diagonal(tmatpow)

    return {
        'ck': {
            state: ckeq[idx]
            for idx, state in enumerate(trajs.states)
        },
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
        macrotrajs = StateTraj(trajs.trajs)
    else:
        macrotrajs = trajs

    # estimate Markov model
    for idx, time in enumerate(times):
        tmat, _ = macrotrajs.estimate_markov_model(lagtime=time)
        ckeq[:, idx] = np.diagonal(tmat)
        is_ergodic[idx] = utils.tests.is_ergodic(tmat)
        is_fuzzy_ergodic[idx] = utils.tests.is_fuzzy_ergodic(tmat)

    return {
        'ck': {
            state: ckeq[idx]
            for idx, state in enumerate(trajs.states)
        },
        'time': times,
        'is_ergodic': is_ergodic,
        'is_fuzzy_ergodic': is_fuzzy_ergodic,
    }


def _calc_times(lagtime, tmax):
    """Return time array."""
    steps = int(np.floor(tmax / lagtime))
    return lagtime * np.arange(1, steps + 1)

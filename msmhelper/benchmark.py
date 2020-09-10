# -*- coding: utf-8 -*-
"""Benchmark Markov State Model.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

Authors: Daniel Nagel

"""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np

from msmhelper import msm, tests, tools
from msmhelper.decorators import shortcut
from msmhelper.statetraj import StateTraj


# ~~~ FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@shortcut('ck_test')
def chapman_kolmogorov_test(trajs, lagtimes, tmax):
    """Calculate the Chapman Kolmogorov equation.

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

    # allocate memory
    ckeqs = {}
    for lagtime in lagtimes:
        ckeqs[lagtime] = _chapman_kolmogorov_test(trajs, lagtime, tmax)
    ckeqs['md'] = _chapman_kolmogorov_test_md(trajs, lagtimes[0], tmax)

    return ckeqs


def _chapman_kolmogorov_test(trajs, lagtime, tmax):
    r"""Calculate the Chapman Kolmogorov equation $T^n(\tau)$."""
    steps = int(np.floor(tmax / lagtime))
    times = lagtime * np.arange(1, steps + 1)
    ntimes = len(times)
    ckeq = np.empty((trajs.nstates, ntimes))

    # estimate Markov model
    tmat, _ = msm.estimate_markov_model(trajs, lagtime=lagtime)

    is_ergodic = tests.is_ergodic(tmat)
    for idx in range(ntimes):
        tmatpow = tools.matrix_power(tmat, idx + 1)
        ckeq[:, idx] = np.diagonal(tmatpow)

    return {'ck': ckeq, 'time': times, 'is_ergodic': is_ergodic}


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

    # estimate Markov model
    for idx, time in enumerate(times):
        tmat, _ = msm.estimate_markov_model(trajs, lagtime=time)
        ckeq[:, idx] = np.diagonal(tmat)
        is_ergodic[idx] = tests.is_ergodic(tmat)

    return {'ck': ckeq, 'time': times, 'is_ergodic': is_ergodic}

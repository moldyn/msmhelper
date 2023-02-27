# -*- coding: utf-8 -*-
# BSD 3-Clause License
# Copyright (c) 2019-2023, Daniel Nagel
# All rights reserved.
"""Plot the waiting time distribution."""
import msmhelper as mh
import numpy as np
import prettypyplot as pplt
from matplotlib import pyplot as plt


def _estimate_stats(coord):
    """Return boxplot stats of data."""
    Q1, Q2, Q3 = np.quantile(
        coord,
        [0.25, 0.5, 0.75],
    )
    IQR = Q3 - Q1
    return (
        coord.min(),
        max(Q1 - IQR, coord.min()),
        Q1,
        Q2,
        Q3,
        min(Q3 + IQR, coord.max()),
        coord.max()
    )


def plot_wtd(
    trajs,
    max_lagtime,
    start,
    final,
    steps,
    n_lagtimes=50,
    frames_per_unit=1,
    unit='frames',
    ax=None,
    show_md=True,
    show_fliers=False,
):
    """Plot waiting time distribution.

    Parameters
    ----------
    trajs : statetraj or list or ndarray or list of ndarray
        State trajectory/trajectories. The states should start from zero and
        need to be integers.
    max_lagtime : int
        Maximal lag time for estimating the markov model given in [frames].
    start : int or list of
        States to start counting.
    final : int or list of
        States to start counting.
    steps : int
        Number of MCMC propagation steps of MCMC run.
    frames_per_unit : float, optional
        Number of frames per given unit. This is used to scale the axis
        accordingly.
    unit : ['frames', 'fs', 'ps', 'ns', 'us'], optional
        Unit to use for label.
    ax : matplotlib.Axes, optional
        Axes to plot figure in. With `None` the current axes is used.
    show_md : bool, optional
        Include boxplot of MD data.
    show_fliers : bool, optional
        Show fliers (outliers) in MD and MSM prediction.

    Returns
    -------
    ax : matplotlib.Axes
        Return axes holding the plot.

    """
    if ax is None:
        ax = plt.gca()

    lagtimes = np.unique(
        np.linspace(1, max_lagtime, num=n_lagtimes, dtype=int),
    )

    # get stats
    FL, LB, Q1, Q2, Q3, UB, FU = (
        np.array([
            _estimate_stats(
                mh.msm.estimate_waiting_times(
                    trajs=trajs,
                    lagtime=lagtime,
                    start=start,
                    final=final,
                    steps=steps,
                    return_list=True,
                )
            )
            for lagtime in lagtimes
        ]) / frames_per_unit
    ).T

    # plot results
    colors = pplt.categorical_color(4, 'C0')
    lagtimes = lagtimes / frames_per_unit
    if show_fliers:
        ax.fill_between(
            lagtimes, FL, FU, color=colors[3], label=r'fliers',
        )
    ax.fill_between(
        lagtimes, LB, UB, color=colors[2], label=r'$Q_{1/3}\pm\mathrm{IQR}$',
    )
    ax.fill_between(lagtimes, Q1, Q3, color=colors[1], label='IQR')
    ax.plot(lagtimes, Q2, color=colors[0], label='$Q_2$')

    max_lagtime_unit = max_lagtime / frames_per_unit
    if show_md:
        wt_md = mh.md.estimate_waiting_times(
            trajs=trajs,
            start=start,
            final=final,
        )
        bxp = ax.boxplot(
            wt_md / frames_per_unit,
            positions=[max_lagtime_unit * 1.25],
            widths=max_lagtime_unit * 0.075,
            showfliers=show_fliers,
        )
        for median in bxp['medians']:
            median.set_color('k')

        ax.axvline(
            max_lagtime_unit * 1.125,
            0,
            1,
            lw=plt.rcParams['axes.linewidth'],
            color='pplt:axes',
        )

    if show_md:
        ax.set_xlim([0, max_lagtime_unit * 1.375])
        xticks = np.array([
            *np.linspace(0, max_lagtime_unit, 4).astype(int),
            max_lagtime_unit * 1.25,
        ])
        xticklabels = [
            f'{xtick:.0f}' if idx + 1 < len(xticks) else 'MD'
            for idx, xtick in enumerate(xticks)
        ]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
    else:
        ax.set_xlim([0, max_lagtime_unit])

    # set legend and labels
    pplt.legend(ax=ax, outside='top')
    ax.set_ylabel(f'time $t$ [{unit}]')
    ax.set_xlabel(fr'$\tau_\mathrm{{lag}}$ [{unit}]')

    return ax

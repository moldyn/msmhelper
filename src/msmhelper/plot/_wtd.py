# -*- coding: utf-8 -*-
# BSD 3-Clause License
# Copyright (c) 2019-2023, Daniel Nagel
# All rights reserved.
"""Plot the waiting time distribution."""
import numpy as np
import prettypyplot as pplt
from matplotlib import pyplot as plt


def plot_wtd(
    wtd,
    frames_per_unit=1,
    unit='frames',
    ax=None,
    show_md=True,
    show_fliers=False,
):
    """Plot waiting time distribution.

    This is a wrapper function to plot the return value of
    [msmhelper.msm.estimate_waiting_time_dist][].

    Parameters
    ----------
    wtd : dict
        Dictionary returned from `msmhelper.msm.estimate_wtd`, holding stats of
        waiting time distributions.
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

    lagtimes = np.array(
        [time for time in wtd.keys() if time != 'MD'], dtype=int,
    )
    max_lagtime = lagtimes.max()

    # convert stats to array
    LB, UB, Q1, Q2, Q3 = np.array([
        np.array([
            wtd[lagtime][key] for lagtime in lagtimes
        ]) / frames_per_unit
        for key in ['whislo', 'whishi', 'q1', 'med', 'q3']
    ])
    FL = np.array([
        min(
            np.min(wtd[lagtime]['fliers']),
            wtd[lagtime]['whislo'],
        ) for lagtime in lagtimes
    ]) / frames_per_unit
    FU = np.array([
        max(
            np.max(wtd[lagtime]['fliers']),
            wtd[lagtime]['whishi'],
        ) for lagtime in lagtimes
    ]) / frames_per_unit

    # plot results
    colors = pplt.categorical_color(4, 'C0')
    lagtimes = lagtimes / frames_per_unit
    if show_fliers:
        ax.fill_between(
            lagtimes, FL, FU, color=colors[3], label=r'fliers',
        )
    ax.fill_between(
        lagtimes,
        LB,
        UB,
        color=colors[2],
        label=r'$Q_{1/3}\pm1.5\mathrm{IQR}$',
    )
    ax.fill_between(lagtimes, Q1, Q3, color=colors[1], label='IQR')
    ax.plot(lagtimes, Q2, color=colors[0], label='$Q_2$')

    max_lagtime_unit = max_lagtime / frames_per_unit
    if show_md:
        bxp = ax.bxp(
            [{
                key: time / frames_per_unit
                for key, time in wtd['MD'][0].items()
            }],
            positions=[max_lagtime_unit * 1.125],
            widths=max_lagtime_unit * 0.075,
            showfliers=show_fliers,
        )
        for median in bxp['medians']:
            median.set_color('k')

        ax.axvline(
            max_lagtime_unit,
            0,
            1,
            lw=plt.rcParams['axes.linewidth'],
            color='pplt:axes',
        )

    if show_md:
        ax.set_xlim([0, max_lagtime_unit * 1.25])
        xticks = np.array([
            *np.linspace(0, max_lagtime_unit, 4).astype(int),
            max_lagtime_unit * 1.125,
        ])
        xticklabels = [
            f'{xtick:.0f}' if idx + 1 < len(xticks) else 'MD'
            for idx, xtick in enumerate(xticks)
        ]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
    else:
        ax.set_xlim([0, max_lagtime_unit])

    # use scientific notation for large values
    ax.ticklabel_format(
        axis='y', style='scientific', scilimits=[0, 2], useMathText=True,
    )
    ax.get_yaxis().get_offset_text().set_ha('right')

    # set legend and labels
    pplt.legend(ax=ax, outside='top', frameon=False)
    ax.set_ylabel(f'time $t$ [{unit}]')
    ax.set_xlabel(fr'$\tau_\mathrm{{lag}}$ [{unit}]')

    return ax

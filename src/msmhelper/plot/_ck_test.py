# -*- coding: utf-8 -*-
# BSD 3-Clause License
# Copyright (c) 2019-2023, Daniel Nagel
# All rights reserved.
"""Plot the Chapman Kolmogorov test."""
import numpy as np
import prettypyplot as pplt
from matplotlib import pyplot as plt


def plot_ck_test(
    ck,
    states=None,
    frames_per_unit=1,
    unit='frames',
    grid=(3, 3),
):
    """Plot CK-Test results.

    This routine is a basic helper function to visualize the results of
    [msmhelper.msm.chapman_kolmogorov_test][].

    Parameters
    ----------
    ck : dict
        Dictionary holding for each lagtime the CK equation and with 'md' the
        reference.
    states : ndarray, optional
        List containing all states to plot the CK-test.
    frames_per_unit : float, optional
        Number of frames per given unit. This is used to scale the axis
        accordingly.
    unit : ['frames', 'fs', 'ps', 'ns', 'us'], optional
        Unit to use for label.
    grid : (int, int), optional
        The number of `(n_rows, n_cols)` to use for the grid layout.

    Returns
    -------
    fig : matplotlib.Figure
        Figure holding plots.

    """
    # load colors
    pplt.load_cmaps()
    pplt.load_colors()

    lagtimes = np.array([key for key in ck.keys() if key != 'md'])
    if states is None:
        states = np.array(
            list(ck['md']['ck'].keys())
        )

    nrows, ncols = grid
    needed_rows = int(np.ceil(len(states) / ncols))

    fig, axs = plt.subplots(
        needed_rows,
        ncols,
        sharex=True,
        sharey='row',
        gridspec_kw={'wspace': 0, 'hspace': 0},
    )
    axs = np.atleast_2d(axs)

    max_time = np.max(ck['md']['time'])
    for irow, states_row in enumerate(_split_array(states, ncols)):
        for icol, state in enumerate(states_row):
            ax = axs[irow, icol]

            pplt.plot(
                ck['md']['time'] / frames_per_unit,
                ck['md']['ck'][state],
                '--',
                ax=ax,
                color='pplt:gray',
                label='MD',
            )
            for lagtime in lagtimes:
                pplt.plot(
                    ck[lagtime]['time'] / frames_per_unit,
                    ck[lagtime]['ck'][state],
                    ax=ax,
                    label=lagtime / frames_per_unit,
                )
            pplt.text(
                0.5,
                0.9,
                'S{0}'.format(state),
                contour=True,
                va='top',
                transform=ax.transAxes,
                ax=ax,
            )

            # set scale
            ax.set_xscale('log')
            ax.set_xlim([
                lagtimes[0] / frames_per_unit,
                max_time / frames_per_unit,
            ])
            ax.set_ylim([0, 1])
            if irow < len(axs) - 1:
                ax.set_yticks([0.5, 1])
            else:
                ax.set_yticks([0, 0.5, 1])

            ax.grid(True, which='major', linestyle='--')
            ax.grid(True, which='minor', linestyle='dotted')
            ax.set_axisbelow(True)

    # set legend
    legend_kw = {
        'outside': 'right',
        'bbox_to_anchor': (2.0, (1 - nrows), 0.2, nrows),
    } if ncols in {1, 2} else {
        'outside': 'top',
        'bbox_to_anchor': (0.0, 1.0, ncols, 0.01),
    }
    if ncols == 3:
        legend_kw['ncol'] = 3
    pplt.legend(
        ax=axs[0, 0],
        **legend_kw,
        title=fr'$\tau_\mathrm{{lag}}$ [{unit}]',
        frameon=False,
    )

    ylabel = (
        r'self-transition probability $P_{i\to i}$'
    ) if nrows >= 3 else (
        r'$P_{i\to i}$'
    )

    pplt.hide_empty_axes()
    pplt.label_outer()
    pplt.subplot_labels(
        ylabel=ylabel,
        xlabel=r'time $t$ [{unit}]'.format(unit=unit),
    )
    return fig


def _split_array(array, chunksize):
    """Split numpy array in maximal sizes of chunksize."""
    split_limits = [
        chunksize * i for i in range(1, int(len(array) / chunksize) + 1)
    ]
    split = np.split(array, split_limits)
    if not len(split[-1]):
        split = split[:-1]
    return split

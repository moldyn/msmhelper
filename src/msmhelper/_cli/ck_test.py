# -*- coding: utf-8 -*-
# BSD 3-Clause License
# Copyright (c) 2019-2023, Daniel Nagel
# All rights reserved.
"""Plot the Chapman Kolmogorov test."""
import click
import msmhelper as mh
import numpy as np
import prettypyplot as pplt
from matplotlib import pyplot as plt


@click.command(no_args_is_help='-h')
@click.option(
    '--filename',
    '-f',
    required=True,
    type=click.Path(exists=True),
    help='Path to microstate trajectory file (single column ascii file).',
)
@click.option(
    '--macrofilename',
    required=False,
    type=click.Path(exists=True),
    help='Path to macrostate trajectory file (single column ascii file).',
)
@click.option(
    '--concat-limits',
    '-c',
    required=False,
    type=click.Path(exists=True),
    help=(
        'Path to concatination limit file given the length of all ' +
        r'trajectories, e.g. "3\n3\n5"'
    ),
)
@click.option(
    '--lagtimes',
    required=True,
    nargs=5,
    type=click.IntRange(min=1),
    help='5 (!) Lag times given in frames to estimate Markov state model.',
)
@click.option(
    '--frames-per-unit',
    required=True,
    type=click.FLOAT,
    help='Number of frames per unit.',
)
@click.option(
    '--unit',
    required=True,
    type=click.Choice(
        ['fs', 'ps', 'ns', 'us', 'frames'],
        case_sensitive=False,
    ),
    help='Unit of data.',
)
@click.option(
    '--grid',
    type=click.IntRange(min=1),
    nargs=2,
    default=(1, 1),
    help='Number of rows and cols.',
)
@click.option(
    '--max-time',
    type=click.IntRange(min=1),
    default=int(10**4),
    help='Largest time value to evaluate and plot the test.',
)
def ck_test(
    filename,
    macrofilename,
    concat_limits,
    lagtimes,
    frames_per_unit,
    unit,
    grid,
    max_time,
):
    """Calculate and plot CK test."""
    # setup matplotlib
    pplt.use_style(figsize=0.8, true_black=True, colors='pastel_autunm')

    # load file
    trajs = mh.openmicrostates(filename, limits_file=concat_limits)
    if macrofilename:
        macrotrajs = mh.openmicrostates(
            macrofilename, limits_file=concat_limits,
        )
        trajs = mh.LumpedStateTraj(macrotrajs, trajs)
    else:
        trajs = mh.StateTraj(trajs)

    # perform test
    ck = mh.msm.ck_test(trajs, lagtimes=lagtimes, tmax=max_time)

    # plot result
    nrows, ncols = grid
    for nfig, chunk in enumerate(_split_array(trajs.states, nrows * ncols)):
        plot_ck_test(
            ck=ck,
            states=chunk,
            lagtimes=lagtimes,
            frames_per_unit=frames_per_unit,
            unit=unit,
            grid=grid,
        )

        # save figure and continue
        output = filename
        if macrofilename:
            output = '{f}.sh'.format(f=macrofilename)
        pplt.savefig(
            '{f}.cktest.state{start:.0f}-{to:.0f}.pdf'.format(
                f=output,
                start=chunk[0],
                to=chunk[-1],
            )
        )
        plt.close()


def plot_ck_test(
    ck,
    states,
    lagtimes,
    frames_per_unit,
    unit,
    grid,
):
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
                color='pplt:axes',
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

            pplt.grid(ax=ax)

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
        title=fr'$\tau_\text{{lag}}$ [{unit}]',
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


if __name__ == '__main__':
    ck_test()

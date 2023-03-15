# -*- coding: utf-8 -*-
# BSD 3-Clause License
# Copyright (c) 2019-2023, Daniel Nagel
# All rights reserved.
"""Plot the waiting time distribution."""
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
    help='Path to state trajectory file (single column ascii file).',
)
@click.option(
    '--microfilename',
    required=False,
    type=click.Path(exists=True),
    help=(
        'Path to microstate trajectory file (single column ascii file) to use '
        'Hummer-Szabo projection.'
    ),
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
    '--output',
    '-o',
    type=click.Path(),
    help=(
        'Output name of figure. Needs to have a valid suffix (".pdf", ".svg", '
        '".png"). Default format is pdf.'
    ),
)
@click.option(
    '--lagtimes',
    required=True,
    nargs=3,
    type=click.IntRange(min=1),
    help='3 (!) Lag times given in frames to estimate Markov state model.',
)
@click.option(
    '--start',
    required=True,
    type=click.IntRange(min=1),
    help='State to start from.',
)
@click.option(
    '--final',
    required=True,
    type=click.IntRange(min=1),
    help='State to end in.',
)
@click.option(
    '--nsteps',
    required=True,
    type=click.IntRange(min=1),
    default=int(1e8),
    help='State to end in.',
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
def waiting_times(
    filename,
    microfilename,
    concat_limits,
    output,
    lagtimes,
    start,
    final,
    nsteps,
    frames_per_unit,
    unit,
):
    """Estimation and visualization of the waiting times."""
    # setup matplotlib
    pplt.use_style(
        figsize=2.2, true_black=True, colors='pastel_autunm', latex=False,
    )

    # load file
    trajs = mh.openmicrostates(filename, limits_file=concat_limits)
    if microfilename:
        microtrajs = mh.openmicrostates(
            microfilename, limits_file=concat_limits,
        )
        trajs = mh.LumpedStateTraj(trajs, microtrajs, positive=True)
    else:
        trajs = mh.StateTraj(trajs)

    # estimate wts
    wts = {
        lagtime: mh.msm.estimate_waiting_times(
            trajs=trajs,
            lagtime=lagtime,
            start=start,
            final=final,
            steps=nsteps,
        )
        for lagtime in lagtimes
    }
    wts_md = mh.md.estimate_waiting_times(
        trajs=trajs,
        start=start,
        final=final,
    )

    _, ax = plt.subplots()

    # ensure using bins as multiple of frames
    n_bins = 20
    bins = np.arange(
        0,
        wts_md.max() + 1,
        np.ceil(wts_md.max() / n_bins).astype(int),
    )
    md_hist, md_time = np.histogram(wts_md, bins=bins, density=True)
    ax.stairs(
        md_hist,
        md_time / frames_per_unit,
        fill=True,
        color='k',
        label='MD',
    )
    for idx, lagtime in enumerate(lagtimes):
        ax.stairs(
            wts[lagtime][0],
            wts[lagtime][1] / frames_per_unit,
            label=f'{lagtime / frames_per_unit}',
            color=f'C{idx + 1}',
        )

    # set legend and labels
    pplt.legend(
        ax=ax,
        outside='right',
        frameon=False,
        title=fr'$\tau_\mathrm{{lag}}$ [{unit}]',
    )
    ax.set_ylabel('probability $P$')
    ax.set_xlabel(fr'$\tau_\mathrm{{lag}}$ [{unit}]')

    # use scientific notation for small values
    ax.ticklabel_format(
        axis='y', style='scientific', scilimits=[-1, 1], useMathText=True,
    )
    ax.get_yaxis().get_offset_text().set_ha('right')

    # set x limits
    ax.set_xlim(
        0,
        wts_md.max() * 2 / frames_per_unit,
    )

    if output is None:
        basename = f'{filename}.sh' if microfilename else filename
        output = f'{basename}.wts.pdf'
    pplt.savefig(output)


if __name__ == '__main__':
    waiting_times()  # pragma: no cover

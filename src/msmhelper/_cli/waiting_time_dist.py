# -*- coding: utf-8 -*-
# BSD 3-Clause License
# Copyright (c) 2019-2023, Daniel Nagel
# All rights reserved.
"""Plot the waiting time distribution."""
import click
import msmhelper as mh
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
    '--max-lagtime',
    required=True,
    type=click.IntRange(min=1),
    help='Maximal lag time given in frames to estimate Markov state model.',
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
def waiting_time_dist(
    filename,
    microfilename,
    concat_limits,
    output,
    max_lagtime,
    start,
    final,
    nsteps,
    frames_per_unit,
    unit,
):
    """Estimation and visualization of the waiting time distributions."""
    # setup matplotlib
    pplt.use_style(
        figsize=2.4, true_black=True, colors='pastel_autunm', latex=False,
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

    # perform test
    wtd = mh.msm.estimate_wtd(
        trajs, max_lagtime=max_lagtime, start=start, final=final, steps=nsteps,
    )

    _, ax = plt.subplots()
    mh.plot.plot_wtd(
        wtd,
        ax=ax,
        frames_per_unit=frames_per_unit,
        unit=unit,
    )

    if output is None:
        basename = f'{filename}.sh' if microfilename else filename
        output = f'{basename}.wtd.pdf'
    pplt.savefig(output)


if __name__ == '__main__':
    waiting_time_dist()  # pragma: no cover

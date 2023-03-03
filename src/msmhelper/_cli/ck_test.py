# -*- coding: utf-8 -*-
# BSD 3-Clause License
# Copyright (c) 2019-2023, Daniel Nagel
# All rights reserved.
"""Plot the Chapman-Kolmogorov test."""
from os.path import splitext

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
        'Output basename of figure. Needs to have a valid extension (".pdf", '
        '".svg", ".png"). Default format is pdf.'
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
    microfilename,
    concat_limits,
    output,
    lagtimes,
    frames_per_unit,
    unit,
    grid,
    max_time,
):
    """Estimation and visualization of the Chapman-Kolmogorov test."""
    # setup matplotlib
    pplt.use_style(
        figsize=0.8, true_black=True, colors='pastel_autunm', latex=False,
    )

    # load file
    trajs = mh.openmicrostates(filename, limits_file=concat_limits)
    if microfilename:
        microtrajs = mh.openmicrostates(
            microfilename, limits_file=concat_limits,
        )
        trajs = mh.LumpedStateTraj(trajs, microtrajs)
    else:
        trajs = mh.StateTraj(trajs)

    # perform test
    ck = mh.msm.ck_test(trajs, lagtimes=lagtimes, tmax=max_time)

    # plot result
    nrows, ncols = grid
    for chunk in mh.plot._ck_test._split_array(trajs.states, nrows * ncols):
        mh.plot.plot_ck_test(
            ck=ck,
            states=chunk,
            frames_per_unit=frames_per_unit,
            unit=unit,
            grid=grid,
        )

        # save figure and continue
        if output is None:
            basename = f'{filename}.sh' if microfilename else filename
            output = f'{basename}.cktest.pdf'
        # insert state_str between pathname and extension
        path, ext = splitext(output)
        pplt.savefig(f'{path}.state{chunk[0]:.0f}-{chunk[-1]:.0f}{ext}')
        plt.close()


if __name__ == '__main__':
    ck_test()  # pragma: no cover

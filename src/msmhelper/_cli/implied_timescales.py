# -*- coding: utf-8 -*-
# BSD 3-Clause License
# Copyright (c) 2019-2023, Daniel Nagel
# All rights reserved.
"""Plot the implied timescales."""
import click
import matplotlib.pyplot as plt
import numpy as np
import msmhelper as mh
import prettypyplot as pplt


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
    type=click.IntRange(min=0),
    help='Maximal lag time to estimate Markov state model in frames.',
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
    '--n-lagtimes',
    type=click.IntRange(min=1),
    default=3,
    help='Number of lagtimes to plot.',
)
@click.option(
    '--ylog',
    is_flag=True,
    help='Use logarithmic y-axis.',
)
def implied_timescales(
    filename,
    microfilename,
    concat_limits,
    output,
    max_lagtime,
    frames_per_unit,
    unit,
    n_lagtimes,
    ylog,
):
    """Estimation and visualization of the implied timescales."""
    # load file
    trajs = mh.openmicrostates(filename, limits_file=concat_limits)
    if microfilename:
        microtrajs = mh.openmicrostates(
            microfilename, limits_file=concat_limits,
        )
        trajs = mh.LumpedStateTraj(trajs, microtrajs)
    else:
        trajs = mh.StateTraj(trajs)

    # calculate implied timescales
    lagtimes = np.unique(np.linspace(1, max_lagtime, num=50, dtype=int))
    impl_times = mh.msm.implied_timescales(trajs, lagtimes=lagtimes)
    n_lagtimes = np.min((n_lagtimes, len(impl_times)))

    # setup pyplot
    kwargs = {'colors': 'pastel_autunm'}
    if n_lagtimes > 5:
        kwargs = {'colors': 'macaw', 'ncs': n_lagtimes + 1}

    pplt.use_style(figsize=2.2, **kwargs, latex=False, true_black=True)

    # plot result
    fig, ax = plt.subplots()

    for idx in range(n_lagtimes):
        pplt.plot(
            lagtimes / frames_per_unit,
            impl_times[:, idx] / frames_per_unit,
            ax=ax,
            label=idx + 1,
        )

    # set scale
    ax.set_xlim([0, max_lagtime / frames_per_unit])
    if not ylog:
        ax.set_ylim([0, ax.get_ylim()[1]])

    ax.grid(True, which='major', linestyle='--')
    ax.grid(True, which='minor', linestyle='dotted')
    ax.set_axisbelow(True)

    # show diagonal
    ax.fill_between(ax.get_xlim(), ax.get_xlim(), color='pplt:grid', zorder=0)

    # set label
    if unit == 'us':
        unit = r'\textmu{{}}s'
    ax.set_ylabel(r'time scales [{unit}]'.format(unit=unit))
    ax.set_xlabel(r'$\tau _\mathrm{{lag}}$ [{unit}]'.format(unit=unit))

    pplt.legend(
        ax=ax,
        outside='top',
        title='Implied time scale:',
    )

    if ylog:
        ax.set_yscale('log')

    if output is None:
        basename = f'{filename}.sh' if microfilename else filename
        output = f'{basename}.impltime.tmax{max_lagtime:.0f}.pdf'

    # save figure
    pplt.savefig(output)


if __name__ == '__main__':
    implied_timescales()  # pragma: no cover

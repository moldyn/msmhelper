# -*- coding: utf-8 -*-
# BSD 3-Clause License
# Copyright (c) 2019-2023, Daniel Nagel
# All rights reserved.
"""Dynamical coring of state trajectory."""
import click
import msmhelper as mh


@click.command(
    no_args_is_help=True,
)
@click.option(
    '-i',
    '--input',
    'input_file',
    required=True,
    type=click.Path(exists=True),
    help=(
        'Path to input file. Needs to be of shape (n_samples, n_features).'
        ' All comment lines need to start with "#"'
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
    '-t',
    '--tcor',
    required=True,
    type=click.IntRange(min=1),
    help='Coring window given in [frames] used for dynamical coring.',
)
@click.option(
    '-o',
    '--output',
    'output_file',
    type=click.Path(),
    help='Path to output file.',
)
def dynamical_coring(input_file, concat_limits, tcor, output_file):
    """Applying dynamical coring on state trajectory."""
    if not output_file:
        output_file = f'{input_file}.dyncor{tcor:.0f}f'
    traj = mh.openmicrostates(
        input_file, limits_file=concat_limits,
    )

    cored_traj = mh.md.dynamical_coring(traj, lagtime=tcor, iterative=True)

    mh.savetxt(
        output_file,
        cored_traj.trajs_flatten,
        header=f'Iteratively dynamical cored with tcor={tcor:.3f} frames',
        fmt='%.0f',
    )


if __name__ == '__main__':
    dynamical_coring()  # pragma: no cover

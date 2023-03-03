# -*- coding: utf-8 -*-
# BSD 3-Clause License
# Copyright (c) 2019-2023, Daniel Nagel
# All rights reserved.
"""Gaussian filtering of coordinates."""
import click
import numpy as np
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
    '-s',
    '--sigma',
    required=True,
    type=click.FloatRange(min=1),
    help='Standard deviation given in [frames] used for Gaussian kernel',
)
@click.option(
    '-o',
    '--output',
    'output_file',
    type=click.Path(),
    help='Path to output file, will be of same shape as input',
)
def gaussian_filtering(input_file, concat_limits, sigma, output_file):
    """Applying gaussian filter on time series."""
    if not output_file:
        output_file = f'{input_file}.gaussian{sigma:.0f}f'
    data = mh.opentxt_limits(
        input_file, limits_file=concat_limits, dtype=np.float32,
    )
    if data[0].ndim == 1:
        data = [part.reshape(-1, 1) for part in data]

    filtered_data = [
        mh.utils.filtering.gaussian_filter(part, sigma=sigma)
        for part in data
    ]

    mh.savetxt(
        output_file,
        np.vstack(filtered_data),
        header=f'Gaussian filtered along time with std={sigma:.3f} frames',
        fmt='%.5f',
    )


if __name__ == '__main__':
    gaussian_filtering()  # pragma: no cover

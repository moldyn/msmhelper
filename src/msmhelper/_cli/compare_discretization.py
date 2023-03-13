# -*- coding: utf-8 -*-
# BSD 3-Clause License
# Copyright (c) 2019-2023, Daniel Nagel
# All rights reserved.
"""Compare different state discretizations."""
import click
import msmhelper as mh
import numpy as np


@click.command(no_args_is_help='-h')
@click.option(
    '--traj1',
    required=True,
    type=click.Path(exists=True),
    help='Path to first state trajectory file (single column ascii file).',
)
@click.option(
    '--traj2',
    required=True,
    type=click.Path(exists=True),
    help='Path to second state trajectory file (single column ascii file).',
)
@click.option(
    '--method',
    type=click.Choice(['symmetric', 'directed']),
    default='symmetric',
    help='Method of calculating similarity.',
)
def compare_discretization(traj1, traj2, method):
    """Similarity measure of two different state discretizations."""
    # load files
    t1 = mh.opentxt(traj1, dtype=np.int64)
    t2 = mh.opentxt(traj2, dtype=np.int64)

    # estimate similarity measure
    sim = mh.md.compare_discretization(t1, t2, method=method)
    print(f'Similarity: {sim:.5f}\n - {traj1}\n - {traj2}\n')


if __name__ == '__main__':
    compare_discretization()

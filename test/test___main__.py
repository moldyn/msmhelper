# -*- coding: utf-8 -*-
"""Tests for the cli script.

BSD 3-Clause License
Copyright (c) 2019-2023, Daniel Nagel
All rights reserved.

"""
import os.path

import pytest
from click.testing import CliRunner
from msmhelper.__main__ import main

# Current directory
HERE = os.path.dirname(__file__)


def test_main():
    runner = CliRunner()
    result = runner.invoke(main)
    assert result.exit_code == 0
    assert '--help' in result.output
    assert 'Usage:' in result.output


@pytest.mark.parametrize('submodule', [
    'ck-test', 'implied-timescales', 'gaussian-filtering',
])
def test_submodules(submodule):
    runner = CliRunner()
    result = runner.invoke(main, [submodule])
    assert result.exit_code == 0
    assert '--help' in result.output
    assert 'Usage:' in result.output


def test_ck_test(tmpdir):
    runner = CliRunner()

    # create trajectories
    trajfile = 'test/assets/8state_microtraj'
    macrotrajfile = 'test/assets/8state_macrotraj'
    output = tmpdir.join('output.pdf')

    params = (
        '--lagtimes 1 2 3 4 5 --frames-per-unit 1 '
        f'--unit frames --grid 2 2 --max-time 500 -o {output}'
    )

    result = runner.invoke(
        main,
        f'ck-test {params} --filename {macrotrajfile}'.split(),
    )
    assert result.exit_code == 0

    result = runner.invoke(
        main,
        (
            f'ck-test {params} --filename {macrotrajfile} '
            f'--microfilename {trajfile}'
        ).split(),
    )
    assert result.exit_code == 0


def test_implied_timescale(tmpdir):
    runner = CliRunner()

    # create trajectories
    trajfile = 'test/assets/8state_microtraj'
    macrotrajfile = 'test/assets/8state_macrotraj'
    output = tmpdir.join('output.pdf')

    params = f'--max-lagtime 25 --frames-per-unit 1 --unit frames -o {output} '

    result = runner.invoke(
        main,
        f'implied-timescales {params} --filename {macrotrajfile}'.split(),
    )
    assert result.exit_code == 0

    result = runner.invoke(
        main,
        (
            f'implied-timescales {params} --filename {macrotrajfile} '
            f'--microfilename {trajfile}'
        ).split(),
    )
    assert result.exit_code == 0


def test_waiting_time_dist(tmpdir):
    runner = CliRunner()

    # create trajectories
    trajfile = 'test/assets/8state_microtraj'
    macrotrajfile = 'test/assets/8state_macrotraj'
    output = tmpdir.join('output.pdf')

    params = (
        f'--start 1 --final 4 --nsteps 10000 -o {output} '
        '--max-lagtime 25 --frames-per-unit 1 --unit frames'
    )

    result = runner.invoke(
        main,
        f'waiting-time-dist {params} --filename {macrotrajfile}'.split(),
    )
    assert result.exit_code == 0

    result = runner.invoke(
        main,
        (
            f'waiting-time-dist {params} --filename {macrotrajfile} '
            f'--microfilename {trajfile}'
        ).split(),
    )
    assert result.exit_code == 0


def test_waiting_times(tmpdir):
    runner = CliRunner()

    # create trajectories
    trajfile = 'test/assets/8state_microtraj'
    macrotrajfile = 'test/assets/8state_macrotraj'
    output = tmpdir.join('output.pdf')

    params = (
        f'--start 1 --final 4 --nsteps 10000 -o {output} '
        '--lagtimes 1 2 3 --frames-per-unit 1 --unit frames'
    )

    result = runner.invoke(
        main,
        f'waiting-times {params} --filename {macrotrajfile}'.split(),
    )
    assert result.exit_code == 0

    result = runner.invoke(
        main,
        (
            f'waiting-times {params} --filename {macrotrajfile} '
            f'--microfilename {trajfile}'
        ).split(),
    )
    assert result.exit_code == 0


def test_gaussian_filtering(tmpdir):
    runner = CliRunner()

    # create trajectories
    input = 'test/assets/8state_microtraj'
    output = tmpdir.join('traj.gaussian')

    params = '--sigma 2'

    result = runner.invoke(
        main,
        f'gaussian-filtering {params} -i {input} -o {output}'.split(),
    )
    assert result.exit_code == 0


def test_dynamical_coring(tmpdir):
    runner = CliRunner()

    # create trajectories
    input = 'test/assets/8state_microtraj'
    output = tmpdir.join('traj.gaussian')

    params = '--tcor 5'

    result = runner.invoke(
        main,
        f'dynamical-coring {params} -i {input} -o {output}'.split(),
    )
    assert result.exit_code == 0


def test_compare_discretization(tmpdir):
    runner = CliRunner()

    # create trajectories
    microfile = 'test/assets/8state_microtraj'
    macrofile = 'test/assets/8state_macrotraj'

    for method in ('symmetric', 'directed'):
        result = runner.invoke(
            main,
            (
                f'compare-discretization --method {method} '
                f'--traj1 {microfile} --traj2 {macrofile}'
            ).split(),
        )
        assert result.exit_code == 0

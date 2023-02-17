# -*- coding: utf-8 -*-
"""Tests for the cli script.

BSD 3-Clause License
Copyright (c) 2019-2023, Daniel Nagel
All rights reserved.

"""
import os.path

import numpy as np
import pytest
from click.testing import CliRunner
from msmhelper.__main__ import main
from msmhelper.utils import datasets

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
    traj = np.loadtxt('test/assets/8state_microtraj')
    macrotraj = np.loadtxt('test/assets/8state_macrotraj')
    trajfile = tmpdir.join('traj')
    macrotrajfile = tmpdir.join('macrotraj')
    np.savetxt(trajfile, traj, fmt='%.0f')
    np.savetxt(macrotrajfile, macrotraj, fmt='%.0f')

    params = (
        '--lagtimes 1 2 3 4 5 --frames-per-unit 1 '
        '--unit frames --grid 2 2 --max-time 500 '
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
    traj, macrotraj = datasets.hummer15_8state(
        0.2, 0.05, 50000, return_macrotraj=True,
    )
    trajfile = tmpdir.join('traj')
    macrotrajfile = tmpdir.join('macrotraj')
    np.savetxt(trajfile, traj, fmt='%.0f')
    np.savetxt(macrotrajfile, macrotraj, fmt='%.0f')

    params = '--max-lagtime 25 --frames-per-unit 1 --unit frames'

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

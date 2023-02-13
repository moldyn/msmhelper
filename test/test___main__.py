# -*- coding: utf-8 -*-
"""Tests for the cli script.

BSD 3-Clause License
Copyright (c) 2019-2023, Daniel Nagel
All rights reserved.

"""
import pytest
from click.testing import CliRunner
from msmhelper.__main__ import main


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

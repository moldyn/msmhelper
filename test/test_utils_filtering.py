# -*- coding: utf-8 -*-
"""Tests for the filtering submodule.

BSD 3-Clause License
Copyright (c) 2019-2023, Daniel Nagel
All rights reserved.

"""
import pytest
from msmhelper.utils import filtering


@pytest.mark.parametrize('data, expected, window', [
    ([1, 1, 1, 2, 2, 2, 3, 3, 3], [1., 1., 1., 2., 2., 2., 3., 3., 3.], 1),
    ([1, 1, 1, 2, 2, 2, 3, 3, 3], [.5, 1., 1., 1.5, 2., 2., 2.5, 3., 3.], 2)])
def test_runningmean(data, expected, window):
    """Test runningmean."""
    assert (filtering.runningmean(data, window) == expected).all()

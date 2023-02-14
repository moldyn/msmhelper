# -*- coding: utf-8 -*-
"""Tests for the plot module.

BSD 3-Clause License
Copyright (c) 2019-2023, Daniel Nagel
All rights reserved.

"""
import pytest
import prettypyplot as pplt
from matplotlib import pyplot as plt

import msmhelper as mh
from msmhelper.utils.datasets import hummer15_8state


@pytest.mark.mpl_image_compare(remove_text=True)
@pytest.mark.parametrize('trajs, kwargs', [
    (hummer15_8state(0.2, 0.05, 100000), {'grid': (4, 2)}),
    (hummer15_8state(0.2, 0.05, 100000), {'grid': (2, 4)}),
    (hummer15_8state(0.2, 0.05, 100000), {'grid': (1, 8)}),
])
def test_plot_ck_test(trajs, kwargs):
    """Test plotting ck_test."""
    lagtimes = [1, 2, 3, 4, 5]
    tmax = 500
    ck = mh.msm.ck_test(trajs, lagtimes=lagtimes, tmax=tmax)
    pplt.use_style(latex=False)
    return mh.plot.plot_ck_test(ck, **kwargs)

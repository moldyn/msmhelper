# -*- coding: utf-8 -*-
"""# Utils

This submodule provides utility functions that can be used to manipulate and test data, such as filtering and validation methods. The functions in this submodule can be used in conjunction with other parts of the software to perform a variety of tasks, making it an essential part of the package.

The submodule is structured into the following submodules:

- [**datasets:**][msmhelper.msm.datasets] This submodule contains all methods related to create example datasets. This submodule needs to be imported explicitly!
- [**msm:**][msmhelper.msm.msm] This submodule contains all methods related to estimate the Markov state model.
- [**tests:**][msmhelper.msm.tests] This submodule holds methods for validating Markov state models.


"""
__all__ = [
    'find_first'
    'find_first',
    'format_state_traj',
    'matrix_power',
    'rename_by_index',
    'rename_by_population',
    'runningmean',
    'shift_data',
    'swapcols',
    'unique',
]

from . import filtering, tests
from ._utils import (
    find_first,
    format_state_traj,
    matrix_power,
    rename_by_index,
    rename_by_population,
    runningmean,
    shift_data,
    swapcols,
    unique,
)

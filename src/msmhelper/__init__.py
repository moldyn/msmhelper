# -*- coding: utf-8 -*-
"""# msmhelper

This package is designed for the analysis of discrete time series data from
Molecular Dynamics (MD) simulations. It focuses on Markov state modeling (MSM),
a powerful technique for analyzing complex systems, and provides a set of
functions for constructing and analyzing Markov models, including methods
calculating transition probabilities, and fitting models to data. The package
is suitable for researchers and engineers who need to analyze large and complex
datasets in order to gain insights into the behavior of the underlying
dynamics.

The module is structured into the following submodules:

- [**io:**][msmhelper.io] This submodule contains all methods related to reading data from text files and writing data to text files, including helpful header comments.

- [**md:**][msmhelper.md] This submodule offers techniques for the analysis of state trajectories&mdash;commonly known as Molecular Dynamics (MD)&mdash;without relying on Markov state models. It encompasses functions for determining timescales, recognizing significant events, correcting dynamical anomalies, and evaluating various state discretization methods.  These functions provide a comprehensive solution for analyzing time-series data and understanding the underlying dynamics of complex systems.

- [**msm:**][msmhelper.msm] This submodule contains methods related to Markov state modeling, a powerful technique for analyzing complex systems. It provides a set of functions for constructing and analyzing Markov models, including methods for calculating transition probabilities and estimating various time scales.

- [**statetraj:**][msmhelper.statetraj] This submodule contains the two classes [StateTraj][msmhelper.statetraj.StateTraj] and [LumpedStateTraj][msmhelper.statetraj.LumpedStateTraj] which are used to represent the time series and allows for a improved performance.

- [**utils:**][msmhelper.utils] This submodule provides utility functions that can be used to manipulate and test data, such as filtering and validation methods. The functions in this submodule can be used in conjunction with other parts of the software to perform a variety of tasks, making it an essential part of the package.

"""
__all__ = [
    'opentxt',
    'savetxt',
    'opentxt_limits',
    'openmicrostates',
    'open_limits',
    'LumpedStateTraj',
    'StateTraj',
    'rename_by_population',
    'rename_by_index',
    'shift_data',
    'unique',
]

from . import md, msm, utils
from .io import (
    opentxt,
    savetxt,
    opentxt_limits,
    openmicrostates,
    open_limits,
)
from .statetraj import LumpedStateTraj, StateTraj
from .utils import (
    rename_by_population,
    rename_by_index,
    shift_data,
    unique,
)

__version__ = '0.6.2'

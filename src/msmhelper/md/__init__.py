# -*- coding: utf-8 -*-
"""# Time Series Analysis

This submodule offers techniques for the analysis of state trajectories&mdash;commonly known as Molecular Dynamics (MD)&mdash;without relying on Markov state models. It encompasses functions for determining timescales, recognizing significant events, correcting dynamical anomalies, and evaluating various state discretization methods.  These functions provide a comprehensive solution for analyzing time-series data and understanding the underlying dynamics of complex systems.

The submodule is structured into the following submodules:

- [**comparison:**][msmhelper.md.comparison] This submodule holds methods to quantify the similarity of different state discretizations.
- [**corrections:**][msmhelper.md.corrections] This submodule provides an implementation of dynamical coring.
- [**timescales:**][msmhelper.msm.timescales] This submodule contains methods for estimating various timescales based on discrete time series.

"""
__all__ = [
    'compare_discretization',
    'dynamical_coring',
    'estimate_waiting_times',
    'estimate_wt',
    'estimate_paths',
]

from .comparison import compare_discretization
from .corrections import dynamical_coring
from .timescales import (
    estimate_waiting_times,
    estimate_wt,
    estimate_paths,
)

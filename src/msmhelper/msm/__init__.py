# -*- coding: utf-8 -*-
"""# Markov State Modeling

This submodule contains methods related to Markov state modeling, a powerful technique for analyzing complex systems. It provides a set of functions for constructing and analyzing Markov models, including methods for calculating transition probabilities and estimating various time scales.

The submodule is structured into the following submodules:

- [**msm:**][msmhelper.msm.msm] This submodule contains all methods related to estimate the Markov state model.
- [**tests:**][msmhelper.msm.tests] This submodule holds methods for validating Markov state models.
- [**timescales:**][msmhelper.msm.timescales] This submodule contains methods for estimating various timescales based on a Markov model.
- [**utils:**][msmhelper.msm.utils] This submodule provides some useful linear algebra methods.

"""
__all__ = [
    'chapman_kolmogorov_test',
    'ck_test',
    'estimate_markov_model',
    'equilibrium_population',
    'peq',
    'row_normalize_matrix',
    'implied_timescales',
    'estimate_waiting_times',
    'estimate_wt',
    'estimate_waiting_time_dist',
    'estimate_wtd',
    'estimate_paths',
]

from .tests import (
    chapman_kolmogorov_test,
    ck_test,
)
from .msm import (
    estimate_markov_model,
    equilibrium_population,
    peq,
    row_normalize_matrix,
)
from .timescales import (
    implied_timescales,
    estimate_waiting_times,
    estimate_wt,
    estimate_waiting_time_dist,
    estimate_wtd,
    estimate_paths,
)

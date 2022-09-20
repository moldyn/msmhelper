# -*- coding: utf-8 -*-
""".. include:: ../../README.md"""

from .benchmark import (
    bh_test,
    buchete_hummer_test,
    chapman_kolmogorov_test,
    ck_test,
)
from .compare import compare_discretization
from .iotext import (
    opentxt,
    savetxt,
    opentxt_limits,
    openmicrostates,
    open_limits,
)
from .md import (
    estimate_msm_waiting_times,
    estimate_msm_wt,
    estimate_paths,
    estimate_waiting_times,
    estimate_wt,
    propagate_MCMC,
)
from .msm import (
    estimate_markov_model,
    equilibrium_population,
    implied_timescales,
    peq,
    row_normalize_matrix,
)
from .statetraj import LumpedStateTraj, StateTraj
from .tools import (
    rename_by_population,
    rename_by_index,
    runningmean,
    shift_data,
    unique,
)

__version__ = '0.6.2'

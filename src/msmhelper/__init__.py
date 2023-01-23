# -*- coding: utf-8 -*-
"""  --8<-- "README.md" """

from .compare import compare_discretization
from .dyncor import dynamical_coring
from .io import (
    opentxt,
    savetxt,
    opentxt_limits,
    openmicrostates,
    open_limits,
)
from .md import (
    estimate_paths,
    estimate_waiting_times,
    estimate_wt,
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

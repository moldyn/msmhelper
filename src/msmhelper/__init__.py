# -*- coding: utf-8 -*-
"""  --8<-- "README.md" """

from . import md, msm
from .compare import compare_discretization
from .io import (
    opentxt,
    savetxt,
    opentxt_limits,
    openmicrostates,
    open_limits,
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

"""
Set of helpful functions.

BSD 3-Clause License
Copyright (c) 2019, Daniel Nagel
All rights reserved.

Author: Daniel Nagel

TODO:
    - create todo

"""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np


# ~~~ FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def shift_data(data, val_old, val_new):
    """
    Shift data from old to new values.

    The basic function is taken from Ashwini_Chaudhary:
    https://stackoverflow.com/a/29408060

    Parameters
    ----------
    data : ndarray or list
        Multi dimensional numpy array.

    val_old : ndarray or list
        Values in data which should be replaced

    val_new : ndarray or list
        Values which will be used instead of old ones.

    Returns
    -------
    data
        Shifted data in same shape as input.

    """
    # convert to np.array
    data = np.asarray(data)
    val_old = np.asarray(val_old)
    val_new = np.asarray(val_new)

    # flatten data
    data_shape = data.shape
    data = data.flatten()

    # shift data
    conv = np.empty(data.max()+1, dtype=val_new.dtype)
    conv[val_old] = val_new
    data_shifted = conv[data]

    return data_shifted.reshape(data_shape)

"""
Set of helpful functions.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

Author: Daniel Nagel

TODO:
    - Correct border effects of running mean

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
    conv = np.empty(data.max() + 1, dtype=val_new.dtype)
    conv[val_old] = val_new
    data_shifted = conv[data]

    return data_shifted.reshape(data_shape)


def runningmean(data, window):
    """
    Computes the running average with window size.

    Function is taken from lapis:
    https://stackoverflow.com/questions/13728392/moving-average-or-running-mean

    Parameters
    ----------
    data : One dimensional numpy array.

    window : Integer which specifies window-width.

    Returns
    -------
    data_rmean
        Data which is time-averaged over the specified window.

    """
    # Calculate running mean
    data_runningmean = np.convolve(data, np.ones(window)/window, mode='same')

    return data_runningmean


def _format_state_trajectory(trajs):
    """Convert state trajectory to list of ndarrays."""
    # 1d ndarray
    if isinstance(trajs, np.ndarray):
        if len(trajs.shape) == 1:
            trajs = [trajs]
    # list
    elif isinstance(trajs, list):
        # list of integers
        if all((np.issubdtype(type(traj), np.integer) for traj in trajs)):
            trajs = [np.array(trajs)]
        # list of lists
        elif all((isinstance(traj, list) for traj in trajs)):
            trajs = [np.asarray(traj) for traj in trajs]
        # not list of ndarrays
        elif not all((isinstance(traj, np.ndarray) for traj in trajs)):
            raise TypeError('Wrong data type of trajs.')

    # check for integers
    if not all((np.issubdtype(traj.dtype, np.integer) for traj in trajs)):
        raise TypeError('States needs to be integers.')

    return trajs


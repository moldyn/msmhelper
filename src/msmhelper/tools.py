# -*- coding: utf-8 -*-
"""Set of helpful functions.

BSD 3-Clause License
Copyright (c) 2019-2023, Daniel Nagel
All rights reserved.

TODO:
    - Correct border effects of running mean

"""
import numba
import numpy as np

from msmhelper.statetraj import StateTraj


def shift_data(array, val_old, val_new, dtype=np.int64):
    """Shift integer array (data) from old to new values.

    !!! warning
        The values of `val_old`, `val_new` and `data` needs to be integers.

    The basic function is based on Ashwini_Chaudhary solution:
    https://stackoverflow.com/a/29408060

    Parameters
    ----------
    array : StateTraj or ndarray or list or list of ndarrays
        1D data or a list of data.
    val_old : ndarray or list
        Values in data which should be replaced. All values needs to be within
        the range of `[data.min(), data.max()]`
    val_new : ndarray or list
        Values which will be used instead of old ones.
    dtype : data-type, optional
        The desired data-type. Needs to be of type unsigned integer.

    Returns
    -------
    array : ndarray
        Shifted data in same shape as input.

    """
    # check data-type
    if not np.issubdtype(dtype, np.integer):
        raise TypeError('An unsigned integer type is needed.')

    # flatten data
    array, shape_kwargs = _flatten_data(array)

    # offset data and val_old to allow negative values
    offset = np.min([np.min(array), np.min(val_new)])

    # convert to np.array
    val_old = (np.asarray(val_old) - offset).astype(dtype)
    val_new = (np.asarray(val_new) - offset).astype(dtype)

    # convert data and shift
    array = (array - offset).astype(dtype)

    # shift data
    conv = np.arange(array.max() + 1, dtype=dtype)
    conv[val_old] = val_new
    array = conv[array]

    # shift data back
    array = array.astype(np.int32) + offset

    # reshape and return
    return _unflatten_data(array, shape_kwargs)


def rename_by_population(trajs, return_permutation=False):
    r"""Rename states sorted by their population starting from 1.

    Parameters
    ----------
    trajs : list or ndarray or list of ndarrays
        State trajectory or list of state trajectories.
    return_permutation : bool
        Return additionaly the permutation to achieve performed renaming.
        Default is False.

    Returns
    -------
    trajs : ndarray
        Renamed data.
    permutation : ndarray
        Permutation going from old to new state nameing. So the `i`th state
        of the new naming corresponds to the old state `permutation[i-1]`.

    """
    # get unique states with population
    states, pop = unique(trajs, return_counts=True)

    # get decreasing order
    idx_sort = np.argsort(pop)[::-1]
    states = states[idx_sort]

    # rename states
    trajs_renamed = shift_data(
        trajs,
        val_old=states,
        val_new=np.arange(len(states)) + 1,
    )
    if return_permutation:
        return trajs_renamed, states
    return trajs_renamed


def rename_by_index(trajs, return_permutation=False):
    r"""Rename states sorted by their numerical values starting from 0.

    Parameters
    ----------
    trajs : list or ndarray or list of ndarrays
        State trajectory or list of state trajectories.
    return_permutation : bool
        Return additionaly the permutation to achieve performed renaming.
        Default is False.

    Returns
    -------
    trajs : ndarray
        Renamed data.
    permutation : ndarray
        Permutation going from old to new state nameing. So the `i`th state
        of the new naming corresponds to the old state `permutation[i-1]`.

    """
    # get unique states
    states = unique(trajs)

    # rename states
    trajs_renamed = shift_data(
        trajs,
        val_old=states,
        val_new=np.arange(len(states)),
    )
    if return_permutation:
        return trajs_renamed, states
    return trajs_renamed


def unique(trajs, **kwargs):
    r"""Apply numpy.unique to traj.

    Parameters
    ----------
    trajs : list or ndarray or list of ndarrays
        State trajectory or list of state trajectories.
    **kwargs
        Arguments of [numpy.unique][]

    Returns
    -------
    unique : ndarray
        Array containing all states, see numpy for more details.

    """
    # flatten data
    trajs, _ = _flatten_data(trajs)

    # get unique states with population
    return np.unique(trajs, **kwargs)


def runningmean(array, window):
    r"""Compute centered running average with given window size.

    This function returns the centered based running average of the given
    data. The output of this function is of the same length as the input,
    by assuming that the given data is zero before and after the given
    series. Hence, there are border affects which are not corrected.

    !!! warning
        If the given window is even (not symmetric) it will be shifted towards
        the beginning of the current value. So for `window=4`, it will consider
        the current position \(i\), the two to the left \(i-2\) and \(i-1\) and
        one to the right \(i+1\).

    Function is taken from lapis:
    https://stackoverflow.com/questions/13728392/moving-average-or-running-mean

    Parameters
    ----------
    array : ndarray
        One dimensional numpy array.
    window : int
        Integer which specifies window-width.

    Returns
    -------
    array_rmean : ndarray
        Data which is time-averaged over the specified window.

    """
    # Calculate running mean
    return np.convolve(
        array,
        np.ones(window) / window,
        mode='same',
    )


def swapcols(array, indicesold, indicesnew):
    r"""Interchange cols of an ndarray.

    This method swaps the specified columns.

    Parameters
    ----------
    array : ndarray
        2D numpy array.
    indicesold : integer or ndarray
        1D array of indices.
    indicesnew : integer or ndarray
        1D array of new indices

    Returns
    -------
    array_swapped : ndarray
        2D numpy array with swappend columns.

    """
    # cast to 1d arrays
    indicesnew = _asindex(indicesnew)
    indicesold = _asindex(indicesold)

    if len(indicesnew) != len(indicesold):
        raise ValueError('Indices needs to be of same shape.')

    # cast data
    array = np.asarray(array)

    if np.all(indicesnew == indicesold):
        return array

    # fails for large data sets
    # noqa: E800 # array.T[indicesold] = array.T[indicesnew]
    array_swapped = np.copy(array)
    array_swapped.T[indicesold] = array.T[indicesnew]

    return array_swapped


def _asindex(idx):
    """Cast to 1d integer ndarray."""
    idx = np.atleast_1d(idx).astype(np.int64)
    if len(idx.shape) > 1:
        raise ValueError('Wrong dimensionality of indices.')
    return idx


def format_state_traj(trajs):
    """Convert state trajectory to list of ndarrays.

    Parameters
    ----------
    trajs : list or ndarray or list of ndarray
        State trajectory/trajectories. The states should start from zero and
        need to be integers.

    Returns
    -------
    trajs : list of ndarray
        Return list of ndarrays of integers.

    """
    # list or tuple
    if isinstance(trajs, (tuple, list)):
        # list of integers
        if all((np.issubdtype(type(state), np.integer) for state in trajs)):
            trajs = [np.array(trajs)]
        # list of lists
        elif all((isinstance(traj, list) for traj in trajs)):
            trajs = [np.array(traj) for traj in trajs]
    # ndarray
    if isinstance(trajs, np.ndarray):
        if len(trajs.shape) == 1:
            trajs = [trajs]
        elif len(trajs.shape) == 2:
            trajs = list(trajs)

    # check for integers
    _check_state_traj(trajs)

    return trajs


def _check_state_traj(trajs):
    """Check if state trajectory is correct formatted."""
    # check for integers
    if isinstance(trajs, StateTraj):
        return True

    for traj in trajs:
        if not isinstance(traj, np.ndarray):
            raise TypeError(
                'Trajs need to be np.ndarray but are {0}'.format(type(traj)),
            )
        if not np.issubdtype(traj.dtype, np.integer):
            raise TypeError(
                'States needs to be integers but are {0}'.format(traj.dtype),
            )
    return True


def _flatten_data(array):
    """Flatten data to 1D ndarray.

    This method flattens ndarrays, list of ndarrays to a 1D ndarray. This can
    be undone with [msmhelper.tools._unflatten_data][].

    Parameters
    ----------
    array : list or ndarray or list of ndarrays
        1D data or a list of data.

    Returns
    -------
    data : ndarray
        Flattened data.
    kwargs : dict
        Dictionary with information to restore shape.

    """
    kwargs = {}

    # flatten data
    if isinstance(array, (tuple, list)):
        # list of ndarrays, lists or tuples
        if all((isinstance(row, (np.ndarray, tuple, list)) for row in array)):
            # get shape and flatten
            kwargs['limits'] = np.cumsum([len(row) for row in array])
            array = np.concatenate(array)
        # list of numbers
        else:
            array = np.asarray(array)
    elif isinstance(array, np.ndarray):
        # get shape and flatten
        kwargs['data_shape'] = array.shape
        array = array.flatten()

    return array, kwargs


def _unflatten_data(array, kwargs):
    """Unflatten data to original structure.

    This method undoes [msmhelper.tools._flatten_data][].

    Parameters
    ----------
    array : ndarray
        Flattened data.
    kwargs : dict
        Dictionary with information to restore shape. Provided by
        _flatten_data().

    Returns
    -------
    array : ndarray, list, list of ndarrays
        Data with restored shape.

    """
    # parse kwargs
    data_shape = kwargs.get('data_shape')
    limits = kwargs.get('limits')

    # reshape
    if data_shape is not None:
        array = array.reshape(data_shape)
    elif limits is not None:
        array = np.split(array, limits)[:-1]

    return array


@numba.njit
def matrix_power(matrix, power):
    """Calculate matrix power with np.linalg.matrix_power.

    Numba wrapper for [numpy.linalg.matrix_power][]. Only for float matrices.

    Parameters
    ----------
    matrix : ndarray
        2d matrix of type float.
    power : int, float
        Power of matrix.

    Returns
    -------
    matpow : ndarray
        Matrix power.

    """
    return np.linalg.matrix_power(matrix, power)


@numba.njit
def find_first(search_val, array):
    """Return first occurance of item in array."""
    for idx, idx_val in enumerate(array):
        if search_val == idx_val:
            return idx
    return -1

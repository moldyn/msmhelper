# -*- coding: utf-8 -*-
# BSD 3-Clause License
# Copyright (c) 2019-2023, Daniel Nagel
# All rights reserved.
"""Set of filtering functions.

!!! todo
    - Correct border effects of running mean

"""
import numpy as _np
from scipy.ndimage import gaussian_filter as _gaussian_filter
from scipy.ndimage import gaussian_filter1d as _gaussian_filter_1d


def runningmean(array, window):
    r"""Compute centered running average with given window size.

    This function returns the centered based running average of the given
    data. The output of this function is of the same length as the input,
    by assuming that the given data is zero before and after the given series.
    Hence, there are border affects which are not corrected.

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
    ndim = _np.asarray(array).ndim
    if ndim > 1:
        raise ValueError(
            'Runningmean is only defined for 1D data, but'
            f'{ndim:.0f}D data were provided.'
        )

    # Calculate running mean
    return _np.convolve(
        array,
        _np.ones(window) / window,
        mode='same',
    )


def gaussian_filter(array, sigma):
    r"""Compute Gaussian filter along axis=0.

    Parameters
    ----------
    array : ndarray
        One dimensional numpy array.
    sigma : float
        Float which specifies the standard deviation of the Gaussian kernel
        (window-width).

    Returns
    -------
    array_filtered : ndarray
        Data which is time-averaged with the specified Gaussian kernel.

    """
    array = _np.asarray(array, dtype=_np.float64)
    ndim = array.ndim
    if ndim > 2:
        raise ValueError(
            'Gaussian filtering is only defined for 1D and 2D data, but'
            f'{ndim:.0f}D data were provided.'
        )

    # Calculate running mean
    if ndim == 1:
        return _gaussian_filter_1d(
            array,
            sigma=sigma,
            mode='nearest',
        )
    return _gaussian_filter(
        array,
        sigma=(sigma, 0),
        mode='nearest',
    )

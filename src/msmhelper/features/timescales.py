# -*- coding: utf-8 -*-
# BSD 3-Clause License
# Copyright (c) 2019-2023, Daniel Nagel
# All rights reserved.
"""# Estimation of autocorrelation function."""
import decorit
import numba
import numpy as np


@decorit.alias('estimate_acf')
def estimate_autocorrelation_function(
    features,
    global_mean=True,
    max_time=None,
    n_times=200,
):
    r"""Calculate the autocorrelation functions.

    Calculate the autocorrelation function, which is defined by

    $$
        \text{ACF}(x|\tau) = \frac{
            \langle
                (x_{t+\tau} - \langle x \rangle)(x_{t} - \langle x \rangle)
            \rangle
        }{\sigma_x^2}
    $$

    This method has no timestep information and therefore estimates all
    timescales in units of frames.

    !!! note
        This method is only valid if the mean has no timedependent drift.

    Parameters
    ----------
    features : ndarray or list of ndarray (n_trajs, n_times, n_features)
        Feature trajectories with given shape.
    global_mean : bool, optional
        If `True` the mean over all trajectories will be used (for each
        feature), else the mean will be estimated for each trajectory.
    max_time : int, optional
        Largest value to estimate the ACF [frames]. By default, it is estimated
        based on the slowest decaying input feature.
    n_times : int, optional
        Number of steps to evaluate the ACF.

    Returns
    -------
    acfs : ndarray (n_features, n_times)
        Matrix containing the ACF for all features.
    times :  ndarray (n_times, )
        List with corresponding time steps [frames].

    """
    # check that 3d array
    if not (
        (
            isinstance(features, list) and
            isinstance(features[0], np.ndarray) and
            features[0].ndim == 2
        ) or (
            isinstance(features, np.ndarray) and features.ndim == 3
        )
    ):
        raise ValueError(
            'features needs to be a either a list of ndarrays or a 3d ndarray.'
            ' Please ensure to reshape your data accordingly.'
        )

    # check if single feature
    if max_time is not None and max_time <= 0:
        raise TypeError('max_time needs to be a positive integer.')

    if max_time is None:
        max_time = _get_max_time(features, global_mean)

    # one order of magnitude larger
    times = np.unique(
        np.geomspace(1, max_time, n_times).astype(int),
    )

    # do not convert for pytest coverage
    if not numba.config.DISABLE_JIT:  # pragma: no cover
        if isinstance(features, list):
            features = numba.typed.List(features)
    return _estimate_acf(features, times, global_mean), times


@numba.njit
def _estimate_acf(features, times, global_mean):
    """Estimate ACF."""
    n_trajs = len(features)
    n_features = len(features[0])
    return np.array([
        _estimate_acf_feature(
            [
                features[idx_feature][:, idx_traj]
                for idx_traj in range(n_trajs)
            ],
            times,
            global_mean,
        ) for idx_feature in range(n_features)
    ])


@numba.njit
def _estimate_acf_feature(feature, times, global_mean):
    """Calculate ACF for a single feature."""
    acf = np.empty_like(times, dtype=np.float64)
    for idx, time in enumerate(times):
        acf[idx] = _acf(feature, time, global_mean)
    return acf


def _get_max_time(features, global_mean):
    """Get time where ACF is decayed."""
    n_frames, n_features = features[0].shape
    time = 2  # in frames
    below_thr = np.full(n_features, False, dtype=bool)
    while time < 0.1 * n_frames:
        acf = _estimate_acf(features, [time], global_mean)
        below_thr = np.logical_or(
            below_thr, acf < 0.1,
        )
        time *= 5
        if np.all(below_thr):
            break
    return time


@numba.njit(parallel=True, fastmath=True)
def _acf(feature, time, global_mean):
    """Calculate acf for single pcs."""
    feature_stacked = np.array([
        val for feature_traj in feature for val in feature_traj
    ])
    n_frames = len(feature_stacked)
    n_trajs = len(feature_stacked)
    nom = 0

    if global_mean:
        mean = np.mean(feature_stacked)
        var = np.var(feature_stacked)

        for feature_traj in feature:
            nom += np.sum(
                (
                    feature_traj[:len(feature_traj) - time] - mean
                ) * (
                    feature_traj[time:] - mean
                ),
            )
        nom = nom / n_frames  # get mean
        return nom / var

    for feature_traj in feature:
        mean = np.mean(feature_traj)
        var = np.var(feature_traj)
        nom += np.mean(
            (
                feature_traj[:len(feature_traj) - time] - mean
            ) * (
                feature_traj[time:] - mean
            ),
        ) / var

    return nom / n_trajs

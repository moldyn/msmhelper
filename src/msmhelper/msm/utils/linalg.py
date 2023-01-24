# -*- coding: utf-8 -*-
"""Basic linear algebra method.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

"""
import decorit
import numpy as np

from msmhelper.utils import tests


@decorit.alias('eigl')
def left_eigenvectors(matrix, nvals=None):
    """Estimate left eigenvectors.

    Estimates the left eigenvectors and corresponding eigenvalues of a
    quadratic matrix.

    Parameters
    ----------
    matrix : ndarray
        Quadratic 2d matrix eigenvectors and eigenvalues or determined of.
    nvals : int, optional
        Number of returned eigenvalues and -vectors. Using ensures probability
        of real valued matrices.

    Returns
    -------
    eigenvalues : ndarray
        N eigenvalues sorted by their value (descending).
    eigenvectors : ndarray
        N eigenvectors sorted by descending eigenvalues.

    """
    matrix = np.asarray(matrix)
    return _eigenvectors(matrix.transpose(), nvals)


@decorit.alias('eig')
def right_eigenvectors(matrix, nvals=None):
    """Estimate right eigenvectors.

    Estimates the right eigenvectors and corresponding eigenvalues of a
    quadratic matrix.

    Parameters
    ----------
    matrix : ndarray
        Quadratic 2d matrix eigenvectors and eigenvalues or determined of.
    nvals : int, optional
        Number of returned eigenvalues and -vectors. Using ensures probability
        of real valued matrices.

    Returns
    -------
    eigenvalues : ndarray
        N eigenvalues sorted by their value (descending).
    eigenvectors : ndarray
        N eigenvectors sorted by descending eigenvalues.

    """
    matrix = np.asarray(matrix)
    return _eigenvectors(matrix, nvals)


def _eigenvectors(matrix, nvals):
    """Estimate eigenvectors."""
    if not tests.is_quadratic(matrix):
        raise TypeError('Matrix needs to be quadratic {0}'.format(matrix))

    if nvals is None:
        nvals = len(matrix)
    elif nvals > len(matrix):
        raise TypeError(
            '{nvals} eigenvalues requested but '.format(nvals=nvals) +
            'matrix of dimension {dim}'.format(dim=len(matrix))
        )

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Transpose eigenvectors, since v[:, i] is eigenvector
    eigenvectors = eigenvectors.T

    # Sort them by descending eigenvalues
    idx_eigenvalues = eigenvalues.argsort()[::-1]

    return (
        np.real_if_close(eigenvalues[idx_eigenvalues][:nvals]),
        np.real_if_close(eigenvectors[idx_eigenvalues][:nvals]),
    )


@decorit.alias('eiglvals')
def left_eigenvalues(matrix, nvals=None):
    """Estimate left eigenvalues.

    Estimates the left eigenvalues of a quadratic matrix.

    Parameters
    ----------
    matrix : ndarray
        Quadratic 2d matrix eigenvalues or determined of.
    nvals : int, optional
        Number of returned eigenvalues and -vectors. Using ensures probability
        of real valued matrices.

    Returns
    -------
    eigenvalues : ndarray
        N eigenvalues sorted by their value (descending).

    """
    matrix = np.asarray(matrix)
    return _eigenvalues(np.transpose(matrix), nvals)


@decorit.alias('eigvals')
def right_eigenvalues(matrix, nvals=None):
    """Estimate right eigenvalues.

    Estimates the right eigenvalues of a quadratic matrix.

    Parameters
    ----------
    matrix : ndarray
        Quadratic 2d matrix eigenvalues or determined of.
    nvals : int, optional
        Number of returned eigenvalues and -vectors. Using ensures probability
        of real valued matrices.

    Returns
    -------
    eigenvalues : ndarray
        N eigenvalues sorted by their value (descending).

    """
    matrix = np.asarray(matrix)
    return _eigenvalues(matrix, nvals)


def _eigenvalues(matrix, nvals):
    """Estimate eigenvalues."""
    return _eigenvectors(matrix, nvals)[0]

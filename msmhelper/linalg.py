# -*- coding: utf-8 -*-
"""Basic linear algebra method.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

"""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np

from msmhelper import tests


# ~~~ FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def left_eigenvectors(matrix):
    """Estimate left eigenvectors.

    Estimates the left eigenvectors and corresponding eigenvalues of a
    quadratic matrix.

    Parameters
    ----------
    matrix : ndarray
        Quadratic 2d matrix eigenvectors and eigenvalues or determined of.

    Returns
    -------
    eigenvalues : ndarray
        N eigenvalues sorted by their value (descending).

    eigenvectors : ndarray
        N eigenvectors sorted by descending eigenvalues.

    """
    matrix = np.asarray(matrix)
    if not tests.is_quadratic(matrix):
        raise TypeError('Matrix needs to be quadratic {0}'.format(matrix))

    return _left_eigenvectors(matrix)


def _left_eigenvectors(matrix):
    """Estimate left eigenvectors."""
    # Transpose matrix and therefore determine eigenvalues and left
    # eigenvectors
    matrix = matrix.transpose()
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Transpose eigenvectors, since v[:,i] is eigenvector
    eigenvectors = eigenvectors.T

    # Sort them by descending eigenvalues
    idx_eigenvalues = eigenvalues.argsort()[::-1]

    return eigenvalues[idx_eigenvalues], eigenvectors[idx_eigenvalues]


def right_eigenvectors(matrix):
    """Estimate right eigenvectors.

    Estimates the right eigenvectors and corresponding eigenvalues of a
    quadratic matrix.

    Parameters
    ----------
    matrix : ndarray
        Quadratic 2d matrix eigenvectors and eigenvalues or determined of.

    Returns
    -------
    eigenvalues : ndarray
        N eigenvalues sorted by their value (descending).

    eigenvectors : ndarray
        N eigenvectors sorted by descending eigenvalues.

    """
    matrix = np.asarray(matrix)
    if not tests.is_quadratic(matrix):
        raise TypeError('Matrix needs to be quadratic {0}'.format(matrix))

    return _right_eigenvectors(matrix)


def _right_eigenvectors(matrix):
    """Estimate right eigenvectors."""
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Transpose eigenvectors, since v[:,i] is eigenvector
    eigenvectors = eigenvectors.T

    # Sort them by descending eigenvalues
    idx_eigenvalues = eigenvalues.argsort()[::-1]

    return eigenvalues[idx_eigenvalues], eigenvectors[idx_eigenvalues]

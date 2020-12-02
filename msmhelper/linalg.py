# -*- coding: utf-8 -*-
"""Basic linear algebra method.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

"""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np

from msmhelper import tests
from msmhelper.decorators import shortcut


# ~~~ FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@shortcut('eigl')
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

    eigenvalues, eigenvectors = _left_eigenvectors(matrix)
    return np.real_if_close(eigenvalues), np.real_if_close(eigenvectors)


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


@shortcut('eig')
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

    eigenvalues, eigenvectors = _right_eigenvectors(matrix)
    return np.real_if_close(eigenvalues), np.real_if_close(eigenvectors)


def _right_eigenvectors(matrix):
    """Estimate right eigenvectors."""
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Transpose eigenvectors, since v[:,i] is eigenvector
    eigenvectors = eigenvectors.T

    # Sort them by descending eigenvalues
    idx_eigenvalues = eigenvalues.argsort()[::-1]

    return eigenvalues[idx_eigenvalues], eigenvectors[idx_eigenvalues]


@shortcut('eiglvals')
def left_eigenvalues(matrix):
    """Estimate left eigenvalues.

    Estimates the left eigenvalues of a quadratic matrix.

    Parameters
    ----------
    matrix : ndarray
        Quadratic 2d matrix eigenvalues or determined of.

    Returns
    -------
    eigenvalues : ndarray
        N eigenvalues sorted by their value (descending).

    """
    matrix = np.asarray(matrix)
    if not tests.is_quadratic(matrix):
        raise TypeError('Matrix needs to be quadratic {0}'.format(matrix))

    return np.real_if_close(_left_eigenvalues(matrix))


def _left_eigenvalues(matrix):
    """Estimate left eigenvalues."""
    # Transpose matrix and therefore determine eigenvalues
    matrix = matrix.transpose()
    eigenvalues = np.linalg.eigvals(matrix)

    # Sort them by descending eigenvalues
    idx_eigenvalues = eigenvalues.argsort()[::-1]

    return eigenvalues[idx_eigenvalues]


@shortcut('eigvals')
def right_eigenvalues(matrix):
    """Estimate right eigenvalues.

    Estimates the right eigenvalues of a quadratic matrix.

    Parameters
    ----------
    matrix : ndarray
        Quadratic 2d matrix eigenvalues or determined of.

    Returns
    -------
    eigenvalues : ndarray
        N eigenvalues sorted by their value (descending).

    """
    matrix = np.asarray(matrix)
    if not tests.is_quadratic(matrix):
        raise TypeError('Matrix needs to be quadratic {0}'.format(matrix))

    return np.real_if_close(_right_eigenvalues(matrix))


def _right_eigenvalues(matrix):
    """Estimate right eigenvalues."""
    eigenvalues = np.linalg.eigvals(matrix)

    # Sort them by descending eigenvalues
    idx_eigenvalues = eigenvalues.argsort()[::-1]

    return eigenvalues[idx_eigenvalues]

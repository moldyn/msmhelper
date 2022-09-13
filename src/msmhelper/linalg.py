# -*- coding: utf-8 -*-
"""Basic linear algebra method.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

"""
import decorit
import numpy as np

from msmhelper import tests


@decorit.alias('eigl')
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

    eigenvalues, eigenvectors = _eigenvectors(matrix.transpose())
    return np.real_if_close(eigenvalues), np.real_if_close(eigenvectors)


@decorit.alias('eig')
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

    eigenvalues, eigenvectors = _eigenvectors(matrix)
    return np.real_if_close(eigenvalues), np.real_if_close(eigenvectors)


def _eigenvectors(matrix):
    """Estimate eigenvectors."""
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Transpose eigenvectors, since v[:,i] is eigenvector
    eigenvectors = eigenvectors.T

    # Sort them by descending eigenvalues
    idx_eigenvalues = eigenvalues.argsort()[::-1]

    return eigenvalues[idx_eigenvalues], eigenvectors[idx_eigenvalues]


@decorit.alias('eiglvals')
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

    return np.real_if_close(_eigenvalues(np.transpose(matrix)))


@decorit.alias('eigvals')
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

    return np.real_if_close(_eigenvalues(matrix))


def _eigenvalues(matrix):
    """Estimate eigenvalues."""
    eigenvalues = np.linalg.eigvals(matrix)

    # Sort them by descending eigenvalues
    idx_eigenvalues = eigenvalues.argsort()[::-1]

    return eigenvalues[idx_eigenvalues]

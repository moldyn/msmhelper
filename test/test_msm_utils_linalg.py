# -*- coding: utf-8 -*-
"""Tests for the linalg module.

BSD 3-Clause License
Copyright (c) 2019-2023, Daniel Nagel
All rights reserved.

"""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pytest

from msmhelper.msm.utils import linalg


# ~~~ TESTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@pytest.mark.parametrize('matrix, eigenvaluesref, eigenvectorsref', [
    (np.array([[1, 6, -1], [2, -1, -2], [1, 0, -1]]), np.array([3, 0, -4]),
     [np.array([-2, -3, 2]) / np.sqrt(17),
      np.array([-1, -6, 13]) / np.sqrt(206),
      np.array([-1, 2, 1]) / np.sqrt(6)])])
def test_left_eigenvectors(matrix, eigenvaluesref, eigenvectorsref):
    """Test left eigenvectors estimate."""
    eigenvalues, eigenvectors = linalg.left_eigenvectors(matrix)
    np.testing.assert_array_almost_equal(eigenvalues, eigenvaluesref)
    np.testing.assert_array_almost_equal(
        np.abs(eigenvectors),
        np.abs(eigenvectorsref),
    )

    with pytest.raises(TypeError):
        linalg.left_eigenvectors(matrix[0])

    # test eigenvalues method
    eigenvalues = linalg.left_eigenvalues(matrix)
    np.testing.assert_array_almost_equal(eigenvalues, eigenvaluesref)

    with pytest.raises(TypeError):
        linalg.left_eigenvalues(matrix[0])


@pytest.mark.parametrize('matrix, eigenvaluesref, eigenvectorsref', [
    (np.array([[1, 6, -1], [2, -1, -2], [1, 0, -1]]), np.array([3, 0, -4]),
     [np.array([8, 3, 2]) / np.sqrt(77),
      np.array([1, 0, 1]) / np.sqrt(2),
      np.array([-9, 8, 3]) / np.sqrt(154)])])
def test_right_eigenvectors(matrix, eigenvaluesref, eigenvectorsref):
    """Test left eigenvectors estimate."""
    eigenvalues, eigenvectors = linalg.right_eigenvectors(matrix)
    np.testing.assert_array_almost_equal(eigenvalues, eigenvaluesref)
    np.testing.assert_array_almost_equal(
        np.abs(eigenvectors),
        np.abs(eigenvectorsref),
    )

    with pytest.raises(TypeError):
        linalg.right_eigenvectors(matrix[0])

    # test eigenvalues method
    eigenvalues = linalg.right_eigenvalues(matrix)
    np.testing.assert_array_almost_equal(eigenvalues, eigenvaluesref)

    with pytest.raises(TypeError):
        linalg.right_eigenvalues(matrix[0])

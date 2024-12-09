import pytest
import pennylane_snowflurry.processing.decompositions.native_decomp_functions as decomp
import pennylane as qml
from functools import reduce
import numpy as np

def are_matrices_equivalent(matrix1, matrix2, tolerance=1e-9):
    """
    Checks if two matrices are equal up to a complex multiplicative factor.

    Args:
        matrix1 (ndarray): First matrix.
        matrix2 (ndarray): Second matrix.
        tolerance (float): Numerical tolerance for comparison.

    Returns:
        bool: True if the matrices are equal up to a complex factor, False otherwise.
    """
    if matrix1.shape != matrix2.shape:
        return False

    # Flatten matrices for comparison
    matrix1_flat = matrix1.flatten()
    matrix2_flat = matrix2.flatten()

    # Remove zero entries to avoid division errors
    nonzero_indices = np.nonzero(matrix1_flat)[0]
    if len(nonzero_indices) == 0:
        # If both matrices are zero matrices
        return np.allclose(matrix1, matrix2, atol=tolerance)

    # Choose a reference index based on the first non-zero element
    ref_index = nonzero_indices[0]

    # Calculate the complex factor
    factor = matrix2_flat[ref_index] / matrix1_flat[ref_index]

    # Scale matrix1 by the factor and compare with matrix2
    scaled_matrix1 = matrix1 * factor

    return np.allclose(scaled_matrix1, matrix2, atol=tolerance)

# Test with multiple values
@pytest.mark.parametrize("a, b, e, expected", [
    (1, 1, 1E-8, True),
    (1, 2, 2, True),
    (2, 1, 2, True),
    (1, 2, 1E-8, False),
    (2, 1, 1E-8, False)
])
def test_is_close_enough_to(a, b, e, expected):
    result = decomp.is_close_enough_to(a, b, e)
    assert result == expected

def test_custom_tdag():
    mat = qml.adjoint(qml.T)([0]).matrix()
    result = decomp._custom_tdag([0])
    mat2 = reduce(lambda i, s: i @ s.matrix(), result, np.identity(2))
    assert are_matrices_equivalent(mat, mat2)

def test_custom_sx():
    mat = qml.SX([0]).matrix()
    result = decomp._custom_sx([0])
    mat2 = reduce(lambda i, s: i @ s.matrix(), result, np.identity(2))
    assert are_matrices_equivalent(mat, mat2)

def test_custom_sxdag():
    mat = qml.adjoint(qml.SX)([0]).matrix()
    result = decomp._custom_sxdag([0])
    mat2 = reduce(lambda i, s: i @ s.matrix(), result, np.identity(2))
    assert are_matrices_equivalent(mat, mat2)

def test_custom_s():
    mat = qml.S([0]).matrix()
    result = decomp._custom_s([0])
    mat2 = reduce(lambda i, s: i @ s.matrix(), result, np.identity(2))
    assert are_matrices_equivalent(mat, mat2)

def test_custom_sdag():
    mat = qml.adjoint(qml.S)([0]).matrix()
    result = decomp._custom_sdag([0])
    mat2 = reduce(lambda i, s: i @ s.matrix(), result, np.identity(2))
    assert are_matrices_equivalent(mat, mat2)

def test_custom_h():
    mat = qml.H([0]).matrix()
    result = decomp._custom_h([0])
    mat2 = reduce(lambda i, s: i @ s.matrix(), result, np.identity(2))
    assert are_matrices_equivalent(mat, mat2)

def test_custom_cnot():
    mat = qml.CNOT([0, 1]).matrix()
    result = decomp._custom_cnot([0, 1])
    mat2 = reduce(lambda i, s: i @ s.matrix(wire_order = [0, 1]), result, np.identity(4))
    assert are_matrices_equivalent(mat, mat2)

def test_custom_cy():
    mat = qml.CY([0, 1]).matrix()
    result = decomp._custom_cy([0, 1])
    mat2 = reduce(lambda i, s: i @ s.matrix(wire_order = [0, 1]), result, np.identity(4))
    assert are_matrices_equivalent(mat, mat2)

def test_custom_swap():
    mat = qml.SWAP([0, 1]).matrix()
    result = decomp._custom_swap([0, 1])
    mat2 = reduce(lambda i, s: i @ s.matrix(wire_order = [0, 1]), result, np.identity(4))
    assert are_matrices_equivalent(mat, mat2)

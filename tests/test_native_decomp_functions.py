import pytest
import pennylane_calculquebec.processing.decompositions.native_decomp_functions as decomp
import pennylane as qml
from functools import reduce
import numpy as np

def rz(angle): 
    return np.array([[np.exp(-1j*angle/2), 0], 
                     [0, np.exp(1j*angle/2)]])

def rx(angle): 
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    return h @ rz(angle) @ h

def ry(angle):
    return rz(np.pi/2) @ rx(angle) @ rz(-np.pi/2)

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
    
    tolerance = tolerance + 1j*tolerance
    
    if matrix1.shape != matrix2.shape:
        return False

    matrix2_dag = np.transpose(np.conjugate(matrix2))
    id = np.round(matrix1 @ matrix2_dag, 4)
    value = id[0][0]
    
    for i in range(id.shape[0]):
        if abs(id[i][i] - value) > tolerance:
            return False
    return True

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

@pytest.mark.parametrize("phi, expected", [
    (2*np.pi, []),
    (0, []),
    (-np.pi/4, ["TDagger"]),
    (7 * np.pi/4, ["TDagger"]),
    (-np.pi/2, ["ZM90"]),
    (3 * np.pi/2, ["ZM90"]),
    (np.pi, ["PauliZ"]),
    (-3*np.pi/2, ["Z90"]),
    (np.pi/2, ["Z90"]),
    (-7*np.pi/4, ["T"]),
    (np.pi/4, ["T"]),
    (1, ["RZ"])
])
def test_custom_rz(phi, expected):
    result = decomp._custom_rz(phi, 0)
    mat1 = rz(phi)
    mat2 = reduce(lambda i, s: i @ s.matrix(), result, np.identity(2))
    assert are_matrices_equivalent(mat1, mat2)
    
    for i, op in enumerate(result):
        assert op.name == expected[i]
    
@pytest.mark.parametrize("phi, expected", [
    (2*np.pi, []),
    (0, []),
    (-np.pi/2, ["XM90"]),
    (3 * np.pi/2, ["XM90"]),
    (np.pi, ["PauliX"]),
    (-3*np.pi/2, ["X90"]),
    (np.pi/2, ["X90"]),
    (1, ["Z90", "X90", "Z90", "RZ", "Z90", "X90", "Z90"])
])
def test_custom_rx(phi, expected):
    result = decomp._custom_rx(phi, 0)
    mat1 = rx(phi)
    mat2 = reduce(lambda i, s: i @ s.matrix(), result, np.identity(2))
    assert are_matrices_equivalent(mat1, mat2)
    
    for i, op in enumerate(result):
        assert op.name == expected[i]

@pytest.mark.parametrize("phi, expected", [
    (2*np.pi, []),
    (0, []),
    (-np.pi/2, ["YM90"]),
    (3 * np.pi/2, ["YM90"]),
    (np.pi, ["PauliY"]),
    (-3*np.pi/2, ["Y90"]),
    (np.pi/2, ["Y90"]),
    (1, ["X90", "RZ", "XM90"])
])
def test_custom_ry(phi, expected):
    
    result = decomp._custom_ry(phi, 0)
    mat1 = ry(phi)
    mat2 = reduce(lambda i, s: i @ s.matrix(), result, np.identity(2))
    assert are_matrices_equivalent(mat1, mat2)
    
    for i, op in enumerate(result):
        assert op.name == expected[i]
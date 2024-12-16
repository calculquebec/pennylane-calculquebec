import numpy as np
import pennylane as qml
from functools import reduce
from pennylane_calculquebec.processing.custom_gates import \
    TDagger, X90, XM90, Y90, YM90, Z90, ZM90
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


def test_tdagger():
    gate = TDagger(0)
    compare = qml.adjoint(qml.T)(0)
    
    # check matrix
    mat1 = gate.matrix()
    mat2 = compare.matrix()
    assert are_matrices_equivalent(mat1, mat2)
    
    # check eigen values
    eig1 = gate.eigvals()
    eig2 = compare.eigvals()

    assert np.allclose(eig1, eig2)
    
    # check
    decomp1 = gate.decomposition()
    decomp2 = compare.decomposition()
    decomp1_mat = reduce(lambda i, s: i @ s.matrix(), decomp1, np.identity(2))
    decomp2_mat = reduce(lambda i, s: i @ s.matrix(), decomp2, np.identity(2))
    
    assert are_matrices_equivalent(decomp1_mat, decomp2_mat)

    pow = lambda i: [qml.PhaseShift((8 - i) * np.pi / 4, 0)]
    
    for i in range(8):
        pow1 = gate.pow(i)
        pow2 = pow(i)
        
        pow1_mat = reduce(lambda i, s: i @ s.matrix(), pow1, np.identity(2))
        pow2_mat = reduce(lambda i, s: i @ s.matrix(), pow2, np.identity(2))
        assert are_matrices_equivalent(pow1_mat, pow2_mat)
    
    assert are_matrices_equivalent(gate.adjoint().matrix(), qml.T(0).matrix())
    
    for angles in zip([-np.pi/4, 0, 0], gate.single_qubit_rot_angles()):
        assert angles[0] == angles[1]

def test_x90():
    gate = X90(0)
    compare = qml.RX(np.pi/2, 0)
    
    # check matrix
    mat1 = gate.matrix()
    mat2 = compare.matrix()
    assert are_matrices_equivalent(mat1, mat2)
    
    # check eigen values
    eig1 = gate.eigvals()
    eig2 = compare.eigvals()

    assert np.allclose(eig1, eig2)
    
    # check
    decomp1 = gate.decomposition()
    decomp2 = [qml.SX(0)]
    decomp1_mat = reduce(lambda i, s: i @ s.matrix(), decomp1, np.identity(2))
    decomp2_mat = reduce(lambda i, s: i @ s.matrix(), decomp2, np.identity(2))
    
    assert are_matrices_equivalent(decomp1_mat, decomp2_mat)

    pow = lambda i: [qml.RX(i * np.pi/2, 0)]
    
    for i in range(4):
        pow1 = gate.pow(i)
        pow2 = pow(i)
        
        pow1_mat = reduce(lambda i, s: i @ s.matrix(), pow1, np.identity(2))
        pow2_mat = reduce(lambda i, s: i @ s.matrix(), pow2, np.identity(2))
        assert are_matrices_equivalent(pow1_mat, pow2_mat)
    
    assert are_matrices_equivalent(gate.adjoint().matrix(), qml.adjoint(qml.SX)(0).matrix())
    
    for angles in zip([np.pi / 2, np.pi/2, -np.pi/2], gate.single_qubit_rot_angles()):
        assert angles[0] == angles[1]

def test_xm90():
    gate = XM90(0)
    compare = qml.RX(-np.pi/2, 0)
    
    # check matrix
    mat1 = gate.matrix()
    mat2 = compare.matrix()
    assert are_matrices_equivalent(mat1, mat2)
    
    # check eigen values
    eig1 = gate.eigvals()
    eig2 = compare.eigvals()

    assert np.allclose(eig1, eig2)
    
    # check
    decomp1 = gate.decomposition()
    decomp2 = [qml.adjoint(qml.SX)(0)]
    decomp1_mat = reduce(lambda i, s: i @ s.matrix(), decomp1, np.identity(2))
    decomp2_mat = reduce(lambda i, s: i @ s.matrix(), decomp2, np.identity(2))
    
    assert are_matrices_equivalent(decomp1_mat, decomp2_mat)

    pow = lambda i: [qml.RX((4 - i) * np.pi / 2, 0)]
    
    for i in range(4):
        pow1 = gate.pow(i)
        pow2 = pow(i)
        
        pow1_mat = reduce(lambda i, s: i @ s.matrix(), pow1, np.identity(2))
        pow2_mat = reduce(lambda i, s: i @ s.matrix(), pow2, np.identity(2))
        assert are_matrices_equivalent(pow1_mat, pow2_mat)
    
    assert are_matrices_equivalent(gate.adjoint().matrix(), qml.RX(-np.pi, 0).matrix())
    
    for angles in zip([np.pi / 2, -np.pi/2, -np.pi/2], gate.single_qubit_rot_angles()):
        assert angles[0] == angles[1]

def test_y90():
    gate = Y90(0)
    compare = qml.RY(np.pi/2, 0)
    
    # check matrix
    mat1 = gate.matrix()
    mat2 = compare.matrix()
    assert are_matrices_equivalent(mat1, mat2)
    
    # check eigen values
    eig1 = gate.eigvals()
    eig2 = compare.eigvals()

    assert np.allclose(eig1, eig2)
    
    # check decomposition
    decomp1 = gate.decomposition()
    decomp2 = [qml.RY(np.pi/2, 0)]
    decomp1_mat = reduce(lambda i, s: i @ s.matrix(), decomp1, np.identity(2))
    decomp2_mat = reduce(lambda i, s: i @ s.matrix(), decomp2, np.identity(2))
    
    assert are_matrices_equivalent(decomp1_mat, decomp2_mat)

    # check powers
    pow = lambda i: [qml.RY(i  * np.pi/2, 0)]
    
    for i in range(4):
        pow1 = gate.pow(i)
        pow2 = pow(i)
        
        pow1_mat = reduce(lambda i, s: i @ s.matrix(), pow1, np.identity(2))
        pow2_mat = reduce(lambda i, s: i @ s.matrix(), pow2, np.identity(2))
        assert are_matrices_equivalent(pow1_mat, pow2_mat)
    
    # check adjoints
    assert are_matrices_equivalent(gate.adjoint().matrix(), qml.RY(-np.pi/2, 0).matrix())
    
    # check single qubit rot angles
    for angles in zip([0, np.pi/2, 0], gate.single_qubit_rot_angles()):
        assert angles[0] == angles[1]

def test_ym90():
    gate = YM90(0)
    compare = qml.RY(-np.pi/2, 0)
    
    # check matrix
    mat1 = gate.matrix()
    mat2 = compare.matrix()
    assert are_matrices_equivalent(mat1, mat2)
    
    # check eigen values
    eig1 = gate.eigvals()
    eig2 = compare.eigvals()

    assert np.allclose(eig1, eig2)
    
    # check decomposition
    decomp1 = gate.decomposition()
    decomp2 = [qml.RY(-np.pi/2, 0)]
    decomp1_mat = reduce(lambda i, s: i @ s.matrix(), decomp1, np.identity(2))
    decomp2_mat = reduce(lambda i, s: i @ s.matrix(), decomp2, np.identity(2))
    
    assert are_matrices_equivalent(decomp1_mat, decomp2_mat)

    # check powers
    pow = lambda i: [qml.RY((4 - i)  * np.pi/2, 0)]
    
    for i in range(4):
        pow1 = gate.pow(i)
        pow2 = pow(i)
        
        pow1_mat = reduce(lambda i, s: i @ s.matrix(), pow1, np.identity(2))
        pow2_mat = reduce(lambda i, s: i @ s.matrix(), pow2, np.identity(2))
        assert are_matrices_equivalent(pow1_mat, pow2_mat)
    
    # check adjoints
    assert are_matrices_equivalent(gate.adjoint().matrix(), qml.RY(np.pi/2, 0).matrix())
    
    # check single qubit rot angles
    for angles in zip([0, -np.pi/2, 0], gate.single_qubit_rot_angles()):
        assert angles[0] == angles[1]

def test_z90():
    gate = Z90(0)
    compare = qml.RZ(np.pi/2, 0)
    
    # check matrix
    mat1 = gate.matrix()
    mat2 = compare.matrix()
    assert are_matrices_equivalent(mat1, mat2)
    
    # check eigen values
    eig1 = gate.eigvals()
    eig2 = compare.eigvals()

    assert np.allclose(eig1, eig2)
    
    # check decomposition
    decomp1 = gate.decomposition()
    decomp2 = [qml.RZ(np.pi/2, 0)]
    decomp1_mat = reduce(lambda i, s: i @ s.matrix(), decomp1, np.identity(2))
    decomp2_mat = reduce(lambda i, s: i @ s.matrix(), decomp2, np.identity(2))
    
    assert are_matrices_equivalent(decomp1_mat, decomp2_mat)

    # check powers
    pow = lambda i: [qml.RZ(i  * np.pi/2, 0)]
    
    for i in range(4):
        pow1 = gate.pow(i)
        pow2 = pow(i)
        
        pow1_mat = reduce(lambda i, s: i @ s.matrix(), pow1, np.identity(2))
        pow2_mat = reduce(lambda i, s: i @ s.matrix(), pow2, np.identity(2))
        assert are_matrices_equivalent(pow1_mat, pow2_mat)
    
    # check adjoints
    assert are_matrices_equivalent(gate.adjoint().matrix(), qml.RZ(-np.pi/2, 0).matrix())
    
    # check single qubit rot angles
    for angles in zip([np.pi/2, 0, 0], gate.single_qubit_rot_angles()):
        assert angles[0] == angles[1]

def test_zm90():
    gate = ZM90(0)
    compare = qml.RZ(-np.pi/2, 0)
    
    # check matrix
    mat1 = gate.matrix()
    mat2 = compare.matrix()
    assert are_matrices_equivalent(mat1, mat2)
    
    # check eigen values
    eig1 = gate.eigvals()
    eig2 = compare.eigvals()

    assert np.allclose(eig1, eig2)
    
    # check decomposition
    decomp1 = gate.decomposition()
    decomp2 = [qml.RZ(-np.pi/2, 0)]
    decomp1_mat = reduce(lambda i, s: i @ s.matrix(), decomp1, np.identity(2))
    decomp2_mat = reduce(lambda i, s: i @ s.matrix(), decomp2, np.identity(2))
    
    assert are_matrices_equivalent(decomp1_mat, decomp2_mat)

    # check powers
    pow = lambda i: [qml.RZ((4 - i)  * np.pi/2, 0)]
    
    for i in range(4):
        pow1 = gate.pow(i)
        pow2 = pow(i)
        
        pow1_mat = reduce(lambda i, s: i @ s.matrix(), pow1, np.identity(2))
        pow2_mat = reduce(lambda i, s: i @ s.matrix(), pow2, np.identity(2))
        assert are_matrices_equivalent(pow1_mat, pow2_mat)
    
    # check adjoints
    assert are_matrices_equivalent(gate.adjoint().matrix(), qml.RZ(np.pi/2, 0).matrix())
    
    # check single qubit rot angles
    for angles in zip([-np.pi/2, 0, 0], gate.single_qubit_rot_angles()):
        assert angles[0] == angles[1]

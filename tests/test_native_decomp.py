import numpy as np
from pennylane_calculquebec.processing.steps.base_decomposition import CliffordTDecomposition
from pennylane_calculquebec.processing.steps.native_decomposition import MonarqDecomposition
from pennylane_calculquebec.utility.api import instructions
import pennylane as qml
from pennylane.tape import QuantumTape
import pytest

from functools import reduce


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



def test_native_decomp_toffoli():
    preproc = CliffordTDecomposition()
    step = MonarqDecomposition()
    
    ops = [qml.Hadamard(0), qml.Hadamard(1), qml.Toffoli([0, 1, 2])]
    tape = QuantumTape(ops=ops, measurements=[qml.probs()])
    new_tape = preproc.execute(tape)
    new_tape = step.execute(new_tape)
            
    assert all(op.name in instructions for op in new_tape.operations)
    
    mat1 = reduce(lambda i, s: i @ s.matrix(wire_order=tape.wires), tape.operations, np.identity(1 << len(tape.wires)))
    mat2 = reduce(lambda i, s: i @ s.matrix(wire_order=new_tape.wires), new_tape.operations, np.identity(1 << len(new_tape.wires)))
    
    assert are_matrices_equivalent(mat1, mat2)

def test_native_decomp_unitary():
    preproc = CliffordTDecomposition()
    step = MonarqDecomposition()
    
    ops = [qml.Hadamard(0), qml.QubitUnitary(np.array([[-1, 1], [1, 1]])/np.sqrt(2), 0)]
    tape = QuantumTape(ops=ops, measurements=[qml.probs()])
    new_tape = preproc.execute(tape)
    new_tape = step.execute(new_tape)

    assert all(op.name in instructions for op in new_tape.operations)
    
    mat1 = reduce(lambda i, s: i @ s.matrix(wire_order=tape.wires), tape.operations, np.identity(1 << len(tape.wires)))
    mat2 = reduce(lambda i, s: i @ s.matrix(wire_order=new_tape.wires), new_tape.operations, np.identity(1 << len(new_tape.wires)))
    
    assert are_matrices_equivalent(mat1, mat2)

def test_native_decomp_cu():
    preproc = CliffordTDecomposition()
    step = MonarqDecomposition()
    
    ops = [qml.Hadamard(0), qml.Hadamard(1), qml.Hadamard(2), qml.ControlledQubitUnitary(np.array([[0, 1], [1, 0]]), [0, 1], [2], [0, 1])]
    tape = QuantumTape(ops=ops, measurements=[qml.probs()])
    new_tape = preproc.execute(tape)
    new_tape = step.execute(new_tape)

    assert all(op.name in instructions for op in new_tape.operations)
    
    mat1 = reduce(lambda i, s: i @ s.matrix(wire_order=tape.wires), tape.operations, np.identity(1 << len(tape.wires)))
    mat2 = reduce(lambda i, s: i @ s.matrix(wire_order=new_tape.wires), new_tape.operations, np.identity(1 << len(new_tape.wires)))
    
    assert are_matrices_equivalent(mat1, mat2)
    
def test_gate_not_in_decomp_map():
    ops = [qml.Toffoli([0, 1, 2])]
    tape = QuantumTape(ops=ops)
    step = MonarqDecomposition()

    with pytest.raises(Exception):
        step.execute(tape)

import numpy as np
from pennylane_calculquebec.processing.steps.base_decomposition import CliffordTDecomposition, BaseDecomposition
from pennylane_calculquebec.processing.steps.native_decomposition import MonarqDecomposition
from pennylane_calculquebec.utility.api import instructions
from pennylane_calculquebec.utility.debug import is_equal_matrices
import pennylane as qml
from pennylane.tape import QuantumTape
import pytest
from functools import reduce

def test_base_decomp_class_empty_gates():
    step = BaseDecomposition()
    assert len(step.base_gates) == 0

def test_base_decomp_toffoli():
    step = CliffordTDecomposition()
    ops = [qml.Hadamard(0), qml.Hadamard(1), qml.Toffoli([0, 1, 2])]
    tape = QuantumTape(ops=ops, measurements=[qml.probs()])
    new_tape = step.execute(tape)
    assert all(op.name in step.base_gates for op in new_tape.operations)

    mat1 = reduce(lambda i, s: i @ s.matrix(wire_order=tape.wires), tape.operations, np.identity(1 << len(tape.wires)))
    mat2 = reduce(lambda i, s: i @ s.matrix(wire_order=new_tape.wires), new_tape.operations, np.identity(1 << len(new_tape.wires)))
    
    assert is_equal_matrices(mat1, mat2)

@pytest.mark.xfail
def test_base_decomp_unitary():
    step = CliffordTDecomposition()
    
    ops = [qml.Hadamard(0), qml.QubitUnitary(np.array([[-1, 1], [1, 1]])/np.sqrt(2), 0)]
    tape = QuantumTape(ops=ops, measurements=[qml.probs()])
    new_tape = step.execute(tape)
    assert all(op.name in step.base_gates for op in new_tape.operations)

    mat1 = reduce(lambda i, s: i @ s.matrix(wire_order=tape.wires), tape.operations, np.identity(1 << len(tape.wires)))
    mat2 = reduce(lambda i, s: i @ s.matrix(wire_order=new_tape.wires), new_tape.operations, np.identity(1 << len(new_tape.wires)))
    
    assert is_equal_matrices(mat1, mat2)

@pytest.mark.xfail
def test_base_decomp_cu():
    step = CliffordTDecomposition()
    
    ops = [qml.Hadamard(0), qml.Hadamard(1), qml.ControlledQubitUnitary(np.array([[1, 1], [1, -1]])/np.sqrt(2), [0, 1], [2], [0, 1])]
    tape = QuantumTape(ops=ops, measurements=[qml.probs()])
    new_tape = step.execute(tape)
    
    assert all(op.name in step.base_gates for op in new_tape.operations)

    mat1 = reduce(lambda i, s: i @ s.matrix(wire_order=tape.wires), tape.operations, np.identity(1 << len(tape.wires)))
    mat2 = reduce(lambda i, s: i @ s.matrix(wire_order=new_tape.wires), new_tape.operations, np.identity(1 << len(new_tape.wires)))
    
    assert is_equal_matrices(mat1, mat2)

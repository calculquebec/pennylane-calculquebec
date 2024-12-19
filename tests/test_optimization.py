import numpy as np
from pennylane_calculquebec.processing.steps.base_decomposition import CliffordTDecomposition
from pennylane_calculquebec.processing.steps.optimization import IterativeCommuteAndMerge
import pennylane as qml
from pennylane.tape import QuantumTape
import pytest
from unittest.mock import patch
from functools import reduce
from pennylane_calculquebec.utility.debug import is_equal_matrices

@pytest.fixture
def mock_commute_and_merge():
    with patch("pennylane_calculquebec.processing.steps.optimization.commute_and_merge") as mock:
        yield mock

# ajouter des tests pour les autres fonctions que execute

def test_execute_calls_commute_and_merge(mock_commute_and_merge):
    mock_commute_and_merge.side_effect = lambda tape: tape

    tape = QuantumTape([], [], 1000)
    IterativeCommuteAndMerge().execute(tape)
    assert mock_commute_and_merge.call_count == 6
    
def test_optimize_qubit_unitary():
    ops = [qml.Hadamard(0), qml.QubitUnitary(np.array([[-1, 1], [1, 1]])/np.sqrt(2), 0)]
    tape = QuantumTape(ops=ops, measurements=[qml.probs()])
    new_tape = CliffordTDecomposition().execute(tape)
    new_tape = IterativeCommuteAndMerge().execute(new_tape)

    assert len(new_tape.operations) == 1
    
    mat1 = reduce(lambda i, s: i @ s.matrix(wire_order=tape.wires), tape.operations, np.identity(1 << len(tape.wires)))
    mat2 = reduce(lambda i, s: i @ s.matrix(wire_order=new_tape.wires), new_tape.operations, np.identity(1 << len(new_tape.wires)))

    zero = np.array([1])
    for i in tape.wires:
        zero = np.kron(np.array([1, 0]), zero)
        
    results1 = mat1 @ zero
    results2 = mat2 @ zero
    
    assert all(np.round(np.real(np.abs(results1)), 4) == np.round(np.real(np.abs(results2)), 4))

def test_optimize_toffoli():
    ops = [qml.Hadamard(0), qml.Hadamard(1), qml.Toffoli([0, 1, 2])]
    tape = QuantumTape(ops=ops, measurements=[qml.probs()])
    new_tape = CliffordTDecomposition().execute(tape)
    new_tape = IterativeCommuteAndMerge().execute(new_tape)
    
    assert len(new_tape.operations) == 31
    
    mat1 = reduce(lambda i, s: i @ s.matrix(wire_order=tape.wires), tape.operations, np.identity(1 << len(tape.wires)))
    mat2 = reduce(lambda i, s: i @ s.matrix(wire_order=new_tape.wires), new_tape.operations, np.identity(1 << len(new_tape.wires)))

    zero = np.array([1])
    for i in tape.wires:
        zero = np.kron(np.array([1, 0]), zero)
        
    results1 = mat1 @ zero
    results2 = mat2 @ zero
    
    assert all(np.round(np.real(np.abs(results1)), 4) == np.round(np.real(np.abs(results2)), 4))

def test_optimize_cu():
    ops = [qml.Hadamard(0), qml.Hadamard(1), qml.ControlledQubitUnitary(np.array([[1, 1], [1, -1]])/np.sqrt(2), [0, 1], [2], [0, 1])]
    tape = QuantumTape(ops=ops, measurements=[qml.probs()])
    new_tape = CliffordTDecomposition().execute(tape)
    new_tape = IterativeCommuteAndMerge().execute(new_tape)
    
    assert len(new_tape.operations) == 51
    
    
    mat1 = reduce(lambda i, s: i @ s.matrix(wire_order=tape.wires), tape.operations, np.identity(1 << len(tape.wires)))
    mat2 = reduce(lambda i, s: i @ s.matrix(wire_order=new_tape.wires), new_tape.operations, np.identity(1 << len(new_tape.wires)))

    zero = np.array([1])
    for i in tape.wires:
        zero = np.kron(np.array([1, 0]), zero)
        
    results1 = mat1 @ zero
    results2 = mat2 @ zero
    
    assert all(np.round(np.real(np.abs(results1)), 4) == np.round(np.real(np.abs(results2)), 4))
    
def test_swap_cnot():
    result = IterativeCommuteAndMerge.swap_cnot([4, 2])
    assert [w for w in result[0].wires] == [4, 2] and result[0].name == "CNOT"
    assert [w for w in result[1].wires] == [2, 4] and result[1].name == "CNOT"
    assert [w for w in result[2].wires] == [4, 2] and result[2].name == "CNOT"
    
    with pytest.raises(ValueError):
        result = IterativeCommuteAndMerge.swap_cnot([3])

def test_HCZH_cnot():
    result = IterativeCommuteAndMerge.HCZH_cnot([4, 2])
    
    assert result[0].wires[0] == 2 and result[0].name == "Hadamard"
    assert [w for w in result[1].wires] == [4, 2] and result[1].name == "CZ"
    assert result[2].wires[0] == 2 and result[2].name == "Hadamard"
    
    with pytest.raises(ValueError):
        IterativeCommuteAndMerge.HCZH_cnot([3])

def test_ZXZ_Hadamard():
    result = IterativeCommuteAndMerge.ZXZ_Hadamard([4])
    assert result[0].wires[0] == 4 and result[0].name == "S"
    assert result[1].wires[0] == 4 and result[1].name == "SX"
    assert result[2].wires[0] == 4 and result[2].name == "S"
    
    with pytest.raises(ValueError):
        IterativeCommuteAndMerge.ZXZ_Hadamard([1, 2])


def test_Y_to_ZXZ():
    op = qml.CNOT([0, 1])
    with pytest.raises(ValueError):
        IterativeCommuteAndMerge.Y_to_ZXZ(op)
    
    op = qml.RX(-np.pi/5, 0)
    with pytest.raises(ValueError):
        IterativeCommuteAndMerge.Y_to_ZXZ(op)
    
    op = qml.RY(-np.pi/5, 0)
    result = IterativeCommuteAndMerge.Y_to_ZXZ(op)
    assert result[0] == qml.RZ(np.pi/2, 0)
    assert result[1] == qml.RX(-np.pi/5, 0)
    assert result[2] == qml.RZ(-np.pi/2, 0)
    
def test_get_rid_of_Y_rotations():
    with patch("pennylane_calculquebec.processing.steps.optimization.IterativeCommuteAndMerge.Y_to_ZXZ") as Y_to_ZXZ_mock:
        Y_to_ZXZ_mock.return_value = []
        
        # non-y gate dont call Y_to_ZXZ
        tape = QuantumTape([qml.CNOT([0, 1])])
        IterativeCommuteAndMerge.get_rid_of_y_rotations(tape)
        Y_to_ZXZ_mock.assert_not_called()
        
        # Y gates call Y_to_ZXZ
        tape = QuantumTape([qml.RY(0, 0)])
        IterativeCommuteAndMerge.get_rid_of_y_rotations(tape)
        Y_to_ZXZ_mock.assert_called_once()
    
    # 2 qubit gate
    tape = QuantumTape([qml.CNOT([0, 1])])
    result = IterativeCommuteAndMerge.get_rid_of_y_rotations(tape)
    assert tape.operations == result.operations
    
    # x basis rotation
    tape = QuantumTape([qml.RX(np.pi/5, 0)])
    result = IterativeCommuteAndMerge.get_rid_of_y_rotations(tape)
    assert tape.operations == result.operations
    
    # y basis rotation
    tape = QuantumTape([qml.RY(np.pi/5, 0)])
    result = IterativeCommuteAndMerge.get_rid_of_y_rotations(tape)
    assert result.operations == [qml.RZ(np.pi/2, 0), qml.RX(np.pi/5, 0), qml.RZ(-np.pi/2, 0)]

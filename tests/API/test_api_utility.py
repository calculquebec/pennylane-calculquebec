import numpy as np
import pennylane as qml
import pytest
from unittest.mock import patch
from pennylane_calculquebec.utility.api import ApiUtility, keys
from pennylane.tape import QuantumTape


@pytest.fixture
def mock_convert_instruction():
    with patch(
        "pennylane_calculquebec.utility.api.ApiUtility.convert_instruction"
    ) as mock:
        yield mock


@pytest.fixture
def mock_basic_auth():
    with patch("pennylane_calculquebec.utility.api.ApiUtility.basic_auth") as mock:
        yield mock


def test_convert_instructions():
    op = qml.CNOT([0, 1])

    with pytest.raises(ValueError):
        ApiUtility.convert_instruction(op)

    op = qml.CZ([0, 1])
    op_dict = ApiUtility.convert_instruction(op)
    assert op_dict[keys.TYPE] == "cz"
    assert op_dict[keys.QUBITS] == [0, 1]
    assert keys.PARAMETERS not in op_dict

    op = qml.RZ(np.pi, 0)
    op_dict = ApiUtility.convert_instruction(op)
    assert op_dict[keys.TYPE] == "rz"
    assert op_dict[keys.QUBITS] == [0]
    assert abs(op_dict[keys.PARAMETERS]["lambda"] - np.pi) < 1e-8


def test_convert_circuit(mock_convert_instruction):
    mock_convert_instruction.side_effect = lambda op: op.name

    wires = [4, 1, 5]

    tape = QuantumTape(
        ops=[qml.PauliX(4), qml.PauliY(1), qml.PauliZ(5)], measurements=[], shots=1000
    )
    result = ApiUtility.convert_circuit(tape)
    assert mock_convert_instruction.call_count == 3

    for i, op in enumerate(result[keys.OPERATIONS]):
        assert op == tape.operations[i].name

    tape = QuantumTape(ops=[], measurements=[qml.counts(wires=wires)], shots=1000)
    result = ApiUtility.convert_circuit(tape)

    for i, op in enumerate(result[keys.OPERATIONS]):
        assert op[keys.QUBITS] == [wires[i]]
        assert op[keys.BITS] == [i]

    tape = QuantumTape(
        ops=[], measurements=[qml.expval(qml.PauliZ(4) @ qml.PauliZ(1))], shots=1000
    )
    result = ApiUtility.convert_circuit(tape)

    for i, op in enumerate(result[keys.OPERATIONS]):
        assert op[keys.QUBITS] == [wires[i]]
        assert op[keys.BITS] == [i]


def test_basic_auth():
    test = ApiUtility.basic_auth("user", "password")
    assert test == "Basic dXNlcjpwYXNzd29yZA=="


def test_headers(mock_basic_auth):

    mock_basic_auth.side_effect = lambda u, p: f"{u}{p}"

    user = "user"
    password = "password"
    realm = "realm"

    result = ApiUtility.headers(user, password, realm)

    assert result["Authorization"] == user + password
    assert result["Content-Type"] == "application/json"
    assert result["X-Realm"] == realm


def test_measurement_bits_start_from_zero():
    """
    Test that measurement bits are always assigned as a sequence starting from zero,
    regardless of the qubit wire indices used in the circuit.
    
    This test prevents a regression where measurement bits might not form a proper
    sequence starting from 0, which could cause representation issues when displaying
    results to users.
    """
    # Test case 1: Non-sequential qubit wires should still have bits starting from 0
    non_sequential_wires = [5, 2, 8, 1]
    tape1 = QuantumTape(
        ops=[], 
        measurements=[qml.counts(wires=non_sequential_wires)], 
        shots=1000
    )
    result1 = ApiUtility.convert_circuit(tape1)
    
    # Extract readout operations (they come after gate operations)
    readout_ops = [op for op in result1[keys.OPERATIONS] if op.get(keys.TYPE) == "readout"]
    
    # Verify that bits are sequential starting from 0
    expected_bits = list(range(len(non_sequential_wires)))
    actual_bits = [op[keys.BITS][0] for op in readout_ops]
    assert actual_bits == expected_bits, f"Expected bits {expected_bits}, got {actual_bits}"
    
    # Verify that qubits correspond to the original wire order
    expected_qubits = non_sequential_wires
    actual_qubits = [op[keys.QUBITS][0] for op in readout_ops]
    assert actual_qubits == expected_qubits, f"Expected qubits {expected_qubits}, got {actual_qubits}"
    
    # Test case 2: Large gap in wire indices should still have consecutive bits
    large_gap_wires = [100, 50, 200]
    tape2 = QuantumTape(
        ops=[], 
        measurements=[qml.counts(wires=large_gap_wires)], 
        shots=1000
    )
    result2 = ApiUtility.convert_circuit(tape2)
    
    readout_ops2 = [op for op in result2[keys.OPERATIONS] if op.get(keys.TYPE) == "readout"]
    expected_bits2 = [0, 1, 2]
    actual_bits2 = [op[keys.BITS][0] for op in readout_ops2]
    assert actual_bits2 == expected_bits2, f"Expected bits {expected_bits2}, got {actual_bits2}"
    
    # Test case 3: Multiple measurements should each start their bits from 0
    tape3 = QuantumTape(
        ops=[], 
        measurements=[
            qml.counts(wires=[10, 15]), 
            qml.expval(qml.PauliZ(20) @ qml.PauliZ(25))
        ], 
        shots=1000
    )
    result3 = ApiUtility.convert_circuit(tape3)
    
    readout_ops3 = [op for op in result3[keys.OPERATIONS] if op.get(keys.TYPE) == "readout"]
    
    # First measurement should have bits [0, 1]
    first_measurement_bits = [op[keys.BITS][0] for op in readout_ops3[:2]]
    assert first_measurement_bits == [0, 1], f"First measurement bits should be [0, 1], got {first_measurement_bits}"
    
    # Second measurement should have bits [0, 1] (starting fresh)
    second_measurement_bits = [op[keys.BITS][0] for op in readout_ops3[2:]]
    assert second_measurement_bits == [0, 1], f"Second measurement bits should be [0, 1], got {second_measurement_bits}"


def test_body():
    result = ApiUtility.job_body(
        circuit="a", circuit_name="b", project_id="c", machine_name="d", shots="e"
    )

    assert result[keys.CIRCUIT] == "a"
    assert result[keys.NAME] == "b"
    assert result[keys.PROJECT_ID] == "c"
    assert result[keys.MACHINE_NAME] == "d"
    assert result[keys.SHOT_COUNT] == "e"

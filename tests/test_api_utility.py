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


def test_body():
    result = ApiUtility.job_body(
        circuit="a", circuit_name="b", project_id="c", machine_name="d", shots="e"
    )

    assert result[keys.CIRCUIT] == "a"
    assert result[keys.NAME] == "b"
    assert result[keys.PROJECT_ID] == "c"
    assert result[keys.MACHINE_NAME] == "d"
    assert result[keys.SHOT_COUNT] == "e"


@pytest.fixture
def mock_tape():
    """A simple mock quantum circuit for testing convert_circuit."""
    ops = [qml.Hadamard(wires=0), qml.CNOT(wires=[0, 1])]
    measurements = [qml.counts(wires=[0, 1])]
    return QuantumTape(ops, measurements)


## This checks if the bits associated with the measurement is a sequence starting at 0
def test_convert_circuit(mock_tape):
    circuit_dict = ApiUtility.convert_circuit(mock_tape)
    # Extract the operations of type "readout" from the circuit dictionary
    readout_operations = [
        operation
        for operation in circuit_dict.get("operations", [])
        if operation.get("type") == "readout"
    ]
    # Collect the bits from the readout operations
    measurement_bits = [
        bit for operation in readout_operations for bit in operation.get("bits", [])
    ]
    # Check if the bits are a sequence starting at 0
    assert measurement_bits == list(
        range(len(measurement_bits))
    ), "Measurement bits are not a sequence starting at 0"

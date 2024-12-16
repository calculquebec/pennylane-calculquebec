import numpy as np
import pennylane as qml
import pytest
from unittest.mock import patch
from pennylane_snowflurry.utility.api import ApiUtility, keys
from pennylane.tape import QuantumTape

@pytest.fixture
def mock_convert_instruction():
    with patch("pennylane_snowflurry.utility.api.ApiUtility.convert_instruction") as mock:
        yield mock

@pytest.fixture
def mock_basic_auth():
    with patch("pennylane_snowflurry.utility.api.ApiUtility.basic_auth") as mock:
        yield mock


def test_convert_instructions():
    op = qml.CNOT([0, 1])
    
    with pytest.raises(ValueError):
        ApiUtility.convert_instruction(op)
    
    op = qml.CZ([0, 1])
    op_dict = ApiUtility.convert_instruction(op)
    assert op_dict[keys.type] == "cz"
    assert op_dict[keys.qubits] == [0, 1]
    assert keys.parameters not in op_dict
    
    op = qml.RZ(np.pi, 0)
    op_dict = ApiUtility.convert_instruction(op)
    assert op_dict[keys.type] == "rz"
    assert op_dict[keys.qubits] == [0]
    assert abs(op_dict[keys.parameters]["lambda"] - np.pi) < 1E-8

def test_convert_circuit(mock_convert_instruction):
    mock_convert_instruction.side_effect = lambda op : op.name
    
    wires = [4, 1, 5]
    
    tape = QuantumTape([qml.X(4), qml.Y(1), qml.Z(5)], [], 1000)
    result = ApiUtility.convert_circuit(tape)
    assert mock_convert_instruction.call_count == 3
    
    for i, op in enumerate(result[keys.operations]):
        assert op == tape.operations[i].name
    
    tape = QuantumTape([], [qml.counts(wires=wires)], 1000)
    result = ApiUtility.convert_circuit(tape)
    
    for i, op in enumerate(result[keys.operations]):
        assert op[keys.qubits] == [wires[i]]
        assert op[keys.bits] == [i]

def test_basic_auth():
    test = ApiUtility.basic_auth("user", "password")
    assert test == "Basic dXNlcjpwYXNzd29yZA=="

def  test_headers(mock_basic_auth):
    
    mock_basic_auth.side_effect = lambda u, p: f"{u}{p}"
    
    user = "user"
    password = "password"
    realm = "realm"
    
    result = ApiUtility.headers(user, password, realm)
    
    assert result["Authorization"] == user + password
    assert result["Content-Type"] == "application/json"
    assert result["X-Realm"] == realm

def test_body():
    result = ApiUtility.job_body(circuit="a", 
                                 circuit_name="b", 
                                 project_id="c", 
                                 machine_name="d", 
                                 shots="e")
    
    assert result[keys.circuit] == "a"
    assert result[keys.name] == "b"
    assert result[keys.projectID] == "c"
    assert result[keys.machineName] == "d"
    assert result[keys.shotCount] == "e"
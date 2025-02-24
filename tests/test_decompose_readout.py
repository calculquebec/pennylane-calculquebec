import pennylane as qml
from pennylane.tape import QuantumTape
import numpy as np
from pennylane_calculquebec.processing.processing_exception import ProcessingException
from pennylane_calculquebec.processing.steps import DecomposeReadout
import pytest
from unittest.mock import patch
from pennylane.ops import Prod

@pytest.fixture
def mock_get_ops_for_product():
    with patch("pennylane_calculquebec.processing.steps.DecomposeReadout.get_ops_for_product") as mock:
        yield mock

@pytest.mark.parametrize("obs", [
    qml.PauliZ(0) @ qml.PauliZ(1),
    qml.PauliX(0) @ qml.PauliZ(1),
    qml.PauliY(0) @ qml.PauliZ(1),
    qml.Hadamard(0) @ qml.PauliZ(1),
    qml.PauliZ(0) @ qml.PauliX(1),
    qml.PauliX(0) @ qml.PauliX(1),
    qml.PauliY(0) @ qml.PauliX(1),
    qml.Hadamard(0) @ qml.PauliX(1),
    qml.PauliZ(0) @ qml.PauliY(1),
    qml.PauliX(0) @ qml.PauliY(1),
    qml.PauliY(0) @ qml.PauliY(1),
    qml.Hadamard(0) @ qml.PauliY(1),
    qml.PauliZ(0) @ qml.Hadamard(1),
    qml.PauliX(0) @ qml.Hadamard(1),
    qml.PauliY(0) @ qml.Hadamard(1),
    qml.Hadamard(0) @ qml.Hadamard(1)
])
def test_get_ops_for_product(obs : Prod):
    step = DecomposeReadout()
    
    expected = [u for o in obs.operands for u in o.diagonalizing_gates()]
    results = step.get_ops_for_product(obs)
    assert len(results) == len(expected)
    for i, res in enumerate(results):
        res2 = expected[i]
        assert res == res2


def test_get_ops_for_product_edge_cases():
    step = DecomposeReadout()
    
    # three operands
    obs : Prod = qml.PauliX(0) @ qml.PauliZ(1) @ qml.Hadamard(2)
    results = step.get_ops_for_product(obs)
    solution = [u for o in obs.operands for u in o.diagonalizing_gates()]
    for i, r in enumerate(results):
        r2 = solution[i]
        assert r == r2

    # nested operands
    obs = qml.PauliX(0) @ (qml.PauliZ(1) @ qml.Hadamard(2))
    results = step.get_ops_for_product(obs)
    solution = [u for o in obs.operands for u in o.diagonalizing_gates()]
    for i, r in enumerate(results):
        r2 = solution[i]
        assert r == r2
    
    # two operands on same wires
    obs = qml.PauliX(0) @ qml.PauliY(0)
    with pytest.raises(ProcessingException):
        _ = step.get_ops_for_product(obs)
        
    # unsupported operand
    obs = qml.T(0) @ qml.PauliZ(1)
    with pytest.raises(ProcessingException):
        _ = step.get_ops_for_product(obs)


def test_execute(mock_get_ops_for_product):
    obs = [qml.Z(0), qml.X(0), qml.Y(0), qml.Hadamard(0)]
    step = DecomposeReadout()
    
    # X, Y, Z and H
    for observable in obs:
        tape = QuantumTape([], [qml.counts(observable)])
        diag = observable.diagonalizing_gates()
        tape = step.execute(tape)
        assert len(tape.operations) == len(diag) and len(tape.measurements) == 1
        for i, op in enumerate(tape.operations):
            op == diag[i]
        mock_get_ops_for_product.assert_not_called()
    
    # observable X @ Y
    obs = qml.X(0)
    mock_get_ops_for_product.return_value = [obs]
    tape = QuantumTape([], [qml.counts(qml.PauliZ(0) @ qml.PauliZ(1))])
    tape = step.execute(tape)
    assert len(tape.operations) == 1 and len(tape.measurements) == 1
    assert tape.operations[0] is obs
    mock_get_ops_for_product.assert_called_once()
    
    # double operand on 1 wire
    tape = QuantumTape([], [qml.counts(qml.PauliZ(0) @ qml.PauliZ(0))])
    with pytest.raises(ProcessingException):
        _ = step.execute(tape)

    # no observable
    tape = QuantumTape([], [qml.counts()])
    tape = step.execute(tape)
    assert tape.operations == []
    
    # unsupported observable
    tape = QuantumTape([], [qml.counts(qml.S(0))])
    with pytest.raises(ProcessingException):
        tape = step.execute(tape)
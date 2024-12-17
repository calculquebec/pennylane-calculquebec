import pennylane as qml
from pennylane.tape import QuantumTape
import numpy as np
from pennylane_calculquebec.processing.steps import DecomposeReadout
import pytest
from unittest.mock import patch


@pytest.fixture
def mock_get_ops_for_product():
    with patch("pennylane_calculquebec.processing.steps.DecomposeReadout.get_ops_for_product") as mock:
        yield mock

@pytest.mark.parametrize("obs, expected", [
    (qml.Z(0) @ qml.Z(1), [qml.Identity(0), qml.Identity(1)]),
    (qml.X(0) @ qml.Z(1), [qml.RY(np.pi / 2, 0), qml.Identity(1)]),
    (qml.Y(0) @ qml.Z(1), [qml.RX(-np.pi / 2, 0), qml.Identity(1)]),
    (qml.H(0) @ qml.Z(1), [qml.RY(np.pi / 4, 0), qml.Identity(1)]),
    (qml.Z(0) @ qml.X(1), [qml.Identity(0), qml.RY(np.pi / 2, 1)]),
    (qml.X(0) @ qml.X(1), [qml.RY(np.pi / 2, 0), qml.RY(np.pi / 2, 1)]),
    (qml.Y(0) @ qml.X(1), [qml.RX(-np.pi / 2, 0), qml.RY(np.pi / 2, 1)]),
    (qml.H(0) @ qml.X(1), [qml.RY(np.pi / 4, 0), qml.RY(np.pi / 2, 1)]),
    (qml.Z(0) @ qml.Y(1), [qml.Identity(0), qml.RX(-np.pi / 2, 1)]),
    (qml.X(0) @ qml.Y(1), [qml.RY(np.pi / 2, 0), qml.RX(-np.pi / 2, 1)]),
    (qml.Y(0) @ qml.Y(1), [qml.RX(-np.pi / 2, 0), qml.RX(-np.pi / 2, 1)]),
    (qml.H(0) @ qml.Y(1), [qml.RY(np.pi / 4, 0), qml.RX(-np.pi / 2, 1)]),
    (qml.Z(0) @ qml.H(1), [qml.Identity(0), qml.RY(np.pi / 4, 1)]),
    (qml.X(0) @ qml.H(1), [qml.RY(np.pi / 2, 0), qml.RY(np.pi / 4, 1)]),
    (qml.Y(0) @ qml.H(1), [qml.RX(-np.pi / 2, 0), qml.RY(np.pi / 4, 1)]),
    (qml.H(0) @ qml.H(1), [qml.RY(np.pi / 4, 0), qml.RY(np.pi / 4, 1)])
])
def test_get_ops_for_product(obs, expected):
    step = DecomposeReadout()
    
    results = step.get_ops_for_product(obs)
    assert len(results) == 2
    for i, res in enumerate(results):
        res2 = expected[i]
        assert res == res2


def test_get_ops_for_product_edge_cases():
    step = DecomposeReadout()
    
    # three operands
    obs = qml.X(0) @ qml.Z(1) @ qml.H(2)
    results = step.get_ops_for_product(obs)
    solution = [qml.RY(np.pi/2, 0), qml.Identity(1), qml.RY(np.pi/4, 2)]
    for i, r in enumerate(results):
        r2 = solution[i]
        assert r == r2

    # nested operands
    obs = qml.X(0) @ (qml.Z(1) @ qml.H(2))
    results = step.get_ops_for_product(obs)
    solution = [qml.RY(np.pi/2, 0), qml.Identity(1), qml.RY(np.pi/4, 2)]
    for i, r in enumerate(results):
        r2 = solution[i]
        assert r == r2
    
    # two operands on same wires
    obs = qml.X(0) @ qml.Y(0)
    results = step.get_ops_for_product(obs)
    solution = [qml.RY(np.pi/2, 0), qml.RX(-np.pi/2, 0)]
    for i, r in enumerate(results):
        r2 = solution[i]
        assert r == r2
        
    # unsupported operand
    obs = qml.T(0) @ qml.Z(1)
    with pytest.raises(ValueError):
        step.get_ops_for_product(obs)


def test_execute(mock_get_ops_for_product):
    step = DecomposeReadout()
    
    # observable Z
    tape = QuantumTape([], [qml.counts(qml.Z(0))])
    tape = step.execute(tape)
    assert len(tape.operations) == 1 and len(tape.measurements) == 1
    assert tape.operations[0] == qml.Identity(0)
    mock_get_ops_for_product.assert_not_called()
    
    # observable X
    tape = QuantumTape([], [qml.counts(qml.X(0))])
    tape = step.execute(tape)
    assert len(tape.operations) == 1 and len(tape.measurements) == 1
    assert tape.operations[0] == qml.RY(np.pi/2, 0)
    mock_get_ops_for_product.assert_not_called()
    
    # observable Y
    tape = QuantumTape([], [qml.counts(qml.Y(0))])
    tape = step.execute(tape)
    assert len(tape.operations) == 1 and len(tape.measurements) == 1
    assert tape.operations[0] == qml.RX(-np.pi/2, 0)
    mock_get_ops_for_product.assert_not_called()
    
    # observable H
    tape = QuantumTape([], [qml.counts(qml.H(0))])
    tape = step.execute(tape)
    assert len(tape.operations) == 1 and len(tape.measurements) == 1
    assert tape.operations[0] == qml.RY(np.pi/4, 0)
    mock_get_ops_for_product.assert_not_called()
    
    # observable Z @ Z
    mock_get_ops_for_product.return_value = ["success"]
    tape = QuantumTape([], [qml.counts(qml.Z(0) @ qml.Z(0))])
    tape = step.execute(tape)
    assert len(tape.operations) == 1 and len(tape.measurements) == 1
    assert tape.operations[0] == "success"
    mock_get_ops_for_product.assert_called_once()
    
    # pas d'observable
    tape = QuantumTape([], [qml.counts()])
    tape = step.execute(tape)
    assert tape.operations == []
    
    # observable non-supporté
    tape = QuantumTape([], [qml.counts(qml.S(0))])
    with pytest.raises(ValueError):
        tape = step.execute(tape)
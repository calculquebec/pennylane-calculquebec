import pytest
from unittest.mock import patch
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane_calculquebec.processing.steps.routing import Swaps, RoutingException
import networkx as nx


@pytest.fixture
def mock_circuit_graph():
    with patch("pennylane_calculquebec.processing.steps.routing.circuit_graph") as mock:
        yield mock


@pytest.fixture
def mock_machine_graph():
    with patch("pennylane_calculquebec.processing.steps.routing.machine_graph") as mock:
        yield mock


@pytest.fixture
def mock_r1_cz_fidelities():
    with patch(
        "pennylane_calculquebec.utility.graph.get_readout1_and_cz_fidelities"
    ) as mock:
        yield mock


def test_directly_connected(mock_machine_graph):
    mock_machine_graph.return_value = nx.Graph([(0, 1), (1, 2), (2, 3)])

    tape = QuantumTape(ops=[qml.CNOT([0, 1])])
    step = Swaps("yamaska")
    tape2 = step.execute(tape)
    assert all(op1 == op2 for op1, op2 in zip(tape.operations, tape2.operations))


def test_not_directly_connected(mock_machine_graph, mock_r1_cz_fidelities):
    mock_machine_graph.return_value = nx.Graph([(0, 1), (1, 2), (2, 3)])
    mock_r1_cz_fidelities.return_value = {
        "readoutState1Fidelity": {"0": 1, "1": 1, "2": 1},
        "czGateFidelity": {(0, 1): 1, (1, 2): 1, (2, 3): 1},
    }

    expected = [qml.SWAP([1, 2]), qml.CNOT([0, 1]), qml.SWAP([1, 2])]
    tape = QuantumTape(ops=[qml.CNOT([0, 2])])

    step = Swaps("yamaska")
    tape2 = step.execute(tape)

    assert [op1 == op2 for op1, op2 in zip(expected, tape2.operations)]


def test_inexistant_wire(mock_machine_graph, mock_r1_cz_fidelities):
    mock_machine_graph.return_value = nx.Graph([(0, 1), (2, 3)])
    mock_r1_cz_fidelities.return_value = {
        "readoutState1Fidelity": {"0": 1, "1": 1, "2": 1},
        "czGateFidelity": {(0, 1): 1, (2, 3): 1},
    }

    tape = QuantumTape(ops=[qml.CNOT([0, 2])])

    step = Swaps("yamaska")
    with pytest.raises(RoutingException):
        _ = step.execute(tape)

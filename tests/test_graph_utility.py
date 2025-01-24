import networkx as nx
import pennylane_calculquebec.utility.graph as g
import pennylane as qml
import networkx as nx
import pytest
from unittest.mock import patch
from pennylane_calculquebec.utility.api import keys
from pennylane.tape import QuantumTape

@pytest.fixture
def mock_broken_qubits_couplers():
    with patch("pennylane_calculquebec.utility.graph.get_broken_qubits_and_couplers") as mock:
        yield mock

@pytest.fixture
def mock_connectivity():
    with patch("pennylane_calculquebec.utility.graph.connectivity") as mock:
        yield mock

def test_find_biggest_group():
    # two groups, one is bigger
    graph = nx.Graph([(0, 1), (0, 2), (3, 4)])
    expected = [0, 1, 2]
    results = g.find_biggest_group(graph)
    assert all(a == b for a, b in zip(expected, results))

    # two groups, same size
    graph = nx.Graph([(0, 1), (2, 3)])
    expected = [0, 1]
    results = g.find_biggest_group(graph)
    assert all(a == b for a, b in zip(expected, results))

    # one group
    graph = nx.Graph([(0, 1)])
    expected = [0, 1]
    results = g.find_biggest_group(graph)
    assert all(a == b for a, b in zip(expected, results))

    # no group
    graph = nx.Graph([])
    expected = []
    results = g.find_biggest_group(graph)
    assert all(a == b for a, b in zip(expected, results))

def test_is_directly_connected():
    graph = nx.Graph([(0, 1), (1, 2), (3, 4)])
    # single qubit operation
    x = qml.PauliX(0)
    with pytest.raises(g.GraphException):
        g.is_directly_connected(x, graph)

    # wire is not mapped to qubit
    cx = qml.CNOT([6, 5])
    with pytest.raises(g.GraphException):
        g.is_directly_connected(cx, graph)

    # is not directly connected
    cx = qml.CNOT([2, 4])
    assert not g.is_directly_connected(cx, graph)

    # is directly connected
    cx = qml.CNOT([3, 4])
    assert g.is_directly_connected(cx, graph)

def test_circuit_graph():
    # typical use case
    tape = QuantumTape(ops = [qml.CNOT([0, 1]), qml.CZ([1, 2]), qml.SWAP([2, 3])])
    expected = [(0, 1), (1, 2), (2, 3)]
    results = g.circuit_graph(tape)
    assert len(expected) == results.number_of_edges()
    assert all(edge in expected for edge in results.edges)

    # no ops, 3 measurements
    tape = QuantumTape(ops = [], measurements=[qml.counts(wires=[0, 1, 2])])
    expected = [0, 1, 2]
    results = g.circuit_graph(tape)
    assert results.number_of_edges() == 0
    assert results.number_of_nodes() == len(expected)
    assert all(node in expected for node in results.nodes)

    # 3 1q ops
    tape = QuantumTape(ops = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)])
    expected = [0, 1, 2]
    results = g.circuit_graph(tape)
    assert results.number_of_edges() == 0
    assert results.number_of_nodes() == len(expected)
    assert all(node in expected for node in results.nodes)

    # 1 3q op
    tape = QuantumTape(ops = [qml.Toffoli([0, 1, 2])])
    with pytest.raises(g.GraphException):
        results = g.circuit_graph(tape)

    # no op
    tape = QuantumTape(ops = [])
    results = g.circuit_graph(tape)
    assert results.number_of_edges() == 0
    assert results.number_of_nodes() == 0
    assert all(node in expected for node in results.nodes)

def test_machine_graph(mock_broken_qubits_couplers, mock_connectivity):
    mock_connectivity.return_value = {
        keys.QUBITS : [ 0, 1, 2, 3 ],   
        keys.COUPLERS : {
            "0": (0, 1),
            "1": (1, 2),
            "2": (2, 3)
        }
    }

    mock_broken_qubits_couplers.return_value = {
        keys.QUBITS : [],
        keys.COUPLERS : [(1, 2)]
    }

    expected = [(0, 1), (1, 2), (2, 3)]
    results = g.machine_graph(False, 0.5, 0.5)
    assert all(a == b for a, b in zip(expected, list(results.edges)))

    expected = [(0, 1), (2, 3)]
    results = g.machine_graph(True, 0.5, 0.5)
    assert all(a == b for a, b in zip(expected, list(results.edges)))

    expected = [(1, 2), (2, 3)]
    results = g.machine_graph(False, 0.5, 0.5, [0])
    assert all(a == b for a, b in zip(expected, list(results.edges)))

    expected = [(0, 1), (2, 3)]
    results = g.machine_graph(False, 0.5, 0.5, [], [(1, 2)])
    assert all(a == b for a, b in zip(expected, list(results.edges)))

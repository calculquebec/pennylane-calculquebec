import networkx as nx
import pennylane_calculquebec.utility.graph as g
import pennylane as qml
import networkx as nx
import pytest

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

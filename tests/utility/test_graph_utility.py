import networkx as nx
import pennylane_calculquebec.utility.graph as g
import pennylane as qml
import networkx as nx
import pytest
from unittest.mock import patch
from pennylane_calculquebec.utility.api import keys
from pennylane.tape import QuantumTape


@pytest.fixture
def mock_readout1_cz_fidelities():
    with patch(
        "pennylane_calculquebec.utility.graph.get_readout1_and_cz_fidelities"
    ) as mock:
        yield mock


@pytest.fixture
def mock_broken_qubits_couplers():
    with patch(
        "pennylane_calculquebec.utility.graph.get_broken_qubits_and_couplers"
    ) as mock:
        yield mock


@pytest.fixture
def mock_connectivity():
    with patch("pennylane_calculquebec.utility.graph.get_connectivity") as mock:
        yield mock


@pytest.fixture
def mock_calculate_score():
    with patch("pennylane_calculquebec.utility.graph.calculate_score") as mock:
        yield mock


@pytest.fixture
def mock_shortest_path():
    with patch("pennylane_calculquebec.utility.graph.shortest_path") as mock:
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
    tape = QuantumTape(ops=[qml.CNOT([0, 1]), qml.CZ([1, 2]), qml.SWAP([2, 3])])
    expected = [(0, 1), (1, 2), (2, 3)]
    results = g.circuit_graph(tape)
    assert len(expected) == results.number_of_edges()
    assert all(edge in expected for edge in results.edges)

    # no ops, 3 measurements
    tape = QuantumTape(ops=[], measurements=[qml.counts(wires=[0, 1, 2])])
    expected = [0, 1, 2]
    results = g.circuit_graph(tape)
    assert results.number_of_edges() == 0
    assert results.number_of_nodes() == len(expected)
    assert all(node in expected for node in results.nodes)

    # 3 1q ops
    tape = QuantumTape(ops=[qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)])
    expected = [0, 1, 2]
    results = g.circuit_graph(tape)
    assert results.number_of_edges() == 0
    assert results.number_of_nodes() == len(expected)
    assert all(node in expected for node in results.nodes)

    # 1 3q op
    tape = QuantumTape(ops=[qml.Toffoli([0, 1, 2])])
    with pytest.raises(g.GraphException):
        results = g.circuit_graph(tape)

    # no op
    tape = QuantumTape(ops=[])
    results = g.circuit_graph(tape)
    assert results.number_of_edges() == 0
    assert results.number_of_nodes() == 0
    assert all(node in expected for node in results.nodes)


def test_machine_graph(mock_broken_qubits_couplers, mock_connectivity):
    mock_connectivity.return_value = {"0": (0, 1), "1": (1, 2), "2": (2, 3)}

    mock_broken_qubits_couplers.return_value = {
        keys.QUBITS: [],
        keys.COUPLERS: [(1, 2)],
    }

    expected = [(0, 1), (1, 2), (2, 3)]
    results = g.machine_graph("yamaska", False, 0.5, 0.5)
    assert all(a == b for a, b in zip(expected, list(results.edges)))

    expected = [(0, 1), (2, 3)]
    results = g.machine_graph("yamaska", True, 0.5, 0.5)
    assert all(a == b for a, b in zip(expected, list(results.edges)))

    expected = [(1, 2), (2, 3)]
    results = g.machine_graph("yamaska", False, 0.5, 0.5, [0])
    assert all(a == b for a, b in zip(expected, list(results.edges)))

    expected = [(0, 1), (2, 3)]
    results = g.machine_graph("yamaska", False, 0.5, 0.5, [], [(1, 2)])
    assert all(a == b for a, b in zip(expected, list(results.edges)))


def test_find_isomorphism():
    # isomorphism exists
    subgraph = nx.Graph([(0, 1), (0, 2)])
    graph = nx.Graph([(4, 5), (5, 6)])

    expected = {0: 5, 1: 4, 2: 6}
    results = g._find_isomorphisms(subgraph, graph)

    assert len(expected.items()) == len(results.items())
    assert all(a == b for a, b in zip(expected.items(), results.items()))

    # isomoprhism doesn't exist
    subgraph = nx.Graph([(0, 1), (1, 2)])
    graph = nx.Graph([(4, 5)])

    results = g._find_isomorphisms(subgraph, graph)

    assert results == None

    # there are no links
    subgraph = nx.Graph()
    subgraph.add_nodes_from((0, 1, 2))
    graph = nx.Graph([(4, 5), (5, 6)])

    expected = {0: 4, 1: 5, 2: 6}
    results = g._find_isomorphisms(subgraph, graph)

    assert len(expected) == len(results)
    assert all(a == b for a, b in zip(expected.items(), results.items()))


def test_find_largest_common_subgraph_vf2():
    subgraph = nx.Graph([(0, 1), (1, 2), (0, 3)])
    graph = nx.Graph([(0, 1), (1, 2)])

    expected = {0: 1, 1: 0, 3: 2}
    results = g.find_largest_common_subgraph_vf2(subgraph, graph)

    assert len(expected.items()) == len(results.items())
    assert all(a == b for a, b in zip(expected.items(), results.items()))


def test_find_largest_common_subgraph_ismags():
    subgraph = nx.Graph([(0, 1), (1, 2), (0, 3)])
    graph = nx.Graph([(0, 1), (1, 2)])

    expected = {0: 0, 1: 1, 2: 2}
    results = g.find_largest_common_subgraph_ismags(subgraph, graph)

    assert len(expected.items()) == len(results.items())
    assert all(a == b for a, b in zip(expected.items(), results.items()))


def test_shortest_path(mock_readout1_cz_fidelities):
    mock_readout1_cz_fidelities.return_value = {
        keys.READOUT_STATE_1_FIDELITY: {
            "0": 1,
            "1": 1,
            "2": 1,
            "3": 1,
            "4": 1,
            "5": 1,
            "6": 1,
            "7": 1,
            "8": 1,
        },
        keys.CZ_GATE_FIDELITY: {
            (0, 1): 1,
            (1, 2): 1,
            (2, 3): 1,
            (5, 6): 1,
            (2, 4): 1,
            (4, 5): 1,
            (1, 6): 1,
            (7, 8): 1,
        },
    }

    graph = nx.Graph([(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (1, 6), (5, 6), (7, 8)])

    # path exists
    start = 6
    end = 4
    expected = [6, 5, 4]
    results = g.shortest_path(start, end, graph, "yamaska")
    assert len(expected) == len(results)
    assert all(a == b for a, b in zip(expected, results))

    # excluded nodes changes path
    expected = [6, 1, 2, 4]
    results = g.shortest_path(start, end, graph, "yamaska", excluding=[5])
    assert len(expected) == len(results)
    assert all(a == b for a, b in zip(expected, results))

    # prioritized node changes path
    expected = [6, 1, 2, 4]
    results = g.shortest_path(start, end, graph, "yamaska", prioritized_nodes=[1, 2])
    assert len(expected) == len(results)
    assert all(a == b for a, b in zip(expected, results))

    # path doesn't exist
    start = 4
    end = 8
    results = g.shortest_path(start, end, graph, "yamaska")
    assert results == None


def test_find_best_neighbour(mock_calculate_score):
    # return the number of the node as cost (for test)
    mock_calculate_score.side_effect = lambda a, b, c, d: a

    graph = nx.Graph([(0, 1), (0, 2), (0, 3), (0, 4)])
    graph.add_node(5)
    expected = 4

    results = g.find_best_neighbour(0, graph, "yamaska")
    assert results == expected
    mock_calculate_score.assert_called()

    with pytest.raises(g.GraphException):
        g.find_best_neighbour(5, graph, "yamaska")

    with pytest.raises(g.GraphException):
        g.find_best_neighbour(6, graph, "yamaska")


def test_find_best_wire(mock_calculate_score):
    # return the number of the node as cost (for test)
    mock_calculate_score.side_effect = lambda a, b, c, d: a

    graph = nx.Graph([(0, 1), (0, 2), (0, 3), (0, 4)])
    expected = 4

    results = g.find_best_wire(graph, "yamaska")
    assert results == expected
    mock_calculate_score.assert_called()

    expected = 3
    results = g.find_best_wire(graph, "yamaska", [4])
    assert results == expected


def test_find_closest_wire(mock_readout1_cz_fidelities):
    mock_readout1_cz_fidelities.return_value = {
        keys.READOUT_STATE_1_FIDELITY: {
            "0": 1,
            "1": 1,
            "2": 1,
            "3": 1,
            "4": 1,
            "5": 1,
            "6": 1,
            "7": 1,
            "8": 1,
        },
        keys.CZ_GATE_FIDELITY: {
            (0, 1): 1,
            (1, 2): 1,
            (2, 3): 1,
            (5, 6): 1,
            (2, 4): 1,
            (4, 5): 1,
            (1, 6): 1,
            (7, 8): 1,
        },
    }

    graph = nx.Graph([(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (1, 6), (5, 6), (7, 8)])

    result = g.find_closest_wire(1, graph, "yamaska")
    expected = 0
    assert result == expected

    result = g.find_closest_wire(1, graph, "yamaska", [0])
    expected = 2
    assert result == expected

    result = g.find_closest_wire(1, graph, "yamaska", [0, 2, 6])
    expected = 3
    assert result == expected

    graph = nx.Graph()
    graph.add_node(1)
    with pytest.raises(g.GraphException):
        g.find_closest_wire(1, graph, "yamaska")

    graph.remove_node(1)
    graph.add_nodes_from([0, 2, 3])
    with pytest.raises(g.GraphException):
        g.find_closest_wire(1, graph, "yamaska")


def test_path_length():
    path = None
    expected = g.MAX_INT
    assert expected == g.path_length(path)

    path = [0, 1, 2]
    expected = len(path)
    assert expected == g.path_length(path)


def test_node_with_shortest_path_from_selection(mock_readout1_cz_fidelities):
    mock_readout1_cz_fidelities.return_value = {
        keys.READOUT_STATE_1_FIDELITY: {
            "0": 1,
            "1": 1,
            "2": 1,
            "3": 1,
            "4": 1,
            "5": 1,
            "6": 1,
            "7": 1,
            "8": 1,
        },
        keys.CZ_GATE_FIDELITY: {
            (0, 1): 1,
            (1, 2): 1,
            (2, 3): 1,
            (5, 6): 1,
            (2, 4): 1,
            (4, 5): 1,
            (1, 6): 1,
            (7, 8): 1,
        },
    }

    graph = nx.Graph([(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (1, 6), (5, 6), (7, 8)])

    source = 0

    selection = [0, 1, 2, 3]
    expected = 1
    result = g.node_with_shortest_path_from_selection(
        source, selection, graph, "yamaska"
    )

    assert result == expected

    selection = [0, 2, 3]
    expected = 2
    result = g.node_with_shortest_path_from_selection(
        source, selection, graph, "yamaska"
    )

    assert result == expected

    selection = [0, 3]
    expected = 3
    result = g.node_with_shortest_path_from_selection(
        source, selection, graph, "yamaska"
    )

    assert result == expected

    selection = [0]

    with pytest.raises(g.GraphException):
        g.node_with_shortest_path_from_selection(source, selection, graph, "yamaska")


def test_calculate_score(mock_readout1_cz_fidelities):
    # calculate score without benchmarking is always zero
    assert abs(g.calculate_score(None, None, "yamaska", False) - 1) < 1e-5

    graph = nx.Graph([(0, 1), (1, 2)])

    mock_readout1_cz_fidelities.return_value = {
        "readoutState1Fidelity": {
            "0": 0.1,
            "1": 0.1,
            "2": 0.1,
        },
        "czGateFidelity": {
            (0, 1): 0.5,
            (1, 2): 0.5,
        },
    }

    # sum(neighbour readout 1) / n + readout1 + sum(neighbour cz) / n
    # 0.1 + 0.1 + 0.5 = 0.7 in any case
    expected = 0.7
    result = g.calculate_score(1, graph, "yamaska")
    assert abs(result - expected) < 1e-5

    # 1 * 0.1 / 1 + 0.1 + 0.5
    expected = 0.7
    result = g.calculate_score(0, graph, "yamaska")
    assert abs(result - expected) < 1e-5

    # specific case
    mock_readout1_cz_fidelities.return_value = {
        "readoutState1Fidelity": {
            "0": 0.1,
            "1": 0.2,
            "2": 0.3,
        },
        "czGateFidelity": {
            (0, 1): 0.4,
            (1, 2): 0.5,
        },
    }
    expected = 0.85
    result = g.calculate_score(1, graph, "yamaska")
    assert abs(result - expected) < 1e-5

    expected = 1
    result = g.calculate_score(2, graph, "yamaska")
    assert abs(result - expected) < 1e-5

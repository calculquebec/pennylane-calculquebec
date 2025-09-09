"""
Contains graph algorithm utility functions (mainly for placement and routing steps)
"""

from pennylane.tape import QuantumTape
from pennylane.operation import Operation
import networkx as nx
from networkx.algorithms.isomorphism.ismags import ISMAGS
from typing import Tuple
from copy import deepcopy
from itertools import combinations
from pennylane_calculquebec.monarq_data import (
    get_connectivity,
    get_broken_qubits_and_couplers,
    get_readout1_and_cz_fidelities,
)
from pennylane_calculquebec.utility.api import keys
from networkx.exception import NetworkXNoPath
import sys

MAX_INT = sys.maxsize


class GraphException(Exception):
    pass


def find_biggest_group(graph: nx.Graph) -> list:
    """Returns the biggest array of connected components in the graph

    Args:
        graph (nx.Graph): the graph for which you want to find the biggest group

    Returns:
        list: the biggest group
    """
    if graph.number_of_edges() == 0:
        return []
    return max(nx.connected_components(graph), key=len)


def is_directly_connected(operation: Operation, machine_topology: nx.Graph) -> bool:
    """
    Checks if a 2 qubits operation is mapped to a coupler in the machine

    Args:
        operation (Operation) : a two qubits operation
        machine_topology (Graph) : the machine's graph
    Returns:
        bool: does the graph have a link that maps given operation?
    """
    if len(operation.wires) < 2:
        raise GraphException(f"{operation.name} is not a 2 qubit operation")
    if (
        operation.wires[1] not in machine_topology.nodes
        or operation.wires[0] not in machine_topology.nodes
    ):
        raise GraphException(
            f"operation {operation} is not properly mapped to physical qubits"
        )

    return operation.wires[1] in machine_topology.neighbors(operation.wires[0])


def circuit_graph(tape: QuantumTape) -> nx.Graph:
    """
    builds a bidirectional graph from the two qubits gates in a circuit

    Args:
        tape (QuantumTape) : QuantumTape, a tape representing the quantum circuit

    Returns:
        nx.Graph: a graph representing the connections between the wires in the circuit
    """
    links: list[Tuple[int, int]] = []

    for operation in tape.operations:
        if len(operation.wires) > 2:
            raise GraphException(
                "All operations in the circuit should be using <= 2 wires"
            )
        if len(operation.wires) < 2:
            continue
        toAdd = (operation.wires[0], operation.wires[1])
        links.append(toAdd)
    graph = nx.Graph(set(links))
    graph.add_nodes_from([wire for wire in tape.wires if wire not in graph.nodes])
    return graph


def machine_graph(
    machine_name,
    use_benchmark,
    q1Acceptance,
    q2Acceptance,
    excluded_qubits=[],
    excluded_couplers=[],
) -> nx.Graph:
    """
    builds a bidirectional graph from the qubits and coupler of a machine

    Args:
        machine_name (str) : the quantum machine's name
        use_benchmark (bool) : should we check qubit and coupler fidelities?
        q1Acceptance (float) : at what fidelity is a qubit considered broken?
        q2Acceptance (float) : at what fidelity is a coupler considered broken?
        excluded_qubits (list[int]) : which qubits should we avoid?
        excluded_couplers (list[list[int]]) : which couplers should we avoid?

    Returns:
        nx.Graph : a graph representing the machine's topology
    """
    broken_qubits_and_couplers = (
        get_broken_qubits_and_couplers(q1Acceptance, q2Acceptance, machine_name)
        if use_benchmark
        else None
    )
    broken_nodes = (
        [qubit for qubit in broken_qubits_and_couplers[keys.QUBITS]]
        if use_benchmark
        else []
    )
    broken_nodes += [qubit for qubit in excluded_qubits if qubit not in broken_nodes]

    broken_couplers = (
        [coupler for coupler in broken_qubits_and_couplers[keys.COUPLERS]]
        if use_benchmark
        else []
    )

    # add excluded couplers to couplers
    broken_couplers += [
        coupler
        for coupler in excluded_couplers
        if not any(
            [
                broken_coupler[0] == coupler[0]
                and broken_coupler[1] == coupler[1]
                or broken_coupler[1] == coupler[0]
                and broken_coupler[0] == coupler[1]
                for broken_coupler in broken_couplers
            ]
        )
    ]
    links = [
        (coupler[0], coupler[1])
        for coupler in get_connectivity(machine_name, use_benchmark).values()
    ]

    return nx.Graph(
        [
            link
            for link in links
            if link[0] not in broken_nodes
            and link[1] not in broken_nodes
            and link not in broken_couplers
            and list(reversed(link)) not in broken_couplers
        ]
    )


def _find_isomorphisms(circuit: nx.Graph, machine: nx.Graph) -> dict[int, int]:
    """
    finds an isomorphism between two graphs using VF2 algorith

    Args:
        circuit (Graph) : the graph of the circuit
        machine (Graph) : the graph of the machine
    Returns:
        dict[int, int] : a mapping between the circuit's wires and the machines qubits
    """
    vf2 = nx.isomorphism.GraphMatcher(machine, circuit)
    for mono in vf2.subgraph_monomorphisms_iter():
        return {v: k for k, v in mono.items()}
    return None


def find_largest_common_subgraph_vf2(circuit: nx.Graph, machine: nx.Graph):
    """
    Uses vf2 and combinations to find the largest common graph between two graphs

    Args:
        circuit (Graph) : the graph of the circuit
        machine (Graph) : the graph of the machine
    Returns:
        dict[int, int] : a mapping between the circuit's wires and the machines qubits
    """
    edges = [e for e in circuit.edges]
    if len(edges) <= 0:
        return _find_isomorphisms(circuit, machine)

    for i in reversed(range(len(edges) + 1)):
        for comb in combinations(edges, i):
            result = _find_isomorphisms(nx.Graph(comb), machine)
            if result:
                return result


def find_largest_common_subgraph_ismags(circuit: nx.Graph, machine: nx.Graph):
    """
    Uses IMAGS to find the largest common graph between two graphs

    Args:
        circuit (Graph) : the graph of the circuit
        machine (Graph) : the graph of the machine
    Returns:
        dict[int, int] : a mapping between the circuit's wires and the machines qubits
    """
    ismags = ISMAGS(machine, circuit)
    for mapping in ismags.largest_common_subgraph():
        return (
            {v: k for (k, v) in mapping.items()}
            if mapping is not None and len(mapping) > 0
            else mapping
        )


def shortest_path(
    start: int,
    end: int,
    graph: nx.Graph,
    machine_name: str,
    excluding: list[int] = [],
    prioritized_nodes: list[int] = [],
    use_benchmark=True,
):
    """
    find the shortest path between node start and end

    Args :
        start : start node
        end : end node
        graph : the graph to find a path in
        machine_name (str) : the quantum machine's name
        excluding : nodes we dont want to use
        prioritized_nodes : nodes we want to use if possible
        use_benchmark : should we consider fidelities in choosing the paths?
        Returns:
            list[int] : the shortest path from start to end
    """
    r1_cz_fidelities = (
        get_readout1_and_cz_fidelities(machine_name) if use_benchmark else {}
    )
    g_copy = deepcopy(graph)
    g_copy.remove_nodes_from(excluding)

    def weight(source_node, dest_node):
        """
        this function is used to determine the cost of a link
        it is determined by the error of the source + the error of the coupler + the error of the destination
        """

        if not use_benchmark:
            return 1  # return default value of one if we should not use benchmarks

        # there should be only one cz weight from 0 to 1
        source_dest_cz = [
            dest_fidelity
            for coupler, dest_fidelity in r1_cz_fidelities[
                keys.CZ_GATE_FIDELITY
            ].items()
            if source_node in coupler and dest_node in coupler
        ]
        source_readout1 = r1_cz_fidelities[keys.READOUT_STATE_1_FIDELITY][
            str(source_node)
        ]
        dest_readout1 = r1_cz_fidelities[keys.READOUT_STATE_1_FIDELITY][str(dest_node)]

        # this node has no coupler. we should never chose it!
        if len(source_dest_cz) < 1:
            return MAX_INT

        # weight corresponds to the cz error (ie 1 - fidelity)
        # we add one at the end so that if the node is prioritized,
        # it doesn't become negative when it is subtracted one
        w = 3 - (source_dest_cz[0] + source_readout1 + dest_readout1) + 1

        if source_node in prioritized_nodes or dest_node in prioritized_nodes:
            return w - 1
        return w

    try:
        return nx.astar_path(g_copy, start, end, weight=lambda u, v, _: weight(u, v))
    except NetworkXNoPath:
        return None


def find_best_neighbour(
    wire, topology: nx.Graph, machine_name: str, use_benchmark=True
):
    """
    Finds the neighbour to a node which has the highest mean fidelity

    Args:
        wire (int) : the node for which we want to find the neighbour
        topology (Graph) : the graph on which we are searching for neighbours
        machine_name (str) : the quantum machine's name
        use_benchmark (bool) : should we use fidelities?

    Returns:
        int : the neighbour with best score
    """
    if wire not in topology:
        raise GraphException(f"node {wire} is not in graph")

    neigh = list(topology.neighbors(wire))

    if len(neigh) <= 0:
        raise GraphException(f"there are no neighbour to node {wire}")

    return max(
        neigh, key=lambda n: calculate_score(n, topology, machine_name, use_benchmark)
    )


def find_best_wire(
    graph: nx.Graph, machine_name: str, excluded: list[int] = [], use_benchmark=True
):
    """
    find node with highest degree in graph

    Args:
        graph (Graph) : the graph from which we want to find the best wire
        machine_name (str) : the quantum machine's name
        excluded (list[int]) : wires we want to skip
        use_benchmark (bool) : should we use fidelities?
    Returns:
        int : the wire with best score
    """
    graph_copy = deepcopy(graph)
    graph_copy.remove_nodes_from(excluded)
    return max(
        [node for node in graph_copy.nodes],
        key=lambda other: calculate_score(
            other, graph_copy, machine_name, use_benchmark
        ),
    )


def find_closest_wire(
    source: int,
    machine_graph: nx.Graph,
    machine_name: str,
    excluding: list[int] = [],
    prioritized: list[int] = [],
    use_benchmark=True,
):
    """
    find node in graph that is closest to given node, while skipping nodes from the excluding list

    Args:
        source (int) : the node from which we are searching
        machine_graph (Graph) : the graph of the machine
        excluding (list[int]) : the nodes that we want to skip
        machine_name (str) : the quantum machine's name
        prioritized (list[int]) : nodes that should be in the path if possible
        use_benchmark (bool) : should we use qubit fidelities?
    Returns:
        int : the wire that has the smallest path from source
    """
    if source not in machine_graph:
        raise GraphException(f"node {source} doesn't exist in the machine's graph")

    nodes = [
        node for node in machine_graph.nodes if node not in excluding and node != source
    ]

    if len(nodes) <= 0:
        raise GraphException(f"Impossible to find closest wire for {source}")

    return min(
        nodes,
        key=lambda dest: path_length(
            shortest_path(
                source,
                dest,
                machine_graph,
                machine_name,
                prioritized_nodes=prioritized,
                use_benchmark=use_benchmark,
            )
        ),
    )


def path_length(path):
    """the length of a path

    Args:
        path (list[int]): the path to find the length of

    Returns:
        int: the length of a given path
    """
    if path is None:
        return MAX_INT
    return len(path)


def node_with_shortest_path_from_selection(
    source: int,
    selection: list[int],
    graph: nx.Graph,
    machine_name: str,
    use_benchmark=True,
):
    """
    find the unmapped node node in graph minus mapped nodes that has shortest path to given source node

    Args:
        source (int) : the source node
        selection (list[int]) : a selection of nodes to consider in operation
        graph (Graph) : the graph to work on
        machine_name (str) : the name of the machine. Usually yukon or yamaska
        use_benchmark (bool) : should we use real benchmarks for this operation?
    Returns:
        int : the closest node to source that was contained in selection
    """

    nodes_minus_source = [node for node in selection if node != source]

    if len(nodes_minus_source) <= 0:
        raise GraphException(
            "There are not enough nodes for this circuit to run on the machine"
        )

    return min(
        nodes_minus_source,
        key=lambda n: path_length(
            shortest_path(source, n, graph, machine_name, use_benchmark=use_benchmark)
        ),
    )


def calculate_score(
    source: int, graph: nx.Graph, machine_name: str, use_benchmark=True
) -> float:
    """Defines a score for a node by using cz fidelities on neighbouring couplers and state 1 readout fidelity\n
    the bigger the better

    Args:
        source (int): the node you want to define a cost for
        graph (nx.Graph): the graph in which the node you want to define a cost for is
        machine_name (str) : the quantum machine's name
        use_benhmark (bool) : should we use real benchmark for this operation?

    Returns:
        float : a cost, where the highest cost is the best one.
    """

    if not use_benchmark:
        return 1  # score should always be the same if we don't use benchmarks

    fidelities = get_readout1_and_cz_fidelities(machine_name)
    neighbours = [n for n in graph.neighbors(source)]

    source_readout1 = fidelities[keys.READOUT_STATE_1_FIDELITY][str(source)]

    if len(neighbours) <= 0:
        return source_readout1

    all_cz = fidelities[keys.CZ_GATE_FIDELITY]
    adjacent_cz = [
        all_cz[f] for f in all_cz if source in f and any(n in f for n in neighbours)
    ]
    adjacent_readout1 = [
        fidelities[keys.READOUT_STATE_1_FIDELITY][str(n)] for n in neighbours
    ]

    # mean readout1 of neighbours + mean cz of neighbours + readout1 of source (value from 0 to 3)
    return (
        sum(adjacent_cz) / len(neighbours)
        + source_readout1
        + sum(adjacent_readout1) / len(neighbours)
    )

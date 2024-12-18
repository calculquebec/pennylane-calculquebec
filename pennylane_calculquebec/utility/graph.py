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
from pennylane_calculquebec.monarq_data import connectivity, get_broken_qubits_and_couplers, get_readout1_and_cz_fidelities
from pennylane_calculquebec.utility.api import keys

def find_biggest_group(graph : nx.Graph) -> list:
    """Returns the biggest array of connected components in the graph

    Args:
        graph (nx.Graph): the graph for which you want to find the biggest group

    Returns:
        list: the biggest group
    """
    return max(nx.connected_components(graph), key=len)

def is_directly_connected(op : Operation, machine_topology : nx.Graph) -> bool:
    return op.wires[1] in machine_topology.neighbors(op.wires[0])

def circuit_graph(tape : QuantumTape) -> nx.Graph:
    """
    input : 
    tape : QuantumTape, a tape representing the quantum circuit

    output : nx.Graph, a graph representing the connections between the wires in the circuit
    """
    links : list[Tuple[int, int]] = []

    for op in tape.operations:
        if len(op.wires) != 2:
            continue
        toAdd = (op.wires[0], op.wires[1])
        links.append(toAdd)
    g = nx.Graph(set(links))
    g.add_nodes_from([w for w in tape.wires if w not in g.nodes])
    return g

def machine_graph(use_benchmark, q1Acceptance, q2Acceptance, excluded_qubits = [], excluded_couplers = []):
    
    broken_qubits_and_couplers = get_broken_qubits_and_couplers(q1Acceptance, q2Acceptance) if use_benchmark else None
    broken_nodes = [q for q in broken_qubits_and_couplers[keys.qubits]] if use_benchmark else []
    broken_nodes += [q for q in excluded_qubits if q not in broken_nodes]
    
    broken_couplers = [q for q in broken_qubits_and_couplers[keys.couplers]] if use_benchmark else []
    broken_couplers += [q for q in excluded_couplers if not any([b[0] == q[0] and b[1] == q[1] or b[1]== q[0] and b[0] == q[1] for b in broken_couplers])]
    links = [(v[0], v[1]) for (_, v) in connectivity[keys.couplers].items()]
    
    return nx.Graph([i for i in links if i[0] not in broken_nodes and i[1] not in broken_nodes \
            and i not in broken_couplers and list(reversed(i)) not in broken_couplers])

def _find_isomorphisms(circuit : nx.Graph, machine : nx.Graph) -> dict[int, int]:
    vf2 = nx.isomorphism.GraphMatcher(machine, circuit)
    for mono in vf2.subgraph_monomorphisms_iter():
       return {v : k for k, v in mono.items()}
    return None

def find_largest_subgraph_isomorphism_vf2(circuit : nx.Graph, machine : nx.Graph):
    """
    Uses vf2 and combinations to find the largest common graph between two graphs
    """
    edges = [e for e in circuit.edges]
    for i in reversed(range(len(edges) + 1)):
        for comb in combinations(edges, i):
            result = _find_isomorphisms(nx.Graph(comb), machine)
            if result: return result

def find_largest_subgraph_isomorphism_imags(circuit : nx.Graph, machine : nx.Graph):
    """
    Uses IMAGS to find the largest common graph between two graphs
    """
    ismags = ISMAGS(machine, circuit)
    for mapping in ismags.largest_common_subgraph():
        return {v:k for (k, v) in mapping.items()} if mapping is not None and len(mapping) > 0 else mapping

def shortest_path(a : int, b : int, graph : nx.Graph, excluding : list[int] = [], prioritized_nodes : list[int] = [], use_benchmark=True):
    """
    find the shortest path between node a and b

    Args :
        a : start node
        b : end node
        graph : the graph to find a path in
        excluding : nodes we dont want to use
        prioritized_nodes : nodes we want to use if possible
    """
    r1_cz_fidelities = get_readout1_and_cz_fidelities() if use_benchmark else {}
    g_copy = deepcopy(graph)
    g_copy.remove_nodes_from(excluding)

    def weight(node_u, node_v):
        if not use_benchmark: 
            return 1
    
        weights = [v for k, v in r1_cz_fidelities[keys.czGateFidelity].items() if node_u in k and node_v in k]
        r1_node_u = r1_cz_fidelities[keys.readoutState1Fidelity][str(node_u)]
        r1_node_v = r1_cz_fidelities[keys.readoutState1Fidelity][str(node_v)]
        
        if len(weights) < 1:
            return 10
        w = 4 - weights[0] - r1_node_u - r1_node_v
        
        if node_u in prioritized_nodes or node_v in prioritized_nodes:
            return w - 1
        return w
    
    return nx.astar_path(g_copy, a, b, weight = lambda u, v, _: weight(u, v))

def find_best_neighbour(wire, topology : nx.Graph, use_benchmark = True):
    """
    
    """
    neigh = list(topology.neighbors(wire))
    return max(neigh, key = lambda n : calculate_score(n, topology, use_benchmark))
        

def find_best_wire(graph : nx.Graph, excluded : list[int] = [], use_benchmark = True):
    """
    find node with highest degree in graph
    """
    g = deepcopy(graph)
    g.remove_nodes_from(excluded)
    return max([n for n in g.nodes], key=lambda n: calculate_score(n, g, use_benchmark))

def find_closest_wire(start : int, machine_graph : nx.Graph, excluding : list[int] = [], prioritized : list[int] = [], use_benchmark = True):
    """
    find node in graph that is closest to given node, not considering arbitrary excluding list
    """
    nodes = [n for n in machine_graph if n not in excluding]
    return min(nodes, key=lambda end: len(shortest_path(start, end, 
                                                      machine_graph, 
                                                      excluding=excluding, 
                                                      prioritized_nodes=prioritized, 
                                                      use_benchmark=use_benchmark)))


def node_with_shortest_path_from_selection(source : int, selection : list[int], graph : nx.Graph, use_benchmark = True):
    """
    find the unmapped node node in graph minus mapped nodes that has shortest path to given source node
    """
    # all_unmapped_nodes = [n for n in graph.nodes if n not in mapping and n != source]
    # mapping_minus_source = [n for n in mapping if n != source]

    nodes_minus_source = [node for node in selection if node != source]
    return min(nodes_minus_source, key=lambda n: len(shortest_path(source, n, graph, use_benchmark=use_benchmark)))
    # return min(all_unmapped_nodes, key = lambda n : len(_shortest_path(source, n, graph, mapping_minus_source)))

def calculate_score(source : int, graph : nx.Graph, use_benchmark = True) -> float:
    """Defines a score for a node by using cz fidelities on neighbouring couplers and state 1 readout fidelity\n
    the bigger the better

    Args:
        source (int): the node you want to define a cost for
        graph (nx.Graph): the graph in which the node you want to define a cost for is

    Returns:
        float : a cost, where the highest cost is the best one.
    """
    
    if not use_benchmark:
        return 1
    
    fidelities = get_readout1_and_cz_fidelities()
    neighbours = [n for n in graph.neighbors(source)]
    if len(neighbours) <= 0:
        return 0
    
    r1 = fidelities[keys.readoutState1Fidelity][str(source)]
    
    all_cz = fidelities[keys.czGateFidelity]
    cz = [all_cz[f] for f in all_cz if source in f and any(n in f for n in neighbours)]
    n_r1 = [fidelities[keys.readoutState1Fidelity][str(n)] for n in neighbours]
    return sum(cz)/len(neighbours) + r1 + sum(n_r1)/len(neighbours)

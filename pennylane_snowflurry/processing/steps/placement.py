"""
Contains placement pre-processing steps
"""

from pennylane.tape import QuantumTape
import pennylane_snowflurry.utility.graph as graph_util
from pennylane_snowflurry.processing.interfaces import PreProcStep


class Placement(PreProcStep):
    """
    base class for any placement algorithm. 
    """
    def __init__(self, use_benchmark = True, q1_acceptance = 0.5, q2_acceptance = 0.5, excluded_qubits=[], excluded_couplers=[]):
        """constructor for placement algorithms

        Args:
            use_benchmark (bool, optional): should we use benchmarks during placement? Defaults to True.
            q1_acceptance (float, optional): what is the level of acceptance for state 1 readout? Defaults to 0.5.
            q2_acceptance (float, optional): what is the level of acceptance for cz fidelity? Defaults to 0.5.
            excluded_qubits (list, optional): what qubits should we exclude from the mapping? Defaults to [].
            excluded_couplers (list, optional): what couplers should we exclude from the mapping? Defaults to [].
        """
        self.use_benchmark = use_benchmark
        self.q1_acceptance = q1_acceptance
        self.q2_acceptance = q2_acceptance
        self.excluded_qubits = excluded_qubits
        self.excluded_couplers = excluded_couplers

class ISMAGS(Placement):
    """
    finds a mapping between the circuit's wires and the machine's qubits using the ISMAGS subgraph isomorphism algorithm
    """
    def execute(self, tape):   
        """
        places the circuit on the machine's connectivity using ISMAGS subgraph isomorphism algorithm
        """
        circuit_topology = graph_util.circuit_graph(tape)
        machine_topology = graph_util.machine_graph(self.use_benchmark, self.q1_acceptance, self.q2_acceptance, self.excluded_qubits, self.excluded_couplers)

        if len(graph_util.find_biggest_group(circuit_topology)) > len(graph_util.find_biggest_group(machine_topology)):
            raise Exception(f"There are {machine_topology.number_of_nodes} qubits on the machine but your circuit has {circuit_topology.number_of_nodes}.")
        
        # 1. trouver un isomorphisme de sous-graph entre le circuit et la machine, maximisant le nombre de noeuds pris en compte
        mapping = graph_util.find_largest_subgraph_isomorphism_imags(circuit_topology, machine_topology)

        # 2. identifier les noeuds du circuit manquant dans le mapping (a)
        missing = [node for node in circuit_topology.nodes if node not in mapping.keys()]
            
        for node in missing:
            # 3. Trouver le noeud le plus connecté à A dans le circuit
            most_connected_node = graph_util.find_best_neighbour(node, circuit_topology, self.use_benchmark)

            # 4. trouver un noeud dans la machine (a') qui minimise le chemin entre a' et b'  
            possibles = [n for n in machine_topology.nodes if n not in mapping.values()]
            shortest_path_mapping = graph_util.node_with_shortest_path_from_selection(mapping[most_connected_node], possibles, machine_topology, self.use_benchmark)
                
            mapping[node] = shortest_path_mapping
        
        # 5. corriger les connexions dans le circuit en fonction du mappage choisi
        new_tape = type(tape)([op.map_wires(mapping) for op in tape.operations], [m.map_wires(mapping) for m in tape.measurements], shots=tape.shots)

        return new_tape

class VF2(Placement):
    """
    finds a mapping between the circuit's wires and the machine's qubits using the VF2 subgraph isomorphism algorithm
    """
    def execute(self, tape):
        """
        places the circuit on the machine's connectivity using VF2 algorithm
        """
        circuit_topology = graph_util.circuit_graph(tape)
        machine_topology = graph_util.machine_graph(self.use_benchmark, self.q1_acceptance, self.q2_acceptance, self.excluded_qubits, self.excluded_couplers)

        if len(graph_util.find_biggest_group(circuit_topology)) > len(graph_util.find_biggest_group(machine_topology)):
            raise Exception(f"There are {machine_topology.number_of_nodes} qubits on the machine but your circuit has {circuit_topology.number_of_nodes}.")
        
        # 1. trouver un isomorphisme de sous-graph entre le circuit et la machine, maximisant le nombre de noeuds pris en compte
        mapping = graph_util.find_largest_subgraph_isomorphism_vf2(circuit_topology, machine_topology)

        # 2. identifier les noeuds du circuit manquant dans le mapping (a)
        missing = [node for node in circuit_topology.nodes if node not in mapping.keys()]
            
        for node in missing:
            # 3. Trouver le noeud le plus connecté à A dans le circuit
            most_connected_node = graph_util.find_best_neighbour(node, circuit_topology, self.use_benchmark)

            # 4. trouver un noeud dans la machine (a') qui minimise le chemin entre a' et b'  
            possibles = [n for n in machine_topology.nodes if n not in mapping.values()]
            shortest_path_mapping = graph_util.node_with_shortest_path_from_selection(mapping[most_connected_node], possibles, machine_topology, self.use_benchmark)
                
            mapping[node] = shortest_path_mapping
        
        # 5. corriger les connexions dans le circuit en fonction du mappage choisi
        new_tape = type(tape)([op.map_wires(mapping) for op in tape.operations], [m.map_wires(mapping) for m in tape.measurements], shots=tape.shots)

        return new_tape

class ASTAR(Placement):
    """
    finds a mapping between the circuit's wires and the machine's qubits using an ASTAR based traversal heuristic
    """
    def _recurse(self, a_key, b_key, mapping, to_explore, machine_topology, circuit_topology):
        """traverses the circuit graph, finding mappings for nodes recursively

        Args:
            a_key (int): a source node
            b_key (int): a destination node
            mapping (dict[int, int]): the current mapping
            to_explore (list[int]): which nodes have not yet been explored
            machine_topology (nx.Graph): the machine's graph
            circuit_topology (nx.Graph): the circuit's graph
        """
        if b_key in mapping:
            return
        
        mapping[b_key] = graph_util.find_closest_wire(mapping[a_key], machine_topology, [v for (_, v) in mapping.items()], [v for _, v in mapping.items()], self.use_benchmark)

        a_key2 = b_key
        for b_key2 in to_explore:
            if (a_key2, b_key2) not in circuit_topology.edges:
                continue

            self._recurse(a_key2, b_key2, mapping, to_explore, machine_topology, circuit_topology)

    def execute(self, tape : QuantumTape) -> QuantumTape:
        """
        places the circuit on the machine's connectivity using astar algorithm and comparing path lengths
        """
        circuit_topology = graph_util.circuit_graph(tape)
        machine_topology = graph_util.machine_graph(self.use_benchmark, self.q1_acceptance, self.q2_acceptance, self.excluded_qubits, self.excluded_couplers)

        if len(graph_util.find_biggest_group(circuit_topology)) > len(graph_util.find_biggest_group(machine_topology)):
            raise Exception(f"There are {machine_topology.number_of_nodes} qubits on the machine but your circuit has {circuit_topology.number_of_nodes}.")
        
        mapping = {}
        # sort nodes by degree descending, so that we map the most connected node first
        to_explore = list(reversed(sorted([w for w in tape.wires], key=lambda w: circuit_topology.degree(w))))

        for a_key in to_explore:
            if a_key in mapping:
                continue
            mapping[a_key] = graph_util.find_best_wire(machine_topology, [v for _, v in mapping.items()], self.use_benchmark)

            for b_key in to_explore:
                if (a_key, b_key) not in circuit_topology.edges: 
                    continue
                
                self._recurse(a_key, b_key, mapping, to_explore, machine_topology, circuit_topology)

        new_tape = type(tape)([op.map_wires(mapping) for op in tape.operations], [m.map_wires(mapping) for m in tape.measurements], shots=tape.shots)
        return new_tape
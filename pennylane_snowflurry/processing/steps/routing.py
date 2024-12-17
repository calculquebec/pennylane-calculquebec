"""
Contains routing pre-processing steps
"""

from pennylane.tape import QuantumTape
from pennylane.operation import Operation
import pennylane as qml
from pennylane_snowflurry.processing.interfaces import PreProcStep
from pennylane_snowflurry.utility.graph import circuit_graph, shortest_path, machine_graph, is_directly_connected

class Routing(PreProcStep):
    """
    base class for routing algorithms
    """
    def __init__(self, use_benchmark = True, q1_acceptance = 0.5, q2_acceptance = 0.5, excluded_qubits=[], excluded_couplers=[]):
        """constructor for usual routing algorithms

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
        self.excluded_qubits=excluded_qubits
        self.excluded_couplers=excluded_couplers

class Swaps(Routing):
    """
    a routing algorithm that uses swaps
    """
    def execute(self, tape):
        """
        uses swap to permute wires when 2 qubits operation appear which are not directly mapped to a coupler in the machine
        
        ie. cnot(0, 1), qubit 0 and 1 are not directly connected in the machine's graph. 
        the shortest path from 0 to 1 is [0, 4, 1]
        the new circuit will be : swap(4, 1), cnot(0, 4), swap(4, 1)
        """
        # en fonction du mappage choisi, connecter les qubits non-couplés à la position des portes touchées en utilisant des swaps
        circuit_topology = circuit_graph(tape)
        machine_topology = machine_graph(self.use_benchmark, self.q1_acceptance, self.q2_acceptance, self.excluded_qubits, self.excluded_couplers)
        new_operations : list[Operation] = []
        list_copy = tape.operations.copy()

        for oper in list_copy:
            # s'il s'agit d'une porte à 2 qubit n'étant pas mappéee sur un coupleur physique, 
            # on la route avec des cnots en utilisant astar
            if oper.num_wires == 2 and not is_directly_connected(oper, machine_topology):
                path = shortest_path(oper.wires[0], oper.wires[1], machine_topology, prioritized_nodes=[n for n in circuit_topology.nodes], use_benchmark=self.use_benchmark)

                for i in reversed(range(1, len(path) - 1)): 
                    new_operations += [qml.SWAP([path[i], path[i+1]])]

                new_operations += [oper.map_wires({k:v for (k,v) in zip(oper.wires, [path[0], path[1]])})]

                for i in range(1, len(path) - 1):
                    new_operations += [qml.SWAP([path[i],path[i+1]])]
            else:
                new_operations += [oper]

        new_tape = type(tape)(new_operations, tape.measurements, shots=tape.shots)

        return new_tape

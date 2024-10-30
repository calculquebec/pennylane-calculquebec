import unittest.mock
from pennylane_snowflurry.transpiler.placement import placement_ismags, placement_astar
from pennylane_snowflurry.transpiler.routing import swap_routing
import unittest
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane_snowflurry.monarq_data import connectivity
import networkx as nx

class test_place_route(unittest.TestCase):
    NONE : int = 0
    ACCEPTANCE : int = 1 << 0
    
    @unittest.mock.patch("pennylane_snowflurry.utility.graph_utility.machine_graph")
    def test_place_no_4(self, machine_graph):
        links = [v for k,v in connectivity["couplers"].items() if 4 not in v]
        machine_graph.return_value = nx.Graph(links)
        answer = [5, 10, 9, 2, 1]
        ops = [qml.CNOT([0, 1]), 
               qml.CNOT([0, 2]), 
               qml.CNOT([0, 3]), 
               qml.CNOT([0, 4])]

        tape = QuantumTape(ops=ops)
        new_tape = placement_ismags(tape, self.ACCEPTANCE)
        self.assertListEqual(sorted(answer), sorted(int(w) for w in new_tape.wires))
    
    def test_place_trivial(self):
        answer = [4, 0, 1, 8, 9]
        ops = [qml.CNOT([0, 1]), 
               qml.CNOT([0, 2]), 
               qml.CNOT([0, 3]), 
               qml.CNOT([0, 4])]

        tape = QuantumTape(ops=ops)
        new_tape = placement_astar(tape, self.NONE)
        self.assertListEqual(sorted(answer), sorted(int(w) for w in new_tape.wires))

    @unittest.mock.patch("pennylane_snowflurry.utility.graph_utility.machine_graph")
    def test_place_too_many_wires(self, machine_graph):
        machine_graph.return_value = nx.Graph([[0, 4]])
        ops = [
            qml.CNOT([0, 1]),
            qml.CNOT([1, 2])
        ]
        tape = QuantumTape(ops = ops)
        self.assertRaises(Exception, lambda : placement_astar(tape, self.ACCEPTANCE))
    
    @unittest.mock.patch("pennylane_snowflurry.utility.graph_utility.machine_graph")
    def test_place_too_many_wires_holed_machine(self, machine_graph):
        machine_graph.return_value = nx.Graph([[0, 4], [1, 5]])
        ops = [
            qml.CNOT([0, 1]),
            qml.CNOT([1, 2])
        ]
        tape = QuantumTape(ops = ops)
        self.assertRaises(Exception, lambda : placement_astar(tape, self.ACCEPTANCE))
        
    def test_place_too_connected(self):
        answer = [4, 0, 9, 8, 1, 5]
        ops = [qml.CNOT([0, 1]), 
               qml.CNOT([0, 2]), 
               qml.CNOT([0, 3]), 
               qml.CNOT([0, 4]), 
               qml.CNOT([0, 5])]

        tape = QuantumTape(ops=ops)
        new_tape = placement_astar(tape, self.NONE)
        self.assertListEqual(sorted(answer), sorted(int(w) for w in new_tape.wires))

    def test_route_trivial(self):
        ops = [qml.CNOT([0, 4])]
        tape = QuantumTape(ops=ops)
        new_tape = swap_routing(tape, self.NONE)
        self.assertListEqual(ops, new_tape.operations)

    def test_route_distance1(self):
        results = [qml.SWAP([4, 1]), qml.CNOT([0, 4]), qml.SWAP([4, 1])]
        ops = [qml.CNOT([0, 1])]
        tape = QuantumTape(ops=ops)
        new_tape = swap_routing(tape, self.NONE)
        self.assertListEqual(results, new_tape.operations)

    def test_route_distance2(self):
        results = [qml.SWAP([1, 5]), qml.SWAP([4, 1]), qml.CNOT([0, 4]), qml.SWAP([4, 1]), qml.SWAP([1, 5])]
        ops = [qml.CNOT([0, 5])]
        tape = QuantumTape(ops=ops)
        new_tape = swap_routing(tape, self.NONE)
        self.assertListEqual(results, new_tape.operations)

    def test_route_short_loop(self):
        results = [qml.CNOT([4, 1]), qml.CNOT([1, 5]), qml.SWAP([1, 4]), qml.CNOT([5, 1]), qml.SWAP([1, 4])]
        ops = [qml.CNOT([4, 1]), 
               qml.CNOT([1, 5]), 
               qml.CNOT([5, 4])]
        tape = QuantumTape(ops=ops)
        new_tape = swap_routing(tape, self.NONE)
        self.assertListEqual(results, new_tape.operations)
 
    @unittest.mock.patch("pennylane_snowflurry.utility.graph_utility.machine_graph")       
    def test_route_impossible_connection(self, machine_graph):
        machine_graph.return_value = nx.Graph([[0, 4], [1, 5]])
        ops = [qml.CNOT([0, 1])]
        tape = QuantumTape(ops=ops)
        self.assertRaises(Exception, lambda : swap_routing(tape, self.NONE))

if __name__ == "__main__":
    unittest.main()
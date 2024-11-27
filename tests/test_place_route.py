from pennylane_snowflurry.processing.steps.placement import ISMAGS, ASTAR
from pennylane_snowflurry.processing.steps.routing import Swaps
from unittest.mock import patch
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane_snowflurry.monarq_data import connectivity
import networkx as nx
import pytest
from pennylane.wires import Wires

@pytest.fixture
def mock_machine_graph():
    with patch("pennylane_snowflurry.utility.graph.machine_graph") as machine_graph:
        yield machine_graph


class TestPlaceRoute:
    def test_place_no_4(self, mock_machine_graph):
        links = [v for k,v in connectivity["couplers"].items() if 4 not in v]
        mock_machine_graph.return_value = nx.Graph(links)
        answer = [5, 10, 9, 2, 1]
        ops = [qml.CNOT([0, 1]), 
               qml.CNOT([0, 2]), 
               qml.CNOT([0, 3]), 
               qml.CNOT([0, 4])]

        tape = QuantumTape(ops=ops)
        new_tape = ISMAGS().execute(tape)
        assert [w in new_tape.wires for w in answer] and len(answer) == len(new_tape.wires)
    
    
    def test_place_trivial(self):
        answer = [4, 0, 1, 8, 9]
        ops = [qml.CNOT([0, 1]), 
               qml.CNOT([0, 2]), 
               qml.CNOT([0, 3]), 
               qml.CNOT([0, 4])]

        tape = QuantumTape(ops=ops)
        new_tape = ISMAGS(False).execute(tape)
        print(new_tape.wires)
        print(answer)
        assert [w in new_tape.wires for w in answer] and len(answer) == len(new_tape.wires)


    def test_place_too_many_wires(self, mock_machine_graph):
        mock_machine_graph.return_value = nx.Graph([[0, 4]])
        ops = [
            qml.CNOT([0, 1]),
            qml.CNOT([1, 2])
        ]
        tape = QuantumTape(ops = ops)
        with pytest.raises(Exception):
            ISMAGS().execute(tape)
    
    
    def test_place_too_many_wires_holed_machine(self, mock_machine_graph):
        mock_machine_graph.return_value = nx.Graph([[0, 4], [1, 5]])
        ops = [
            qml.CNOT([0, 1]),
            qml.CNOT([1, 2])
        ]
        tape = QuantumTape(ops = ops)
        with pytest.raises(Exception):
            ISMAGS().execute(tape)
        
        
    def test_place_too_connected(self):
        answer = [4, 0, 9, 8, 1, 5]
        ops = [qml.CNOT([0, 1]), 
               qml.CNOT([0, 2]), 
               qml.CNOT([0, 3]), 
               qml.CNOT([0, 4]), 
               qml.CNOT([0, 5])]

        tape = QuantumTape(ops=ops)
        new_tape = ISMAGS(False).execute(tape)
        assert all(sorted(answer)[i] == a for i, a in enumerate(sorted(int(w) for w in new_tape.wires)))


    def test_route_trivial(self):
        ops = [qml.CNOT([0, 4])]
        tape = QuantumTape(ops=ops)
        new_tape = Swaps(False).execute(tape)
        assert all(ops[i] == a for i, a in enumerate(new_tape.operations))


    def test_route_distance1(self):
        results = [qml.SWAP([4, 1]), qml.CNOT([0, 4]), qml.SWAP([4, 1])]
        ops = [qml.CNOT([0, 1])]
        tape = QuantumTape(ops=ops)
        new_tape = Swaps(False).execute(tape)
        assert all(results[i] == a for i, a in enumerate(new_tape.operations))


    def test_route_distance2(self):
        results = [qml.SWAP([1, 5]), qml.SWAP([4, 1]), qml.CNOT([0, 4]), qml.SWAP([4, 1]), qml.SWAP([1, 5])]
        ops = [qml.CNOT([0, 5])]
        tape = QuantumTape(ops=ops)
        new_tape = Swaps(False).execute(tape)
        assert all(results[i] == a for i, a in enumerate(new_tape.operations))


    def test_route_short_loop(self):
        results = [qml.CNOT([4, 1]), qml.CNOT([1, 5]), qml.SWAP([1, 4]), qml.CNOT([5, 1]), qml.SWAP([1, 4])]
        ops = [qml.CNOT([4, 1]), 
               qml.CNOT([1, 5]), 
               qml.CNOT([5, 4])]
        tape = QuantumTape(ops=ops)
        new_tape = Swaps(False).execute(tape)
        assert all(results[i] == a for i, a in enumerate(new_tape.operations))
 
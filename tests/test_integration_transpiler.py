import numpy as np
import unittest
import pennylane as qml
from pennylane_snowflurry.transpiler.monarq_transpile import Transpiler
from pennylane_snowflurry.device_configuration import MonarqConfig
import pennylane_snowflurry.transpiler.transpiler_enums as enums
import pennylane_snowflurry.test_circuits as test_circuits
from pennylane_snowflurry.utility.debug_utility import arbitrary_circuit


def count_gates(tape):
    ops = {}
    for op in tape.operations:
        if op.name not in ops:
            ops[op.name] = 0
        
        ops[op.name] += 1
    return ops


class test_integration_transpiler(unittest.TestCase):
    def __init__(self, _):
        super().__init__(_)
        self.dev = qml.device("default.qubit", shots=1000)
        
        
    def test_ghz6(self):
        results={'X90': 21, 'CZ': 11, 'PauliZ': 15, 'Z90': 2, 'T': 1, 'PhaseShift': 1}
        
        qnode = qml.QNode(lambda : test_circuits.GHZ(6), self.dev)
        qnode()
        transpiler = Transpiler.get_transpiler(MonarqConfig(useBenchmark=enums.Benchmark.NONE))
        tape = transpiler(qnode.tape)[0][0]
        counts = count_gates(tape)
        self.assertDictEqual(results, counts)
    
    def test_bernstein_vazirani(self):
        results={'CZ': 4, 'PauliX': 1, 'PauliZ': 5, 'X90': 9}
        
        qnode = qml.QNode(lambda : test_circuits.bernstein_vazirani(54), self.dev)
        qnode()
        transpiler = Transpiler.get_transpiler(MonarqConfig(useBenchmark=enums.Benchmark.NONE))
        tape = transpiler(qnode.tape)[0][0]
        counts = count_gates(tape)
        self.assertDictEqual(results, counts)

    def test_AQFT(self):
        results={'CZ': 279, 'PauliZ': 436, 'PhaseShift': 39, 'T': 16, 'X90': 478, 'Z90': 57}
        
        def AQFT():
            qml.adjoint(qml.QFT(range(6)))
            return qml.counts(wires=range(6))
        
        qnode = qml.QNode(AQFT, self.dev)
        qnode()
        transpiler = Transpiler.get_transpiler(MonarqConfig(useBenchmark=enums.Benchmark.NONE))
        tape = transpiler(qnode.tape)[0][0]
        counts = count_gates(tape)
        self.assertDictEqual(results, counts)
        
if __name__ == "__main__":
    unittest.main()
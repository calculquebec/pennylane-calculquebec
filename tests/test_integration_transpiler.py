import numpy as np
import pennylane as qml
from pennylane_snowflurry.transpiler.monarq_transpile import Transpiler
from pennylane_snowflurry.transpiler.transpiler_config import MonarqDefaultConfig
import pennylane_snowflurry.utility.test_circuits as test_circuits
from pennylane_snowflurry.utility.debug_utility import arbitrary_circuit
from pennylane_snowflurry.transpiler.steps.placement import ISMAGS
import pytest

def count_gates(tape):
    ops = {}
    for op in tape.operations:
        if op.name not in ops:
            ops[op.name] = 0
        
        ops[op.name] += 1
    return ops


class TestIntegrationTranspiler:
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.dev = qml.device("default.qubit", shots=1000)
        self.conf = MonarqDefaultConfig(use_benchmark=False)
        self.conf.steps[1] = ISMAGS(use_benchmark=False)
        
    def test_ghz6(self):
        results={'X90': 21, 'CZ': 11, 'PauliZ': 15}
        
        qnode = qml.QNode(lambda : test_circuits.GHZ(6), self.dev)
        qnode()
        
        transpiler = Transpiler.get_transpiler(self.conf)
        tape = transpiler(qnode.tape)[0][0]
        counts = count_gates(tape)
        
        assert len(results) == len(counts)
        assert all([results[key] == counts[key] for key in results])
    
    def test_bernstein_vazirani(self):
        results={'CZ': 4, 'PauliX': 1, 'PauliZ': 4, 'X90': 9}
        
        qnode = qml.QNode(lambda : test_circuits.bernstein_vazirani(54, 7), self.dev)
        result = qnode()
        
        transpiler = Transpiler.get_transpiler(MonarqDefaultConfig(use_benchmark=False))
        tape = transpiler(qnode.tape)[0][0]
        counts = count_gates(tape)
        
        assert len(results) == len(counts)
        assert all([results[key] == counts[key] for key in results])

    def test_AQFT(self):
        results={'CZ': 279, 'PauliZ': 436, 'PhaseShift': 36, 'X90': 478}
        
        def AQFT():
            qml.adjoint(qml.QFT(range(6)))
            return qml.counts(wires=range(6))
        
        qnode = qml.QNode(AQFT, self.dev)
        qnode()
        transpiler = Transpiler.get_transpiler(MonarqDefaultConfig(use_benchmark=False))
        tape = transpiler(qnode.tape)[0][0]
        counts = count_gates(tape)
        
        assert len(results) == len(counts)
        assert all([results[key] == counts[key] for key in results])

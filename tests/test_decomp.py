import numpy as np
from pennylane_snowflurry.transpiler.steps.base_decomposition import CliffordTDecomposition
from pennylane_snowflurry.transpiler.steps.native_decomposition import MonarqDecomposition
from pennylane_snowflurry.utility.api_utility import instructions
import pennylane as qml
from pennylane.tape import QuantumTape
import pytest


class TestDecomp:
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.cliffordTDecomposition = CliffordTDecomposition()
        self.monarqDecomposition = MonarqDecomposition()
        self.dev = qml.device("default.qubit")

    def test_base_decomp_toffoli(self):
        ops = [qml.Hadamard(0), qml.Hadamard(1), qml.Toffoli([0, 1, 2])]
        tape = QuantumTape(ops=ops, measurements=[qml.probs()])
        new_tape = self.cliffordTDecomposition.execute(tape)
        assert all(op.name in self.cliffordTDecomposition.base_gates for op in new_tape.operations)

        a = qml.execute([tape], self.dev)
        b = qml.execute([new_tape], self.dev)
        assert all(abs(c[0] - c[1]) < 1E-8 for c in zip(a[0], b[0]))

    def test_base_decomp_unitary(self):
        ops = [qml.Hadamard(0), qml.QubitUnitary(np.array([[-1, 1], [1, 1]])/np.sqrt(2), 0)]
        tape = QuantumTape(ops=ops, measurements=[qml.probs()])
        new_tape = self.cliffordTDecomposition.execute(tape)
        assert all(op.name in self.cliffordTDecomposition.base_gates for op in new_tape.operations)

        a = qml.execute([tape], self.dev)
        b = qml.execute([new_tape], self.dev)
        assert all(abs(c[0] - c[1]) < 1E-8 for c in zip(a[0], b[0]))

    def test_base_decomp_cu(self):
        ops = [qml.Hadamard(0), qml.Hadamard(1), qml.Hadamard(2), qml.ControlledQubitUnitary(np.array([[0, 1], [1, 0]]), [0, 1], [2], [0, 1])]
        tape = QuantumTape(ops=ops, measurements=[qml.probs()])
        new_tape = self.cliffordTDecomposition.execute(tape)
        assert all(op.name in self.cliffordTDecomposition.base_gates for op in new_tape.operations)

        a = qml.execute([tape], self.dev)
        b = qml.execute([new_tape], self.dev)
        assert all(abs(c[0] - c[1]) < 1E-8 for c in zip(a[0], b[0]))

    def test_native_decomp_toffoli(self):
        ops = [qml.Hadamard(0), qml.Hadamard(1), qml.Hadamard(2), qml.Toffoli([0, 1, 2])]
        tape = QuantumTape(ops=ops, measurements=[qml.probs()])
        new_tape = self.cliffordTDecomposition.execute(tape)
        new_tape = self.monarqDecomposition.execute(new_tape)
                
        assert all(op.name in instructions for op in new_tape.operations)
        a = qml.execute([tape], self.dev)
        b = qml.execute([new_tape], self.dev)
        assert all(abs(c[0] - c[1]) < 1E-8 for c in zip(a[0], b[0]))

    def test_native_decomp_unitary(self):
        ops = [qml.Hadamard(0), qml.QubitUnitary(np.array([[-1, 1], [1, 1]])/np.sqrt(2), 0)]
        tape = QuantumTape(ops=ops, measurements=[qml.probs()])
        new_tape = self.cliffordTDecomposition.execute(tape)
        new_tape = self.monarqDecomposition.execute(new_tape)

        assert all(op.name in instructions for op in new_tape.operations)
        a = qml.execute([tape], self.dev)
        b = qml.execute([new_tape], self.dev)
        assert all(abs(c[0] - c[1]) < 1E-8 for c in zip(a[0], b[0]))

    def test_native_decomp_cu(self):
        ops = [qml.Hadamard(0), qml.Hadamard(1), qml.Hadamard(2), qml.ControlledQubitUnitary(np.array([[0, 1], [1, 0]]), [0, 1], [2], [0, 1])]
        tape = QuantumTape(ops=ops, measurements=[qml.probs()])
        new_tape = self.cliffordTDecomposition.execute(tape)
        new_tape = self.monarqDecomposition.execute(new_tape)

        assert all(op.name in instructions for op in new_tape.operations)
        a = qml.execute([tape], self.dev)
        b = qml.execute([new_tape], self.dev)
        assert all(abs(c[0] - c[1]) < 1E-8 for c in zip(a[0], b[0]))
        
    def test_gate_not_in_decomp_map(self):
        ops = [qml.Toffoli([0, 1, 2])]
        tape = QuantumTape(ops=ops)
    
        with pytest.raises(Exception):
            self.monarqDecomposition.execute(tape)

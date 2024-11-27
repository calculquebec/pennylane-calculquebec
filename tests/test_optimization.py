import numpy as np
from pennylane_snowflurry.processing.steps.base_decomposition import CliffordTDecomposition
from pennylane_snowflurry.processing.steps.optimization import IterativeCommuteAndMerge
import pennylane as qml
from pennylane.tape import QuantumTape
import pytest

class TestOptimization:
        
    @pytest.fixture(autouse=True)
    def setup(self):
        self.dev = qml.device("default.qubit")

    def test_optimize_qubit_unitary(self):
        ops = [qml.Hadamard(0), qml.QubitUnitary(np.array([[-1, 1], [1, 1]])/np.sqrt(2), 0)]
        tape = QuantumTape(ops=ops, measurements=[qml.probs()])
        new_tape = CliffordTDecomposition().execute(tape)
        new_tape = IterativeCommuteAndMerge().execute(new_tape)

        assert len(new_tape.operations) == 1
        a = qml.execute([tape], self.dev)
        b = qml.execute([new_tape], self.dev)
        assert all(abs(c[0] - c[1]) < 1E-8 for c in zip(a[0], b[0]))

    def test_optimize_toffoli(self):
        ops = [qml.Hadamard(0), qml.Hadamard(1), qml.Hadamard(2), qml.Toffoli([0, 1, 2])]
        tape = QuantumTape(ops=ops, measurements=[qml.probs()])
        new_tape = CliffordTDecomposition().execute(tape)
        new_tape = IterativeCommuteAndMerge().execute(new_tape)
        
        assert len(new_tape.operations) == 33
        a = qml.execute([tape], self.dev)
        b = qml.execute([new_tape], self.dev)
        assert all(abs(c[0] - c[1]) < 1E-8 for c in zip(a[0], b[0]))

    def test_optimize_cu(self):
        ops = [qml.Hadamard(0), qml.Hadamard(1), qml.Hadamard(2), qml.ControlledQubitUnitary(np.array([[0, 1], [1, 0]]), [0, 1], [2], [0, 1])]
        tape = QuantumTape(ops=ops, measurements=[qml.probs()])
        new_tape = CliffordTDecomposition().execute(tape)
        new_tape = IterativeCommuteAndMerge().execute(new_tape)
        
        assert len(new_tape.operations) == 61
        a = qml.execute([tape], self.dev)
        b = qml.execute([new_tape], self.dev)
        assert all(abs(c[0] - c[1]) < 1E-8 for c in zip(a[0], b[0]))

import numpy as np
from pennylane_calculquebec.processing.steps.base_decomposition import (
    CliffordTDecomposition,
    BaseDecomposition,
)
from pennylane_calculquebec.processing.steps.native_decomposition import (
    MonarqDecomposition,
)
from pennylane_calculquebec.utility.api import instructions
from pennylane_calculquebec.utility.debug import are_tape_same_probs
import pennylane as qml
from pennylane.tape import QuantumTape
import pytest
from functools import reduce


def test_base_decomp_class_empty_gates():
    step = BaseDecomposition()
    assert len(step.base_gates) == 0


def test_base_decomp_toffoli():
    step = CliffordTDecomposition()
    ops = [qml.Hadamard(0), qml.Hadamard(1), qml.Toffoli([0, 1, 2])]
    tape = QuantumTape(ops=ops, measurements=[qml.probs()])
    new_tape = step.execute(tape)
    assert all(op.name in step.base_gates for op in new_tape.operations)

    assert are_tape_same_probs(tape, new_tape)


def test_base_decomp_unitary():
    step = CliffordTDecomposition()

    ops = [
        qml.Hadamard(0),
        qml.QubitUnitary(np.array([[-1, 1], [1, 1]]) / np.sqrt(2), 0),
    ]
    tape = QuantumTape(ops=ops, measurements=[qml.probs()])
    new_tape = step.execute(tape)
    assert all(op.name in step.base_gates for op in new_tape.operations)
    assert are_tape_same_probs(tape, new_tape)


def test_base_decomp_cu():
    step = CliffordTDecomposition()

    ops = [
        qml.Hadamard(0),
        qml.Hadamard(1),
        qml.ControlledQubitUnitary(
            np.array([[1, 1], [1, -1]]) / np.sqrt(2), [0, 1], [2], [0, 1]
        ),
    ]
    tape = QuantumTape(ops=ops, measurements=[qml.probs()])
    new_tape = step.execute(tape)

    assert all(op.name in step.base_gates for op in new_tape.operations)
    assert are_tape_same_probs(tape, new_tape)

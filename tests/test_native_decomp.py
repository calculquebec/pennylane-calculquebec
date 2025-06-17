import numpy as np
from pennylane_calculquebec.processing.steps.base_decomposition import (
    CliffordTDecomposition,
)
from pennylane_calculquebec.processing.steps.native_decomposition import (
    MonarqDecomposition,
)
from pennylane_calculquebec.utility.api import instructions
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane_calculquebec.utility.debug import are_tape_same_probs
import pytest

from functools import reduce


def test_native_decomp_toffoli():
    preproc = CliffordTDecomposition()
    step = MonarqDecomposition()

    ops = [qml.Hadamard(0), qml.Hadamard(1), qml.Toffoli([0, 1, 2])]
    tape = QuantumTape(ops=ops, measurements=[qml.probs()])
    new_tape = preproc.execute(tape)
    new_tape = step.execute(new_tape)

    assert all(op.name in instructions for op in new_tape.operations)

    assert are_tape_same_probs(tape, new_tape)


def test_native_decomp_unitary():
    preproc = CliffordTDecomposition()
    step = MonarqDecomposition()

    ops = [
        qml.Hadamard(0),
        qml.QubitUnitary(np.array([[-1, 1], [1, 1]]) / np.sqrt(2), 0),
    ]
    tape = QuantumTape(ops=ops, measurements=[qml.probs()])
    new_tape = preproc.execute(tape)
    new_tape = step.execute(new_tape)

    assert all(op.name in instructions for op in new_tape.operations)

    assert are_tape_same_probs(tape, new_tape)


def test_native_decomp_cu():
    preproc = CliffordTDecomposition()
    step = MonarqDecomposition()

    ops = [
        qml.Hadamard(0),
        qml.Hadamard(1),
        qml.Hadamard(2),
        qml.ControlledQubitUnitary(np.array([[0, 1], [1, 0]]), [0, 1], [2], [0, 1]),
    ]
    tape = QuantumTape(ops=ops, measurements=[qml.probs()])
    new_tape = preproc.execute(tape)
    new_tape = step.execute(new_tape)

    assert all(op.name in instructions for op in new_tape.operations)

    assert are_tape_same_probs(tape, new_tape)


def test_gate_not_in_decomp_map():
    ops = [qml.Toffoli([0, 1, 2])]
    tape = QuantumTape(ops=ops)
    step = MonarqDecomposition()

    with pytest.raises(Exception):
        step.execute(tape)

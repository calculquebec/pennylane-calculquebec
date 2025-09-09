import pennylane as qml
from pennylane.tape import QuantumTape
import numpy as np
from pennylane_calculquebec.exceptions import ProcessingError
from pennylane_calculquebec.processing.steps import DecomposeReadout
import pytest
from unittest.mock import patch
from pennylane.ops import Prod


def test_execute():
    obs = [qml.Z(0), qml.X(0), qml.Y(0), qml.Hadamard(0), qml.X(0) @ qml.Y(1), None]
    step = DecomposeReadout()

    # X, Y, Z and H
    for observable in obs:
        tape = QuantumTape([], [qml.counts(observable)])
        diag = observable.diagonalizing_gates() if observable is not None else []
        tape = step.execute(tape)
        assert len(tape.operations) == len(diag) and len(tape.measurements) == 1
        for i, op in enumerate(tape.operations):
            assert op == diag[i]

    obs = [qml.S(0), qml.PauliX(0) @ qml.PauliZ(0)]

    # unsupported observable
    for observable in obs:
        tape = QuantumTape([], [qml.counts(observable)])
        with pytest.raises(ProcessingError):
            tape = step.execute(tape)

import numpy as np
import pennylane_calculquebec.utility.debug as debug
import pennylane as qml
import pytest
from unittest.mock import patch
import pennylane_calculquebec.processing.custom_gates as custom


@pytest.mark.parametrize(
    "input, expected",
    [
        ([0.5, 0, 0, 0.5], 1),
        ([0.25, 0.25, 0.25, 0.25], 0),
        ({"00": 500, "01": 0, "10": 0, "11": 500}, 1),
        ({"00": 250, "01": 250, "10": 250, "11": 250}, 0),
    ],
)
def test_compute_expval(input, expected):
    # test with probabilities

    result = debug.compute_expval(input)
    assert expected == result


@pytest.mark.parametrize(
    "input, expected",
    [
        ({"00": 500, "01": 0, "10": 0, "11": 500}, [0.5, 0, 0, 0.5]),
        ({"01": 500, "10": 500}, [0, 0.5, 0.5, 0]),
        ({"11": 500, "01": 500}, [0, 0.5, 0, 0.5]),
    ],
)
def test_counts_to_probs(input, expected):
    tolerance = 1e-5

    result = debug.counts_to_probs(input)
    assert all(abs(a - b) < tolerance for a, b in zip(result, expected))


def test_are_tape_same_probs():
    with patch("pennylane.execute") as execute:

        # we dont pass tape in this test. we pass the answer
        # since we dont care about testing the qml.execute function,
        # we just want to test if its outputs are equal
        execute.side_effect = lambda tape, dev: tape

        # probs
        bell = [0.5, 0, 0, 0.5]
        plus_zero = [0.5, 0, 0.5, 0]
        plus = [0.5, 0.5]

        assert debug.are_tape_same_probs(bell, bell)
        assert not debug.are_tape_same_probs(bell, plus_zero)
        assert not debug.are_tape_same_probs(bell, plus)

        # counts
        bell = {"00": 500, "11": 500}
        plus_zero = {"00": 500, "10": 500}
        plus = {"0": 500, "1": 500}

        assert debug.are_tape_same_probs(bell, bell)
        assert not debug.are_tape_same_probs(bell, plus_zero)
        assert not debug.are_tape_same_probs(bell, plus)


@pytest.mark.parametrize(
    "ops, qasm",
    [
        ([qml.PauliX(0)], ["x q[0];"]),
        ([qml.Hadamard(0), qml.CNOT([0, 1])], ["h q[0];", "cx q[0], q[1];"]),
        ([qml.RX(3, 0), qml.adjoint(qml.S)(0)], ["rx(3) q[0];", "sdg q[0];"]),
        ([custom.Y90(0), custom.YM90(1)], ["ry(pi/2) q[0];", "ry(3*pi/2) q[1];"]),
    ],
)
def test_to_qasm(ops, qasm):
    class Tape:
        def __init__(self, ops):
            self.operations = ops

    tape = Tape(ops)
    result = debug.to_qasm(tape)
    # turn the qasm string into an array of strings
    result = [a.strip() for a in result.split("\n") if a != ""]
    assert all(a == b for a, b in zip(result, qasm))


def test_get_labels():
    less_than_0 = -1
    with pytest.raises(ValueError):
        debug.get_labels(less_than_0)

    not_an_int = [[0, 1, 2], "3", 3.14, qml.probs(0), lambda: 3]
    for test in not_an_int:
        with pytest.raises(ValueError):
            debug.get_labels(test)

    # arbitrary value
    test_value = 6
    expected = ["000", "001", "010", "011", "100", "101", "110"]
    result = debug.get_labels(test_value)
    assert all(a == b for a, b in zip(expected, result))


def test_get_measurement_wires():
    class Tape:
        def __init__(self, wires, measures):
            self.wires = wires
            self.measurements = measures

    class Measure:
        def __init__(self, wires):
            self.wires = wires

    tape = Tape([0, 1, 2, 3], [Measure([0]), Measure([1]), Measure([3])])
    expected = [0, 1, 3]
    result = debug.get_measurement_wires(tape)

    assert len(expected) == len(result) and all(a in result for a in expected)

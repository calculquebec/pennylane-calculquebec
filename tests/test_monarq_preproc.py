import pytest
from unittest.mock import patch
from pennylane_calculquebec.processing import PreProcessor
from pennylane_calculquebec.processing.interfaces import PreProcStep, PostProcStep
from autograd.numpy.numpy_boxes import ArrayBox
import numpy as np


@pytest.fixture
def mock_expand_full_measurements():
    with patch(
        "pennylane_calculquebec.processing.PreProcessor.expand_full_measurements"
    ) as mock:
        yield mock


@pytest.fixture
def mock_unroll_array_boxes():
    with patch(
        "pennylane_calculquebec.processing.PreProcessor.unroll_array_boxes"
    ) as mock:
        yield mock


class step_call_counter:
    def __init__(self):
        self.i = 0


class op:
    def __init__(self, wires, data=[]):
        self.wires = wires
        self.data = data

    @property
    def num_params(self):
        return len(self.data)


class Tape:
    def __init__(self, ops=[], mps=[], wires=[], shots=None):
        self.operations = ops
        self.measurements = mps
        self.wires = wires
        self.shots = shots
        self.results = []


class config:
    def __init__(self, *params):
        self.steps = [p for p in params]


class step(PreProcStep):
    def __init__(self, test, call_counter):
        self.test = test
        self.call_counter = call_counter

    def execute(self, tape):
        self.call_counter.i += 1
        tape.results += [self.test]
        return tape


def test_get_processor(mock_expand_full_measurements, mock_unroll_array_boxes):
    mock_expand_full_measurements.side_effect = lambda a, b: a
    mock_unroll_array_boxes.side_effect = lambda a, b: a

    call_counter = step_call_counter()
    conf = config(
        step("a", call_counter),
        step("b", call_counter),
        PostProcStep(),
        step("c", call_counter),
        "not_step",
    )
    tape = Tape()
    process = PreProcessor.get_processor(conf, [0, 1, 2])
    tape2 = process(tape)[0][0]

    solution = ["a", "b", "c"]

    mock_expand_full_measurements.assert_called_once_with(tape, [0, 1, 2])
    assert call_counter.i == 3
    for i, r in enumerate(tape2.results):
        assert solution[i] == r


def test_expand_full_measurements():
    tape = Tape(mps=[op([])])
    result: Tape = PreProcessor.expand_full_measurements(tape, [4, 1, 2])
    solution = [4, 1, 2]
    for i, w in enumerate(solution):
        assert w == result.measurements[0].wires[i]

    tape = Tape(mps=[op([1, 2])])
    result: Tape = PreProcessor.expand_full_measurements(tape, [4, 1, 2])
    solution = [1, 2]
    for i, w in enumerate(solution):
        assert w == result.measurements[0].wires[i]

    tape = Tape(mps=[op([1]), op([2])])
    result: Tape = PreProcessor.expand_full_measurements(tape, [4, 1, 2])
    solution = [1, 2]
    for i, w in enumerate(solution):
        assert w == result.measurements[i].wires[0]


def test_unroll_array_boxes():
    op1 = op([0])
    op2 = op([1], [0.5])
    op3 = op([2], [ArrayBox(np.array([1.5]), None, None)])

    tape = Tape(ops=[op1, op2, op3])
    tape = PreProcessor.unroll_array_boxes(tape, [0, 1])

    assert tape.operations[0].num_params == 0
    assert tape.operations[1].data[0] == 0.5
    assert tape.operations[2].data[0] == 1.5

import pytest
from unittest.mock import patch
from pennylane_calculquebec.processing.steps.print_steps import (
    PrintResults,
    PrintTape,
    PrintWires,
)


class Tape:
    def __init__(self, ops):
        self.operations = ops
        self.wires = [0, 1, 2]


class Op:
    def __init__(self):
        pass


@pytest.fixture
def mock_print():
    with patch("builtins.print") as mock:
        yield mock


def test_print_tape(mock_print):
    step = PrintTape()
    tape = Tape(ops=[Op(), Op(), Op()])

    new_tape = step.execute(tape)

    assert tape is new_tape
    mock_print.assert_called_once()


def test_print_results(mock_print):
    step = PrintResults()
    tape = Tape(ops=[Op(), Op(), Op()])
    results = {"0": 0, "1": 1, "2": 2}

    new_results = step.execute(tape, results)
    assert results is new_results
    mock_print.assert_called_once()


def test_print_wires(mock_print):
    step = PrintWires()
    tape = Tape(ops=[Op(), Op(), Op()])
    results = {"0": 0, "1": 1, "2": 2}

    new_tape = step.execute(tape)
    assert tape is new_tape
    mock_print.assert_called_once()

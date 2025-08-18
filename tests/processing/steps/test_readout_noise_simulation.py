import pennylane_calculquebec.processing.steps.readout_noise_simulation as rns
import pytest
from unittest.mock import patch
import numpy as np


class MP:
    def __init__(self, wire):
        self.wires = [wire]


class Shots:
    def __init__(self, shots):
        self.total_shots = shots


class Tape:

    def __init__(self, wires: list[int], shots: int, measurements: list[int] = None):
        measurements = measurements if measurements is not None else wires
        self.wires = wires
        self.shots = Shots(shots)
        self.measurements = [MP(wire) for wire in measurements]


@pytest.fixture
def mock_readout_noise_matrices():
    with patch(
        "pennylane_calculquebec.processing.steps.readout_noise_simulation.get_readout_noise_matrices"
    ) as mock:
        yield mock


def test_execute(mock_readout_noise_matrices):
    """
    R = np.array([
                [f0, 1 - f1],
                [1 - f0, f1]
            ])
    """
    tol = 1e-5
    mock_readout_noise_matrices.return_value = np.array(
        [
            [[0, 1], [1, 0]],  # q0 has 0% fidelity
            [[0.5, 0.5], [0.5, 0.5]],  # q1 has 50% fidelity
            [[1, 0], [0, 1]],  # q2 has 100% fidelity
            [[0.5, 0.5], [0.5, 0.5]],  # q3 has 50% fidelity
        ]
    )

    tape = Tape([0, 1, 2, 3], 1000, [0, 1, 3])

    results = {"000": 1000}

    expected = {
        "000": 0,
        "001": 0,
        "010": 0,
        "011": 0,
        "100": 250,
        "101": 250,
        "110": 250,
        "111": 250,
    }

    step = rns.ReadoutNoiseSimulation("yamaska", False)
    _ = step.execute(tape, results)
    mock_readout_noise_matrices.assert_not_called()

    step = rns.ReadoutNoiseSimulation("yamaska", True)
    result2 = step.execute(tape, results)
    mock_readout_noise_matrices.assert_called_once()

    assert all(abs(expected[a] - result2[b]) < tol for a, b in zip(expected, result2))

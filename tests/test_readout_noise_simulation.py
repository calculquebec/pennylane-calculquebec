import pennylane_calculquebec.processing.steps.readout_noise_simulation as rns
import pytest
from unittest.mock import patch
import numpy as np
class Shots:
    def __init__(self, shots):
        self.total_shots = shots
    
class Tape:
    
    def __init__(self, wires, shots):
        self.wires = wires
        self.shots = Shots(shots)

@pytest.fixture
def mock_readout_noise_matrices():
    with patch("pennylane_calculquebec.processing.steps.readout_noise_simulation.get_readout_noise_matrices") as mock:
        yield mock

def test_execute(mock_readout_noise_matrices):
    """
    R = np.array([
                [f0, 1 - f1],
                [1 - f0, f1]
            ])
    """

    mock_readout_noise_matrices.return_value = np.array([
        [[0, 1], [1, 0]], # q0 has 0% fidelity
        [[0.5, 0.5], [0.5, 0.5]], # q1 has 50% fidelity
        [[1, 0], [0, 1]], # q2 has 100% fidelity
    ])

    tape = Tape([0, 1, 2], 1000)

    results = {
        "000" : 1000
    }
    
    expected = {
        "000" : 0, "001" : 0, "010" : 0, "011" : 0, "100" : 500, "101" : 0, "110" : 500, "111" : 0
    }
    step = rns.ReadoutNoiseSimulation(True)
    
    results = step.execute(tape, results)

    assert all(v1 == v2 for v1, v2 in zip(expected.values(), results.values()))
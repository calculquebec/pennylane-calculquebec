import numpy as np
import pytest
from unittest.mock import patch
import pennylane_calculquebec.processing.steps.readout_error_mitigation as mitigation
from pennylane_calculquebec.utility.api import keys
from typing import Tuple
from pennylane.tape import QuantumTape
import pennylane as qml
from pennylane_calculquebec.utility.noise import readout_error, TypicalBenchmark

COUNT_ACCEPTANCE = 20
TOLERANCE = 1e-5


class Tape:
    def __init__(self, total_wires, measurements):
        self.wires = total_wires
        self.measurements = measurements
        self.shots = Shots()


class Measure:
    def __init__(self, measure_wires):
        self.wires = measure_wires


class Shots:
    def __init__(self):
        self.total_shots = 1000


def qubits_couplers(*readouts: list[Tuple[int, int]]):
    results = {}

    for i, readout in enumerate(readouts):
        results[str(i)] = {
            keys.READOUT_STATE_0_FIDELITY: readout[0],
            keys.READOUT_STATE_1_FIDELITY: readout[1],
        }

    return {keys.QUBITS: results}


@pytest.fixture
def mock_qubits_couplers():
    with patch(
        "pennylane_calculquebec.API.adapter.ApiAdapter.get_qubits_and_couplers"
    ) as mock:
        yield mock


def test_all_combinations():
    results = ["000", "001", "010", "011", "100", "101", "110", "111"]
    qubits = 3
    combs = mitigation.all_combinations(qubits)
    assert all(a == b for a, b in zip(sorted(results), sorted(combs)))

    with patch(
        "pennylane_calculquebec.processing.steps.readout_error_mitigation.get_labels"
    ) as mock:
        mitigation.all_combinations(qubits)
        mock.assert_called_once()


def test_all_results():
    input = {"00": 5, "10": 7}
    output = {"00": 5, "01": 0, "10": 7, "11": 0}

    result = mitigation.all_results(input, 2)

    assert len(result.items()) == len(output.items())

    assert all(
        a == b
        for a, b in zip(
            sorted(output.items(), key=lambda a: a[0]),
            sorted(result.items(), key=lambda a: a[0]),
        )
    )


def test_get_readout_fidelities(mock_qubits_couplers):
    mock_qubits_couplers.return_value = qubits_couplers(
        (0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4)
    )

    readout0Expected = [0.9, 0.7, 0.6]
    readout1Expected = [0.1, 0.3, 0.4]

    readout0, readout1 = mitigation.get_readout_fidelities("yamaska", [0, 2, 3])

    mock_qubits_couplers.assert_called_once()

    assert all(a == b for a, b in zip(readout0, readout0Expected))
    assert all(a == b for a, b in zip(readout1, readout1Expected))


def test_get_calibration_data(mock_qubits_couplers):
    mock_qubits_couplers.return_value = qubits_couplers(
        (0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4)
    )

    expected = [
        np.array([[0.9, 0.9], [0.1, 0.1]]),
        np.array([[0.7, 0.7], [0.3, 0.3]]),
        np.array([[0.6, 0.6], [0.4, 0.4]]),
    ]
    results = mitigation.get_calibration_data("yamaska", [0, 2, 3])

    assert len(expected) == len(results)
    zipped = list(zip(expected, results))
    for a, b in zipped:
        zipped_matrices = list(zip(a.flatten(), b.flatten()))
        assert all(abs(u - v) < 1e-5 for u, v in zipped_matrices)


def test_tensor_product_calibration(mock_qubits_couplers):
    mock_qubits_couplers.return_value = qubits_couplers(
        (0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4)
    )

    expected = np.array(
        [
            [0.378, 0.378, 0.378, 0.378, 0.378, 0.378, 0.378, 0.378],
            [0.252, 0.252, 0.252, 0.252, 0.252, 0.252, 0.252, 0.252],
            [0.162, 0.162, 0.162, 0.162, 0.162, 0.162, 0.162, 0.162],
            [0.108, 0.108, 0.108, 0.108, 0.108, 0.108, 0.108, 0.108],
            [0.042, 0.042, 0.042, 0.042, 0.042, 0.042, 0.042, 0.042],
            [0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028],
            [0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018],
            [0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012],
        ]
    )

    calibration_matrices = [
        np.array([[0.9, 0.9], [0.1, 0.1]]),
        np.array([[0.7, 0.7], [0.3, 0.3]]),
        np.array([[0.6, 0.6], [0.4, 0.4]]),
    ]
    results = mitigation.tensor_product_calibration(calibration_matrices)
    assert expected.shape == results.shape
    assert all(
        abs(a - b) < TOLERANCE for a, b in zip(expected.flatten(), results.flatten())
    )


def test_matrix_readout_mitigation_full(mock_qubits_couplers):
    TOLERANCE = 5
    from pennylane_calculquebec.processing.steps import ReadoutNoiseSimulation

    typical = (TypicalBenchmark.readout0, TypicalBenchmark.readout1)
    mock_qubits_couplers.return_value = qubits_couplers(
        typical, typical, typical, typical
    )

    expected = {"000": 500, "111": 500}
    tape = Tape([0, 1, 2, 3], [Measure([0]), Measure([2]), Measure([3])])

    sim = ReadoutNoiseSimulation("yamaska", False)
    simulated_noise = sim.execute(tape, expected)

    step = mitigation.MatrixReadoutMitigation("yamaska")

    results = step.execute(tape, simulated_noise)

    for key in mitigation.all_combinations(3):
        if key not in expected:
            assert abs(results[key]) < TOLERANCE
            continue
        assert abs(expected[key] - results[key]) < TOLERANCE


def test_ibu_readout_mitigation_full(mock_qubits_couplers):
    from pennylane_calculquebec.processing.steps import ReadoutNoiseSimulation

    typical = (TypicalBenchmark.readout0, TypicalBenchmark.readout1)
    mock_qubits_couplers.return_value = qubits_couplers(
        typical, typical, typical, typical
    )

    expected = {"000": 500, "111": 500}
    tape = Tape([0, 1, 2, 3], [Measure([0]), Measure([2]), Measure([3])])

    sim = ReadoutNoiseSimulation("yamaska", False)
    simulated_noise = sim.execute(tape, expected)

    step = mitigation.IBUReadoutMitigation("yamaska")

    results = step.execute(tape, simulated_noise)

    for key in mitigation.all_combinations(3):
        if key not in expected:
            assert abs(results[key]) < TOLERANCE
            continue
        assert abs(expected[key] - results[key]) < COUNT_ACCEPTANCE

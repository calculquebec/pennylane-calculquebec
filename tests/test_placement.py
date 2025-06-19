from pennylane_calculquebec.processing.steps.placement import ISMAGS, ASTAR, VF2
import pytest
from unittest.mock import patch
from pennylane.tape import QuantumTape
import pennylane as qml
import networkx as nx
import numpy as np
from pennylane_calculquebec.utility.api import keys


@pytest.fixture
def mock_machine_graph():
    with patch("pennylane_calculquebec.utility.graph.machine_graph") as mock:
        mock.return_value = nx.Graph([(4, 0), (0, 1), (1, 2), (2, 3)])
        yield mock


@pytest.fixture
def mock_broken_qubit_and_couplers():
    with patch(
        "pennylane_calculquebec.utility.graph.get_broken_qubits_and_couplers"
    ) as mock:
        mock.return_value = {"qubits": [0], "couplers": [(2, 3)]}
        yield mock


@pytest.fixture
def mock_connectivity():
    with patch("pennylane_calculquebec.utility.graph.get_connectivity") as mock:
        mock.side_effect = lambda machine_name, use_benchmark: {
            "0": [4, 0],
            "1": [0, 1],
            "2": [1, 2],
            "3": [2, 3],
            "4": [1, 4],
        }
        yield mock


@pytest.fixture
def mock_get_readout1_and_cz_fidelities():
    with patch(
        "pennylane_calculquebec.utility.graph.get_readout1_and_cz_fidelities"
    ) as mock:
        mock.return_value = {
            keys.READOUT_STATE_1_FIDELITY: {
                "0": 0,
                "1": 0.9,
                "2": 0.8,
                "3": 0.9,
                "4": 0.8,
            },
            keys.CZ_GATE_FIDELITY: {
                (4, 0): 0.9,
                (0, 1): 0.9,
                (1, 2): 0.9,
                (2, 3): 0,
                (1, 4): 0.9,
            },
        }
        yield mock


# trivial

# default


# vf2
def test_vf2_trivial_default(mock_broken_qubit_and_couplers, mock_connectivity):
    step = VF2("yamaska")
    tape = QuantumTape(ops=[qml.CNOT([0, 1])])

    new_tape = step.execute(tape)
    wires = np.array(sorted([w for w in new_tape.wires]))
    assert np.array_equal(wires, [1, 2])


# ismags
def test_ismags_trivial_default(mock_broken_qubit_and_couplers, mock_connectivity):
    step = ISMAGS("yamaska")
    tape = QuantumTape(ops=[qml.CNOT([0, 1])])

    new_tape = step.execute(tape)
    wires = np.array(sorted([w for w in new_tape.wires]))
    assert np.array_equal(wires, [1, 2])


# astar
def test_astar_trivial_default(
    mock_broken_qubit_and_couplers,
    mock_get_readout1_and_cz_fidelities,
    mock_connectivity,
):
    step = ASTAR("yamaska")
    tape = QuantumTape(ops=[qml.CNOT([0, 1])])

    new_tape = step.execute(tape)
    wires = np.array(sorted([w for w in new_tape.wires]))
    assert np.array_equal(wires, [1, 2])


# no_benchmark


# vf2
def test_vf2_trivial_no_benchmark(mock_broken_qubit_and_couplers, mock_connectivity):
    step = VF2("yamaska", False)
    tape = QuantumTape(ops=[qml.CNOT([0, 1])])

    new_tape = step.execute(tape)
    wires = np.array(sorted([w for w in new_tape.wires]))
    assert np.array_equal(wires, [0, 4])


# ismags
def test_ismags_trivial_no_benchmark(mock_broken_qubit_and_couplers, mock_connectivity):
    step = ISMAGS("yamaska", False)
    tape = QuantumTape(ops=[qml.CNOT([0, 1])])

    new_tape = step.execute(tape)
    wires = np.array(sorted([w for w in new_tape.wires]))
    assert np.array_equal(wires, [0, 1])


# astar
def test_astar_trivial_no_benchmark(
    mock_broken_qubit_and_couplers,
    mock_get_readout1_and_cz_fidelities,
    mock_connectivity,
):
    step = ASTAR("yamaska", False)
    tape = QuantumTape(ops=[qml.CNOT([0, 1])])

    new_tape = step.execute(tape)
    wires = np.array(sorted([w for w in new_tape.wires]))
    assert np.array_equal(wires, [0, 4])


# excluded_qubits_and_couplers


# vf2
def test_vf2_trivial_excluded(mock_connectivity):
    step = VF2("yamaska", False, excluded_qubits=[0], excluded_couplers=[(2, 3)])
    tape = QuantumTape(ops=[qml.CNOT([0, 1])])

    new_tape = step.execute(tape)
    wires = np.array(sorted([w for w in new_tape.wires]))
    assert np.array_equal(wires, [1, 2])


# ismags
def test_ismags_trivial_excluded(mock_connectivity):
    step = ISMAGS("yamaska", False, excluded_qubits=[0], excluded_couplers=[(2, 3)])
    tape = QuantumTape(ops=[qml.CNOT([0, 1])])

    new_tape = step.execute(tape)
    wires = np.array(sorted([w for w in new_tape.wires]))
    assert np.array_equal(wires, [1, 2])


# astar
def test_astar_trivial_excluded(mock_get_readout1_and_cz_fidelities, mock_connectivity):
    step = ASTAR("yamaska", False, excluded_qubits=[0], excluded_couplers=[(2, 3)])
    tape = QuantumTape(ops=[qml.CNOT([0, 1])])

    new_tape = step.execute(tape)
    wires = np.array(sorted([w for w in new_tape.wires]))
    assert np.array_equal(wires, [1, 2])


# complex

# default


# vf2
def test_vf2_complex_default(mock_broken_qubit_and_couplers, mock_connectivity):
    step = VF2("yamaska")
    tape = QuantumTape(ops=[qml.CNOT([0, 1]), qml.CNOT([1, 2])])

    new_tape = step.execute(tape)
    wires = np.array(sorted([w for w in new_tape.wires]))
    assert np.array_equal(wires, [1, 2, 4])


# ismags
def test_ismags_complex_default(mock_broken_qubit_and_couplers, mock_connectivity):
    step = ISMAGS("yamaska")
    tape = QuantumTape(ops=[qml.CNOT([0, 1]), qml.CNOT([1, 2])])

    new_tape = step.execute(tape)
    wires = np.array(sorted([w for w in new_tape.wires]))
    assert np.array_equal(wires, [1, 2, 4])


# astar
def test_astar_complex_default(
    mock_broken_qubit_and_couplers,
    mock_get_readout1_and_cz_fidelities,
    mock_connectivity,
):
    step = ASTAR("yamaska")
    tape = QuantumTape(ops=[qml.CNOT([0, 1]), qml.CNOT([1, 2])])

    new_tape = step.execute(tape)
    wires = np.array(sorted([w for w in new_tape.wires]))
    assert np.array_equal(wires, [1, 2, 4])


# no_benchmark


# vf2
def test_vf2_complex_no_benchmark(mock_broken_qubit_and_couplers, mock_connectivity):
    step = VF2("yamaska", False)
    tape = QuantumTape(ops=[qml.CNOT([0, 1]), qml.CNOT([1, 2])])

    new_tape = step.execute(tape)
    wires = np.array(sorted([w for w in new_tape.wires]))
    assert np.array_equal(wires, [0, 1, 4])


# ismags
def test_ismags_complex_no_benchmark(mock_broken_qubit_and_couplers, mock_connectivity):
    step = ISMAGS("yamaska", False)
    tape = QuantumTape(ops=[qml.CNOT([0, 1]), qml.CNOT([1, 2])])

    new_tape = step.execute(tape)
    wires = np.array(sorted([w for w in new_tape.wires]))
    assert np.array_equal(wires, [0, 1, 2])


# astar
def test_astar_complex_no_benchmark(
    mock_broken_qubit_and_couplers,
    mock_get_readout1_and_cz_fidelities,
    mock_connectivity,
):
    step = ASTAR("yamaska", False)
    tape = QuantumTape(ops=[qml.CNOT([0, 1]), qml.CNOT([1, 2])])

    new_tape = step.execute(tape)
    wires = np.array(sorted([w for w in new_tape.wires]))
    assert np.array_equal(wires, [0, 1, 4])


# excluded_qubits_and_couplers


# vf2
def test_vf2_complex_excluded(mock_connectivity):
    step = VF2("yamaska", False, excluded_qubits=[0], excluded_couplers=[(2, 3)])
    tape = QuantumTape(ops=[qml.CNOT([0, 1]), qml.CNOT([1, 2])])

    new_tape = step.execute(tape)
    wires = np.array(sorted([w for w in new_tape.wires]))
    assert np.array_equal(wires, [1, 2, 4])


# ismags
def test_ismags_complex_excluded(mock_connectivity):
    step = ISMAGS("yamaska", False, excluded_qubits=[0], excluded_couplers=[(2, 3)])
    tape = QuantumTape(ops=[qml.CNOT([0, 1]), qml.CNOT([1, 2])])

    new_tape = step.execute(tape)
    wires = np.array(sorted([w for w in new_tape.wires]))
    assert np.array_equal(wires, [1, 2, 4])


# astar
def test_astar_complex_excluded(mock_get_readout1_and_cz_fidelities, mock_connectivity):
    step = ASTAR("yamaska", False, excluded_qubits=[0], excluded_couplers=[(2, 3)])
    tape = QuantumTape(ops=[qml.CNOT([0, 1]), qml.CNOT([1, 2])])

    new_tape = step.execute(tape)
    wires = np.array(sorted([w for w in new_tape.wires]))
    assert np.array_equal(wires, [1, 2, 4])


# impossible

# default


# vf2
def test_vf2_impossible_default(mock_broken_qubit_and_couplers, mock_connectivity):
    step = VF2("yamaska")
    tape = QuantumTape(ops=[qml.CNOT([0, 1]), qml.CNOT([1, 2]), qml.CNOT([2, 3])])

    with pytest.raises(Exception):
        new_tape = step.execute(tape)


# ismags
def test_ismags_impossible_default(mock_broken_qubit_and_couplers, mock_connectivity):
    step = ISMAGS("yamaska")
    tape = QuantumTape(ops=[qml.CNOT([0, 1]), qml.CNOT([1, 2]), qml.CNOT([2, 3])])

    with pytest.raises(Exception):
        new_tape = step.execute(tape)


# astar
def test_astar_impossible_default(
    mock_broken_qubit_and_couplers,
    mock_get_readout1_and_cz_fidelities,
    mock_connectivity,
):
    step = ASTAR("yamaska")
    tape = QuantumTape(ops=[qml.CNOT([0, 1]), qml.CNOT([1, 2]), qml.CNOT([2, 3])])

    with pytest.raises(Exception):
        new_tape = step.execute(tape)


# no_benchmark


# vf2
def test_vf2_impossible_no_benchmark(mock_broken_qubit_and_couplers, mock_connectivity):
    step = VF2("yamaska", False)
    tape = QuantumTape(ops=[qml.CNOT([0, 1]), qml.CNOT([1, 2]), qml.CNOT([2, 3])])

    new_tape = step.execute(tape)
    wires = np.array(sorted([w for w in new_tape.wires]))
    assert np.array_equal(wires, [0, 1, 2, 4])


# ismags
def test_ismags_impossible_no_benchmark(
    mock_broken_qubit_and_couplers, mock_connectivity
):
    step = ISMAGS("yamaska", False)
    tape = QuantumTape(ops=[qml.CNOT([0, 1]), qml.CNOT([1, 2]), qml.CNOT([2, 3])])

    new_tape = step.execute(tape)
    wires = np.array(sorted([w for w in new_tape.wires]))
    assert np.array_equal(wires, [0, 1, 2, 3])


# astar
def test_astar_impossible_no_benchmark(
    mock_broken_qubit_and_couplers,
    mock_get_readout1_and_cz_fidelities,
    mock_connectivity,
):
    step = ASTAR("yamaska", False)
    tape = QuantumTape(ops=[qml.CNOT([0, 1]), qml.CNOT([1, 2]), qml.CNOT([2, 3])])

    new_tape = step.execute(tape)
    wires = np.array(sorted([w for w in new_tape.wires]))
    assert np.array_equal(wires, [0, 1, 2, 4])


# excluded_qubits_and_couplers


# vf2
def test_vf2_impossible_excluded(mock_connectivity):
    step = VF2("yamaska", False, excluded_qubits=[0], excluded_couplers=[(2, 3)])
    tape = QuantumTape(ops=[qml.CNOT([0, 1]), qml.CNOT([1, 2]), qml.CNOT([2, 3])])

    with pytest.raises(Exception):
        new_tape = step.execute(tape)


# ismags
def test_ismags_impossible_excluded(mock_connectivity):
    step = ISMAGS("yamaska", False, excluded_qubits=[0], excluded_couplers=[(2, 3)])
    tape = QuantumTape(ops=[qml.CNOT([0, 1]), qml.CNOT([1, 2]), qml.CNOT([2, 3])])

    with pytest.raises(Exception):
        new_tape = step.execute(tape)


# astar
def test_astar_impossible_excluded(
    mock_get_readout1_and_cz_fidelities, mock_connectivity
):
    step = ASTAR("yamaska", False, excluded_qubits=[0], excluded_couplers=[(2, 3)])
    tape = QuantumTape(ops=[qml.CNOT([0, 1]), qml.CNOT([1, 2]), qml.CNOT([2, 3])])

    with pytest.raises(Exception):
        new_tape = step.execute(tape)


# less qubits than total wires


# vf2
def test_vf2_impossible_excluded(
    mock_broken_qubit_and_couplers,
    mock_get_readout1_and_cz_fidelities,
    mock_connectivity,
):
    step = VF2("yamaska", False)
    tape = QuantumTape(
        ops=[qml.Hadamard(0), qml.CNOT([0, 1]), qml.CNOT([1, 2])],
        measurements=[qml.counts(wires=[0, 1, 2, 3])],
    )

    new_tape = step.execute(tape)
    wires = np.array(sorted([w for w in new_tape.wires]))
    assert np.array_equal(wires, [0, 1, 2, 4])


# ismags
def test_ismags_impossible_excluded(mock_connectivity):
    step = ISMAGS("yamaska", False)
    tape = QuantumTape(
        ops=[qml.Hadamard(0), qml.CNOT([0, 1]), qml.CNOT([1, 2])],
        measurements=[qml.counts(wires=[0, 1, 2, 3])],
    )

    new_tape = step.execute(tape)
    wires = np.array(sorted([w for w in new_tape.wires]))
    assert np.array_equal(wires, [0, 1, 2, 4])


# astar
def test_astar_impossible_excluded(
    mock_get_readout1_and_cz_fidelities, mock_connectivity
):
    step = ASTAR("yamaska", False)
    tape = QuantumTape(
        ops=[qml.Hadamard(0), qml.CNOT([0, 1]), qml.CNOT([1, 2])],
        measurements=[qml.counts(wires=[0, 1, 2, 3])],
    )

    new_tape = step.execute(tape)
    wires = np.array(sorted([w for w in new_tape.wires]))
    assert np.array_equal(wires, [0, 1, 2, 4])

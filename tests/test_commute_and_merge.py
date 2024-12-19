from pennylane.tape import QuantumTape
import pennylane as qml
import pytest
from unittest.mock import patch
import pennylane_calculquebec.processing.optimization_methods.iterative_commute_and_merge as iterative_commute_and_merge
import pennylane.transforms as transforms
import numpy as np

file_path = "pennylane_calculquebec.processing.optimization_methods.iterative_commute_and_merge"

@pytest.fixture
def mock_commute_controlled():
    with patch("pennylane.transforms.commute_controlled") as mock:
        yield mock

@pytest.fixture
def mock_remove_root_zs():
    with patch(f"{file_path}.remove_root_zs") as mock:
        yield mock

@pytest.fixture
def mock_remove_leaf_zs():
    with patch(f"{file_path}.remove_leaf_zs") as mock:
        yield mock

@pytest.fixture
def mock_cancel_inverses():
    with patch("pennylane.transforms.cancel_inverses") as mock:
        yield mock

@pytest.fixture
def mock_merge_rotations():
    with patch("pennylane.transforms.merge_rotations") as mock:
        yield mock

@pytest.fixture
def mock_remove_trivials():
    with patch(f"{file_path}._remove_trivials") as mock:
        yield mock

def test_remove_root_zs():
    tape = QuantumTape([qml.Z(0), qml.X(0), qml.X(1), qml.Z(1)], [], 1000)
    tape = iterative_commute_and_merge.remove_root_zs(tape)
    assert tape.operations == [qml.X(0), qml.X(1), qml.Z(1)]
    
    tape = iterative_commute_and_merge.remove_root_zs(tape)
    assert tape.operations == [qml.X(0), qml.X(1), qml.Z(1)]

def test_remove_leaf_zs():
    tape = QuantumTape([qml.Z(0), qml.X(0), qml.X(1), qml.Z(1)], [], 1000)
    tape = iterative_commute_and_merge.remove_leaf_zs(tape)
    assert tape.operations == [qml.Z(0), qml.X(0), qml.X(1)]
    
    tape.operations.append(qml.RZ(3.14, 0))
    tape = iterative_commute_and_merge.remove_leaf_zs(tape)
    assert tape.operations == [qml.Z(0), qml.X(0), qml.X(1)]

def test_remove_trivials():
    tape = QuantumTape([qml.RZ(0, 0), qml.Z(0), qml.RX(0, 0), qml.RY(3.14, 0), qml.RY(0, 0), qml.X(0)])
    tape = iterative_commute_and_merge._remove_trivials(tape)
    assert tape.operations == [qml.Z(0), qml.RY(3.14, 0), qml.X(0)]


def test_commute_and_merge():
    # test bernstein vazirani
    tape = QuantumTape([qml.RZ(np.pi/2, 0), qml.RZ(np.pi/2, 1), qml.RZ(np.pi/2, 2), qml.RZ(np.pi/2, 3),
                        qml.RX(np.pi/2, 0), qml.RX(np.pi/2, 1), qml.RX(np.pi/2, 2), qml.RX(np.pi/2, 3),
                        qml.RZ(np.pi/2, 0), qml.RZ(np.pi/2, 1), qml.RZ(np.pi/2, 2), qml.RZ(np.pi/2, 3),
                        qml.RZ(np.pi, 3), qml.RZ(np.pi/2, 3), qml.RX(np.pi/2, 3), qml.RZ(np.pi/2, 3),
                        qml.CZ([2, 3]), qml.RZ(np.pi/2, 3), qml.RX(np.pi/2, 3), qml.RZ(np.pi/2, 3),
                        qml.RZ(np.pi/2, 3), qml.RX(np.pi/2, 3), qml.RZ(np.pi/2, 3), qml.CZ([0, 3]),
                        qml.RZ(np.pi/2, 3), qml.RX(np.pi/2, 3), qml.RZ(np.pi/2, 3),
                        qml.RZ(np.pi/2, 0), qml.RZ(np.pi/2, 1), qml.RZ(np.pi/2, 2),
                        qml.RX(np.pi/2, 0), qml.RX(np.pi/2, 1), qml.RX(np.pi/2, 2),
                        qml.RZ(np.pi/2, 0), qml.RZ(np.pi/2, 1), qml.RZ(np.pi/2, 2)], [], 1000)
    solution = [qml.RX(np.pi/2, 0), qml.RX(np.pi/2, 1), qml.RX(np.pi/2, 2), qml.RX(np.pi, 3),
                qml.RZ(np.pi, 1), qml.RZ(np.pi, 2), qml.RZ(np.pi, 3), qml.CZ([2, 3]), 
                qml.RX(np.pi/2, 3), qml.RZ(np.pi, 3), qml.RX(np.pi/2, 3), qml.RZ(np.pi, 0),
                qml.RZ(np.pi, 3), qml.CZ([0, 3]), qml.RX(np.pi/2, 3), qml.RX(np.pi/2, 0), 
                qml.RX(np.pi/2, 1), qml.RX(np.pi/2, 2)]
    tape = iterative_commute_and_merge.commute_and_merge(tape)
    assert tape.operations == solution

def test_commute_and_merge_mock(mock_commute_controlled, mock_remove_root_zs, mock_remove_leaf_zs, 
                           mock_cancel_inverses, mock_merge_rotations, mock_remove_trivials):
    
    tape = QuantumTape([], [], 1000)
    tape = iterative_commute_and_merge.commute_and_merge(tape)
    mock_commute_controlled.assert_called()
    mock_remove_root_zs.assert_called()
    mock_remove_leaf_zs.assert_called()
    mock_cancel_inverses.assert_called()
    mock_merge_rotations.assert_called()
    mock_remove_trivials.assert_called()
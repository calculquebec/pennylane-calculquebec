import pytest
from unittest.mock import patch
from pennylane_calculquebec.utility.api import keys
import pennylane_calculquebec.monarq_data as data
from pennylane_calculquebec.API.adapter import ApiAdapter

epsilon = 1e-8

@pytest.fixture
def mock_depolarizing_noise():
    with patch("pennylane_calculquebec.utility.noise.depolarizing_noise") as depolarizing_noise:
        yield depolarizing_noise


@pytest.fixture
def mock_phase_damping():
    with patch("pennylane_calculquebec.utility.noise.phase_damping") as phase_damping:
        yield phase_damping

        
@pytest.fixture
def mock_amplitude_damping():
    with patch("pennylane_calculquebec.utility.noise.amplitude_damping") as amplitude_damping:
        yield amplitude_damping

        
@pytest.fixture
def mock_get_qubits_and_couplers():
    with patch("pennylane_calculquebec.API.adapter.ApiAdapter.get_qubits_and_couplers") as get_qubits_and_couplers:
        connectivity = data.connectivity
        get_qubits_and_couplers.return_value = {keys.QUBITS : {}, keys.COUPLERS : {}}
        for qubit in connectivity[keys.QUBITS]:
            get_qubits_and_couplers.return_value[keys.QUBITS][str(qubit)] = {keys.READOUT_STATE_1_FIDELITY : 0.8, keys.READOUT_STATE_0_FIDELITY : 0.8, keys.SINGLE_QUBIT_GATE_FIDELITY : 0.8, keys.T1 : 1e-5, keys.T2_RAMSEY : 1e-5}
        for coupler in connectivity[keys.COUPLERS]:
            get_qubits_and_couplers.return_value[keys.COUPLERS][coupler] = {keys.CZ_GATE_FIDELITY : 0.8}
            
        yield get_qubits_and_couplers


@pytest.fixture
def mock_is_last_update_expired():
    with patch("pennylane_calculquebec.API.adapter.ApiAdapter.is_last_update_expired") as is_last_update_expired:
        yield is_last_update_expired


def set_benchmark(qubits_and_couplers, **kwargs):
    for key in kwargs:
        k = key[0]
        v = key[1:]
        if k == "q":
            qubits_and_couplers.return_value[keys.QUBITS][v][keys.READOUT_STATE_1_FIDELITY] = kwargs[key]
        else:
            qubits_and_couplers.return_value[keys.COUPLERS][v][keys.CZ_GATE_FIDELITY] = kwargs[key]    
            

def test_get_broken_qubits_and_couplers(mock_get_qubits_and_couplers):
    # nothing is broken
    results = data.get_broken_qubits_and_couplers(0.5, 0.5)
    assert len(results[keys.QUBITS]) == 0
    assert len(results[keys.COUPLERS]) == 0
    
    # qubit 4 is broken
    set_benchmark(mock_get_qubits_and_couplers, q4 = 0)
    results = data.get_broken_qubits_and_couplers(0.5, 0.5)
    assert results[keys.QUBITS][0] == 4
    assert len(results[keys.COUPLERS]) == 0
    
    # coupler 3 is broken
    set_benchmark(mock_get_qubits_and_couplers, q4 = 1, c3 = 0)
    results = data.get_broken_qubits_and_couplers(0.5, 0.5)
    # coupler 3 ==> link from node 5 to node 2
    assert results[keys.COUPLERS][0] == [5, 2] 
    assert len(results[keys.QUBITS]) == 0


def test_get_readout1_and_cz_fidelities(mock_get_qubits_and_couplers, mock_is_last_update_expired):
    # before caching
    results = data.get_readout1_and_cz_fidelities()
    assert all([key in results for key in [keys.READOUT_STATE_1_FIDELITY, keys.CZ_GATE_FIDELITY]])
    assert len(results[keys.READOUT_STATE_1_FIDELITY]) == 24
    assert len(results[keys.CZ_GATE_FIDELITY]) == 35
    
    # after caching
    mock_is_last_update_expired.return_value = False
    set_benchmark(mock_get_qubits_and_couplers, q4 = 0)
    results = data.get_readout1_and_cz_fidelities()
    assert abs(results[keys.READOUT_STATE_1_FIDELITY]["4"] - 0.8) < epsilon
    
    # cache is expired
    mock_is_last_update_expired.return_value = True
    results = data.get_readout1_and_cz_fidelities()
    assert results[keys.READOUT_STATE_1_FIDELITY]["4"] == 0


def test_get_coupler_noise(mock_is_last_update_expired, mock_get_qubits_and_couplers):
    # before cache
    results = data.get_coupler_noise()
    mock_get_qubits_and_couplers.assert_called_once()
    assert len(results) == 35

    # after cache
    mock_get_qubits_and_couplers.reset_mock()
    mock_is_last_update_expired.return_value = False
    results2 = data.get_coupler_noise()
    mock_get_qubits_and_couplers.assert_not_called()
    assert results is results2

    # cache is expired
    mock_get_qubits_and_couplers.reset_mock()
    mock_is_last_update_expired.return_value = True
    results3 = data.get_coupler_noise()
    mock_get_qubits_and_couplers.assert_called_once()
    assert results is not results3

def test_get_qubit_noise(mock_is_last_update_expired, mock_get_qubits_and_couplers):
    # before cache
    results = data.get_qubit_noise()
    mock_get_qubits_and_couplers.assert_called_once()
    assert len(results) == 24

    # after cache
    mock_get_qubits_and_couplers.reset_mock()
    mock_is_last_update_expired.return_value = False
    results2 = data.get_qubit_noise()
    mock_get_qubits_and_couplers.assert_not_called()
    assert results is results2
    
    # cache is expired
    mock_get_qubits_and_couplers.reset_mock()
    mock_is_last_update_expired.return_value = True
    results3 = data.get_qubit_noise()
    mock_get_qubits_and_couplers.assert_called_once()
    assert results is not results3
    

def test_get_phase_damping(mock_is_last_update_expired, mock_get_qubits_and_couplers):
    # before cache
    results = data.get_phase_damping()
    mock_get_qubits_and_couplers.assert_called_once()
    assert len(results) == 24
    
    # after cache
    mock_get_qubits_and_couplers.reset_mock()
    mock_is_last_update_expired.return_value = False
    results2 = data.get_phase_damping()
    mock_get_qubits_and_couplers.assert_not_called()
    assert results is results2
    
    # cache is expired
    mock_get_qubits_and_couplers.reset_mock()
    mock_is_last_update_expired.return_value = True
    results3 = data.get_phase_damping()
    mock_get_qubits_and_couplers.assert_called_once()
    assert results is not results3
    

def test_get_amplitude_damping(mock_is_last_update_expired, mock_get_qubits_and_couplers):
    # before cache
    results = data.get_amplitude_damping()
    mock_get_qubits_and_couplers.assert_called_once()
    assert len(results) == 24
    
    # after cache
    mock_get_qubits_and_couplers.reset_mock()
    mock_is_last_update_expired.return_value = False
    results2 = data.get_amplitude_damping()
    mock_get_qubits_and_couplers.assert_not_called()
    assert results is results2
    
    # cache is expired
    mock_get_qubits_and_couplers.reset_mock()
    mock_is_last_update_expired.return_value = True
    results3 = data.get_amplitude_damping()
    mock_get_qubits_and_couplers.assert_called_once()
    assert results is not results3
    

def test_get_readout_noise_matrices(mock_is_last_update_expired, mock_get_qubits_and_couplers):
    # before cache
    results = data.get_readout_noise_matrices()
    mock_get_qubits_and_couplers.assert_called_once()
    assert len(results) == 24
    
    # after cache
    mock_get_qubits_and_couplers.reset_mock()
    mock_is_last_update_expired.return_value = False
    results2 = data.get_readout_noise_matrices()
    mock_get_qubits_and_couplers.assert_not_called()
    assert results is results2
    
    # cache is expired
    mock_get_qubits_and_couplers.reset_mock()
    mock_is_last_update_expired.return_value = True
    results3 = data.get_readout_noise_matrices()
    mock_get_qubits_and_couplers.assert_called_once()
    assert results is not results3

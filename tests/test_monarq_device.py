import pytest
from unittest.mock import patch
from pennylane_calculquebec.monarq_device import MonarqDevice, DeviceException
from pennylane_calculquebec.API.client import MonarqClient
from pennylane_calculquebec.processing.config import MonarqDefaultConfig, NoPlaceNoRouteConfig
from pennylane_calculquebec.processing import PreProcessor
from pennylane.transforms import transform
from pennylane.tape import QuantumTape
import pennylane as qml
import pennylane_calculquebec.API.job as api_job


client = MonarqClient("host", "user", "token")

@pytest.fixture
def mock_measure():
    with patch("pennylane_calculquebec.monarq_device.MonarqDevice._measure") as meas:
        yield meas

@pytest.fixture
def mock_default_config():
    with patch("pennylane_calculquebec.processing.config.MonarqDefaultConfig") as default_config:
        yield default_config

@pytest.fixture
def mock_api_initialize():
    with patch("pennylane_calculquebec.API.adapter.ApiAdapter.initialize") as initialize:
        yield initialize

@pytest.fixture
def mock_PostProcessor_get_processor():
    with patch("pennylane_calculquebec.processing.PostProcessor.get_processor") as proc:
        yield proc

@pytest.fixture
def mock_PreProcessor_get_processor():
    with patch("pennylane_calculquebec.processing.PreProcessor.get_processor") as proc:
        yield proc

def test_constructor(mock_api_initialize):
    # no shots given, should raise DeviceException
    with pytest.raises(DeviceException):
        dev = MonarqDevice()
    
    mock_api_initialize.assert_not_called()
    
    # no client given, should raise DeviceException
    with pytest.raises(DeviceException):
        dev = MonarqDevice(shots=1000)
    
    mock_api_initialize.assert_not_called()
    
    # client given, no config given, should set default config
    dev = MonarqDevice(client = client, shots=1000)
    mock_api_initialize.assert_called_once()
    assert dev.shots.total_shots == 1000
    assert dev._processing_config == MonarqDefaultConfig()
    
    # config given, should set given config
    mock_api_initialize.reset_mock()
    config = NoPlaceNoRouteConfig()
    dev = MonarqDevice(client = client, processing_config=config, shots=1000)
    mock_api_initialize.assert_called_once()
    assert dev._processing_config is config

def test_preprocess(mock_PreProcessor_get_processor, mock_api_initialize):
    mock_PreProcessor_get_processor.return_value = transform(lambda tape : tape)
    dev = MonarqDevice(client = client, shots = 1000)
    result = dev.preprocess()[0]
    assert len(result) == 1
    mock_PreProcessor_get_processor.assert_called_once()
      
def test_execute(mock_measure, mock_PostProcessor_get_processor):
    mock_PostProcessor_get_processor.return_value = lambda a, b: ["a", "b", "c"]
    dev = MonarqDevice(client=client, shots=1000)
    
    # ran 1 time
    quantum_tape = QuantumTape([], [], 1000)
    results = dev.execute(quantum_tape)
    mock_measure.assert_called_once()
    mock_PostProcessor_get_processor.assert_called_once()
    assert results == ["a", "b", "c"]
    
    # ran 4 times
    result = dev.execute([quantum_tape, quantum_tape, quantum_tape])
    assert mock_measure.call_count == 4
    assert mock_PostProcessor_get_processor.call_count == 4

def test_measure():
    class Job:
        def run(self):
            return 42
    
    quantum_tape = QuantumTape([], [], 1000)
    
    with patch("pennylane_calculquebec.API.job.Job.__new__") as job:
        job.return_value = Job()
        
        # measurement != 1, DeviceException
        with pytest.raises(DeviceException):
            MonarqDevice._measure(None, quantum_tape)
        
        job.assert_not_called()
        
        # measurement != counts
        quantum_tape.measurements.append(qml.probs())
        
        with pytest.raises(DeviceException):
            MonarqDevice._measure(None, quantum_tape)
        
        job.assert_not_called()
        
        # one count measurement only
        quantum_tape.measurements[0] = qml.counts()
        assert MonarqDevice._measure(None, quantum_tape) == 42
        
        job.assert_called_once()
        
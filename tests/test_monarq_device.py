import pytest
from unittest.mock import patch
from pennylane_calculquebec.monarq_device import MonarqDevice, DeviceException
from pennylane_calculquebec.API.client import CalculQuebecClient
from pennylane_calculquebec.processing.config import (
    MonarqDefaultConfig,
    NoPlaceNoRouteConfig,
    EmptyConfig,
)
from pennylane_calculquebec.processing import PreProcessor
from pennylane.transforms import transform
from pennylane.tape import QuantumTape
from pennylane.exceptions import PennyLaneDeprecationWarning
import pennylane as qml
from pennylane_calculquebec.base_device import BaseDevice
import pennylane_calculquebec.API.job as api_job

client = CalculQuebecClient("host", "user", "token", project_id="test_project_id")


@pytest.fixture
def mock_measure():
    with patch("pennylane_calculquebec.monarq_device.MonarqDevice._measure") as meas:
        yield meas


@pytest.fixture
def mock_default_config():
    with patch(
        "pennylane_calculquebec.processing.config.MonarqDefaultConfig"
    ) as default_config:
        yield default_config


@pytest.fixture
def mock_api_initialize():
    with patch(
        "pennylane_calculquebec.API.adapter.ApiAdapter.initialize"
    ) as initialize:
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
    dev = MonarqDevice(client=client)
    mock_api_initialize.assert_called_once()
    assert dev._processing_config == MonarqDefaultConfig("yamaska")

    # config given, should set given config
    mock_api_initialize.reset_mock()
    config = NoPlaceNoRouteConfig()
    dev = MonarqDevice(client=client, processing_config=config)
    mock_api_initialize.assert_called_once()
    assert dev._processing_config is config


def test_constructor_shots_deprecated(mock_api_initialize):
    # Passing shots via constructor should emit a deprecation warning
    with pytest.warns(PennyLaneDeprecationWarning):
        MonarqDevice(client=client, shots=1000)
    mock_api_initialize.assert_called_once()


def test_device_registers_client():
    """Test that MonarqDevice registers the client when initialized."""
    dev = MonarqDevice(client=client)
    assert hasattr(dev, "_client")
    assert isinstance(dev._client, CalculQuebecClient)


def test_preprocess(mock_PreProcessor_get_processor, mock_api_initialize):
    mock_PreProcessor_get_processor.return_value = transform(lambda tape: tape)
    dev = MonarqDevice(client=client)
    result = dev.preprocess()[0]
    assert len(result) == 1
    mock_PreProcessor_get_processor.assert_called_once()


def test_execute(mock_measure):
    mock_measure.return_value = ["a", "b", "c"]
    dev = MonarqDevice(client=client)

    # ran 1 time
    quantum_tape = QuantumTape([], [], 1000)
    results = dev.execute(quantum_tape)
    mock_measure.assert_called_once()
    assert results == ["a", "b", "c"]

    # ran 4 times
    result = dev.execute([quantum_tape, quantum_tape, quantum_tape])
    assert mock_measure.call_count == 4


def test_measure(mock_PostProcessor_get_processor):
    mock_PostProcessor_get_processor.return_value = lambda a, b: b

    class Job:
        def run(self):
            return {"0": 750, "1": 25}

    class Tape:
        def __init__(self):
            self.wires = [0]
            self.measurements = []

    class MockDevice:
        def __init__(self):
            self.machine_name = "yamaska"
            self.circuit_name = "circuit"
            self.project_name = "project"
            self._processing_config = EmptyConfig()
            self.wires = [0]
            self.job_started = None
            self.job_status_changed = None
            self.job_completed = None

    dev = MockDevice()
    expected_counts = Job().run()
    expected_probs = [750 / 775, 25 / 775]
    expected_expectation = 0.935483870967742

    quantum_tape = Tape()

    with patch("pennylane_calculquebec.API.job.Job.__new__") as job:
        job.return_value = Job()

        # measurement != 1, DeviceException
        with pytest.raises(DeviceException):
            MonarqDevice._measure(dev, quantum_tape)

        job.assert_not_called()

        # invalid measurement
        quantum_tape.measurements.append(qml.sample())
        with pytest.raises(DeviceException):
            _ = MonarqDevice._measure(dev, quantum_tape)
        job.assert_not_called()

        # measurement is probs
        quantum_tape.measurements[0] = qml.probs()
        probs = MonarqDevice._measure(dev, quantum_tape)
        tolerance = 1e-5
        assert all(abs(a - b) < tolerance for a, b in zip(probs, expected_probs))

        job.assert_called_once()

        # measurement is counts
        quantum_tape.measurements[0] = qml.counts()
        counts = MonarqDevice._measure(dev, quantum_tape)
        assert counts == expected_counts

        # since the method has been called one time before, the call count is incremented to 2
        assert job.call_count == 2

        # measurement is expval
        quantum_tape.measurements[0] = qml.expval(qml.PauliZ(0))
        expval = MonarqDevice._measure(dev, quantum_tape)
        assert expval == expected_expectation

        # since the method has been called one time before, the call count is incremented to 2
        assert job.call_count == 3

        # too many measurements
        quantum_tape.measurements.append(qml.counts())
        with pytest.raises(DeviceException):
            _ = MonarqDevice._measure(dev, quantum_tape)

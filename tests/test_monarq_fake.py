import pytest
from unittest.mock import patch
from pennylane_calculquebec.monarq_sim import MonarqSim, MonarqDefaultConfig
from pennylane_calculquebec.monarq_device import DeviceException
from pennylane_calculquebec.API.client import CalculQuebecClient
from pennylane_calculquebec.processing.config import EmptyConfig
from pennylane_calculquebec.processing import PreProcessor
from pennylane.transforms import transform
from pennylane.tape import QuantumTape
from pennylane.exceptions import PennyLaneDeprecationWarning
import pennylane as qml
from pennylane_calculquebec.base_device import BaseDevice
import pennylane_calculquebec.API.job as api_job

client = CalculQuebecClient("test", "test", "test", project_id="test_project_id")


@pytest.fixture
def mock_measure():
    with patch("pennylane_calculquebec.monarq_sim.MonarqSim._measure") as meas:
        yield meas


@pytest.fixture
def mock_default_config():
    with patch(
        "pennylane_calculquebec.processing.config.FakeMonarqConfig"
    ) as default_config:
        yield default_config


@pytest.fixture
def mock_api_initialize():
    with patch(
        "pennylane_calculquebec.API.adapter.ApiAdapter.initialize"
    ) as initialize:
        yield initialize


@pytest.fixture
def mock_gate_noise():
    with patch("pennylane_calculquebec.monarq_sim.GateNoiseSimulation.execute") as mock:
        yield mock


@pytest.fixture
def mock_readout_noise():
    with patch(
        "pennylane_calculquebec.monarq_sim.ReadoutNoiseSimulation.execute"
    ) as mock:
        yield mock


@pytest.fixture
def mock_PostProcessor_get_processor():
    with patch("pennylane_calculquebec.processing.PostProcessor.get_processor") as proc:
        yield proc


@pytest.fixture
def mock_PreProcessor_get_processor():
    with patch("pennylane_calculquebec.processing.PreProcessor.get_processor") as proc:
        yield proc


def test_constructor(mock_api_initialize):

    # no client given, no config given, should set default config
    dev = MonarqSim()
    mock_api_initialize.assert_not_called()
    assert dev._processing_config == MonarqDefaultConfig("yamaska", False)

    MonarqSim(client=client)

    mock_api_initialize.assert_called_once()

    # config given, should set given config
    mock_api_initialize.reset_mock()
    config = EmptyConfig()
    dev = MonarqSim(client=client, processing_config=config)
    mock_api_initialize.assert_called_once()
    assert dev._processing_config is config


def test_constructor_shots_deprecated(mock_api_initialize):
    # Passing shots via constructor should emit a deprecation warning
    with pytest.warns(PennyLaneDeprecationWarning):
        MonarqSim(shots=1000)


def test_preprocess(mock_PreProcessor_get_processor):
    mock_PreProcessor_get_processor.return_value = transform(lambda tape: tape)
    dev = MonarqSim()
    result = dev.preprocess()[0]
    assert len(result) == 1
    mock_PreProcessor_get_processor.assert_called_once()


def test_execute(mock_measure):
    mock_measure.return_value = ["a", "b", "c"]
    dev = MonarqSim(client=client)

    # ran 1 time
    quantum_tape = QuantumTape([], [], 1000)
    results = dev.execute(quantum_tape)
    mock_measure.assert_called_once()
    assert results == ["a", "b", "c"]

    # ran 4 times
    result = dev.execute([quantum_tape, quantum_tape, quantum_tape])
    assert mock_measure.call_count == 4


def test_monarqsim_without_client():
    """Test that MonarqSim does not require a client."""
    sim = MonarqSim()
    assert not hasattr(sim, "_client")


def test_measure(mock_PostProcessor_get_processor, mock_gate_noise, mock_readout_noise):
    from pennylane.tape import QuantumTape

    mock_gate_noise.side_effect = lambda tape: tape
    mock_readout_noise.side_effect = lambda tape, result: result

    mock_PostProcessor_get_processor.return_value = lambda a, b: b

    dev = MonarqSim([0], client=None, processing_config=EmptyConfig())
    expected_counts = {"0": 968, "1": 32}
    expected_probs = [968 / 1000, 32 / 1000]
    expected_expectation = 0.936

    quantum_tape = QuantumTape(ops=[qml.PauliX(0)], measurements=[])

    with patch("pennylane.execute") as job:
        with patch(
            "pennylane_calculquebec.monarq_sim.ReadoutNoiseSimulation.execute"
        ) as rns:
            rns.side_effect = lambda _, results: results
            with patch(
                "pennylane_calculquebec.monarq_sim.GateNoiseSimulation.execute"
            ) as gns:
                gns.side_effect = lambda tape: tape
                job.return_value = [expected_counts]

                # measurement != 1, DeviceException
                with pytest.raises(DeviceException):
                    MonarqSim._measure(dev, quantum_tape)

                job.assert_not_called()

                # invalid measurement
                quantum_tape = QuantumTape(
                    ops=[qml.PauliX(0)], measurements=[qml.sample()]
                )

                with pytest.raises(DeviceException):
                    _ = MonarqSim._measure(dev, quantum_tape)
                job.assert_not_called()

                # measurement is probs
                quantum_tape.measurements[0] = qml.probs(wires=[0])
                probs = MonarqSim._measure(dev, quantum_tape)
                tolerance = 1e-1
                assert all(
                    abs(a - b) < tolerance for a, b in zip(probs, expected_probs)
                )

                job.assert_called_once()

                # measurement is counts
                quantum_tape.measurements[0] = qml.counts(wires=[0])
                counts = MonarqSim._measure(dev, quantum_tape)
                assert all(
                    abs(expect - count) < tolerance * 1000
                    for expect, count in zip(counts.values(), expected_counts.values())
                )

                # since the method has been called one time before, the call count is incremented to 2
                assert job.call_count == 2

                # measurement is expval
                quantum_tape.measurements[0] = qml.expval(qml.PauliZ(0))
                expval = MonarqSim._measure(dev, quantum_tape)
                assert abs(expval - expected_expectation) < tolerance

                # since the method has been called one time before, the call count is incremented to 2
                assert job.call_count == 3

                # too many measurements
                quantum_tape.measurements.append(qml.counts())
                with pytest.raises(DeviceException):
                    _ = MonarqSim._measure(dev, quantum_tape)

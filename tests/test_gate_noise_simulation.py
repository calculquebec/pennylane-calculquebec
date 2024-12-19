import pytest
from unittest.mock import patch
from pennylane_calculquebec.processing.steps import GateNoiseSimulation
from pennylane_calculquebec.utility.noise import TypicalBenchmark
from pennylane_calculquebec.monarq_data import connectivity
from pennylane.tape import QuantumTape
import pennylane as qml
import pennylane_calculquebec.utility.noise as noise

class FakeStep:
    def __init__(self, use_benchmark):
        self.use_benchmark = use_benchmark
    
    @property
    def native_gates(self):
        """the set of monarq-native gates"""
        return  [
            "T", "TDagger",
            "PauliX", "PauliY", "PauliZ", 
            "X90", "Y90", "Z90",
            "XM90", "YM90", "ZM90",
            "PhaseShift", "CZ", "RZ"
        ]
    
        
@pytest.fixture
def mock_get_qubit_noise():
    with patch("pennylane_calculquebec.monarq_data.get_qubit_noise") as mock:
        yield mock

@pytest.fixture
def mock_get_coupler_noise():
    with patch("pennylane_calculquebec.monarq_data.get_coupler_noise") as mock2:
        yield mock2

@pytest.fixture
def mock_get_amplitude_damping():
    with patch("pennylane_calculquebec.monarq_data.get_amplitude_damping") as mock3:
        yield mock3

@pytest.fixture
def mock_get_phase_damping():
    with patch("pennylane_calculquebec.monarq_data.get_phase_damping") as mock4:
        yield mock4

def test_execute(mock_get_qubit_noise, 
                 mock_get_coupler_noise, 
                 mock_get_amplitude_damping,
                 mock_get_phase_damping):
    
    # use benchmark, noise should be reciprocal to given benchmark
    mock_get_qubit_noise.return_value = [0.1 for _ in range(4)]
    links = [(0, 1), (1, 2), (2, 3)]
    mock_get_coupler_noise.return_value = {l : 0.2 for i, l in enumerate(links)}
    
    mock_get_amplitude_damping.return_value = [0.3 for _ in range(4)]
    mock_get_phase_damping.return_value = [0.4 for _ in range(4)]
    
    tape = QuantumTape([qml.X(0), qml.Z(1), qml.CZ([2 ,3])], [], 1000)
    tape = GateNoiseSimulation.execute(FakeStep(True), tape)
    
    assert qml.DepolarizingChannel(0.2, 2) in tape.operations
    assert qml.DepolarizingChannel(0.2, 3) in tape.operations
    assert qml.DepolarizingChannel(0.1, 1) in tape.operations
    assert qml.AmplitudeDamping(0.3, 0) in tape.operations
    assert qml.PhaseDamping(0.4, 0) in tape.operations
    
    # invalid placement raises error
    tape = QuantumTape([qml.CZ([0, 10])])
    with pytest.raises(ValueError):
        tape = GateNoiseSimulation.execute(FakeStep(True), tape)
    
    # dont use benchmark, noise should be reciprocal to benchmark
    tape = QuantumTape([qml.X(0), qml.Z(4), qml.CZ([8, 12])], [], 1000)
    tape = GateNoiseSimulation.execute(FakeStep(False), tape)
    
    assert qml.DepolarizingChannel(noise.depolarizing_noise(TypicalBenchmark.cz), 8) in tape.operations
    assert qml.DepolarizingChannel(noise.depolarizing_noise(TypicalBenchmark.cz), 12) in tape.operations
    assert qml.DepolarizingChannel(noise.depolarizing_noise(TypicalBenchmark.qubit), 4) in tape.operations
    assert qml.AmplitudeDamping(noise.amplitude_damping(1E-6, TypicalBenchmark.t1), 0) in tape.operations
    assert qml.PhaseDamping(noise.phase_damping(1E-6, TypicalBenchmark.t2Ramsey), 0) in tape.operations
    
    # invalid gate raises error
    tape = QuantumTape([qml.CNOT([0, 1])])
    with pytest.raises(ValueError):
        tape = GateNoiseSimulation.execute(FakeStep(False), tape)
    
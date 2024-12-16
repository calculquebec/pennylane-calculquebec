import pytest
from pennylane_calculquebec import measurements
from pennylane_calculquebec import PennylaneConverter
import pennylane as qml
import numpy as np
from pennylane.tape import QuantumTape

class TestPennylaneConverterClass:

    @pytest.fixture
    def client_creds(self):
        return {
            "host": "host_url",
            "user": "user",
            "access_token": "access_token",
            "project_id": "project_id",
            "realm": "realm"
        }

    @pytest.fixture
    def bell_state_operations(self):
        ops = [qml.Hadamard(wires=0), qml.CNOT(wires=[0, 1])]
        return ops

    @pytest.fixture
    def bell_state_quantum_tape(self, bell_state_operations):
        ops = bell_state_operations
        return QuantumTape(ops, [qml.expval(qml.PauliZ(0))], shots=100)

    @pytest.fixture
    def pennylane_converter_with_client(self, bell_state_quantum_tape, client_creds):
        _pennylane_circuit = bell_state_quantum_tape
        _host = client_creds["host"]
        _user = client_creds["user"]
        _access_token = client_creds["access_token"]
        _project_id = client_creds["project_id"]
        _realm = client_creds["realm"]
        _wires = [0, 1]
        return PennylaneConverter(pennylane_circuit=_pennylane_circuit, host=_host, user=_user, access_token=_access_token, project_id=_project_id, realm=_realm, wires=_wires)

    @pytest.fixture
    def pennylane_converter_without_client(self, bell_state_quantum_tape):
        _pennylane_circuit = bell_state_quantum_tape
        _wires = [0, 1]
        return PennylaneConverter(pennylane_circuit=_pennylane_circuit, wires=_wires)

    @pytest.fixture
    def mock_get_strategy(self):
        return measurements.Sample()


    @pytest.mark.xfail
    def test_quantum_tape(self):
        ops = [qml.BasisState(np.array([1,1]), wires=(0,"a"))]
        quantum_tape = QuantumTape(ops, [qml.expval(qml.PauliZ(0))])
        converter = PennylaneConverter(quantum_tape)
        assert isinstance(converter.pennylane_circuit, QuantumTape)

    
    @pytest.mark.xfail
    def test_initialization_without_client(self, bell_state_quantum_tape, pennylane_converter_without_client):
        assert pennylane_converter_without_client.pennylane_circuit == bell_state_quantum_tape
        assert pennylane_converter_without_client.wires == [0, 1]

    @pytest.mark.xfail
    def test_get_strategy(self, pennylane_converter_without_client):
        # CountsMP
        mp = qml.measurements.CountsMP()
        strategy = pennylane_converter_without_client.get_strategy(mp)
        assert isinstance(strategy, measurements.Counts)

        # SampleMP
        mp = qml.measurements.SampleMP()
        strategy = pennylane_converter_without_client.get_strategy(mp)
        assert isinstance(strategy, measurements.Sample)

        # ProbabilityMP
        mp = qml.measurements.ProbabilityMP()
        strategy = pennylane_converter_without_client.get_strategy(mp)
        assert isinstance(strategy, measurements.Probabilities)

        # ExpectationMP
        mp = qml.measurements.ExpectationMP()
        strategy = pennylane_converter_without_client.get_strategy(mp)
        assert isinstance(strategy, measurements.ExpectationValue)

        # StateMP
        mp = qml.measurements.StateMP()
        strategy = pennylane_converter_without_client.get_strategy(mp)
        assert isinstance(strategy, measurements.State)


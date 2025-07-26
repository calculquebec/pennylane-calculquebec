import pytest
from pennylane.tape import QuantumTape
from pennylane_calculquebec.API.job import Job, JobException

class DummyCircuit:
    class Shots:
        total_shots = 10
    shots = Shots()

def test_job_measure_bits_start_at_zero():
    # Simulate a circuit dict with first key not zero
    class DummyApiUtility:
        @staticmethod
        def convert_circuit(circuit):
            return {1: 'not_zero'}
    
    # Patch ApiUtility in Job
    original_api_utility = Job.__init__.__globals__['ApiUtility']
    Job.__init__.__globals__['ApiUtility'] = DummyApiUtility
    try:
        with pytest.raises(JobException, match="The circuit must start with 0"):
            Job(DummyCircuit(), 'machine', 'circuit', 'project')
    finally:
        Job.__init__.__globals__['ApiUtility'] = original_api_utility

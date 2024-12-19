"""
Contains a pre-processing step for adding noise relative to MonarQ's noise model.
"""

from pennylane_calculquebec.processing.interfaces import PreProcStep
import pennylane_calculquebec.monarq_data as data
from pennylane_calculquebec.utility.noise import TypicalBenchmark, amplitude_damping, phase_damping, depolarizing_noise
import pennylane as qml
from pennylane_calculquebec.utility.api import keys

class GateNoiseSimulation(PreProcStep):
    """
    Adds gate noise to operations from a circuit using MonarQ's noise model
    """
    def __init__(self, use_benchmark = True):
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
    
    def execute(self, tape):
        # build qubit noise from readout 1 fidelity using typical value if benchmark should not be used
        qubit_noise = data.get_qubit_noise() \
            if self.use_benchmark \
            else [depolarizing_noise(TypicalBenchmark.qubit) for _ in range(24)]
        
        # build coupler noise from cz fidelity using typical value if benchmark should not be used
        cz_noise = data.get_coupler_noise() \
            if self.use_benchmark \
            else {tuple(data.connectivity[keys.COUPLERS][str(i)]):depolarizing_noise(TypicalBenchmark.cz) for i in range(35)}
        
        # build relaxation using typical t1 if not use_benchmark
        relaxation = data.get_amplitude_damping() if self.use_benchmark \
            else [amplitude_damping(1E-6, TypicalBenchmark.t1) for _ in range(24)]
        
        # build decoherence using typical t2 if not use_benchmark
        decoherence = data.get_phase_damping() if self.use_benchmark \
            else [phase_damping(1E-6, TypicalBenchmark.t2Ramsey) for _ in range(24)]
                
        operations = []
        
        if any(operation.name not in self.native_gates for operation in tape.operations):
            raise ValueError("Your circuit should contain only MonarQ native gates. Cannot simulate noise.")
        
        for operation in tape.operations:
            if operation.num_wires != 1: # can only be a cz gate in this case
                operations.append(operation)
                noises = [noise for coupler, noise in cz_noise.items() if all(wire in coupler for wire in operation.wires)]
                if len(noises) < 1 or all(key is None for key in noises):
                        raise ValueError("Cannot find CZ gate noise for operation " + str(operation))
                    
                for wire in operation.wires:
                    operations.append(qml.DepolarizingChannel(noises[0], wires=wire))
                continue
            
            if operation.basis == "Z":
                operations.append(operation)
                for wire in operation.wires:
                    operations.append(qml.DepolarizingChannel(qubit_noise[wire], wires=wire))
                continue
            
            operations.append(operation)
            for wire in operation.wires:
                operations.append(qml.DepolarizingChannel(qubit_noise[wire], wires=wire))    
                operations.append(qml.AmplitudeDamping(relaxation[wire], wires=wire))
                operations.append(qml.PhaseDamping(decoherence[wire], wires=wire))
        
        return type(tape)(operations, tape.measurements, tape.shots)
"""
Contains a pre-processing step for adding noise relative to MonarQ's noise model.
"""

from pennylane_calculquebec.processing.interfaces import PreProcStep
import pennylane_calculquebec.monarq_data as data
from pennylane_calculquebec.utility.noise import TypicalBenchmark, amplitude_damping, phase_damping, depolarizing_noise
import pennylane as qml

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
        
        qubit_noise, cz_noise = (data.get_qubit_noise(), data.get_coupler_noise()) if self.use_benchmark \
            else ([depolarizing_noise(TypicalBenchmark.qubit) for _ in range(24)], {tuple(data.connectivity["couplers"][str(i)]):depolarizing_noise(TypicalBenchmark.cz) for i in range(35)})
            
        relaxation, decoherence = (data.get_amplitude_damping(), data.get_phase_damping()) if self.use_benchmark \
            else ([amplitude_damping(1E-6, TypicalBenchmark.t1) for _ in range(24)], [phase_damping(1E-6, TypicalBenchmark.t2Ramsey) for _ in range(24)])
                
        ops = []
        
        if any(op.name not in self.native_gates for op in tape.operations):
            raise ValueError("Your circuit should contain only MonarQ native gates. Cannot simulate noise.")
        
        for op in tape.operations:
            if op.num_wires != 1: # can only be a cz gate in this case
                ops.append(op)
                noise = [n for link, n in cz_noise.items() if all(w in link for w in op.wires)]
                if len(noise) < 1 or all(n is None for n in noise):
                        raise ValueError("Cannot find CZ gate noise for operation " + str(op))
                    
                for w in op.wires:
                    ops.append(qml.DepolarizingChannel(noise[0], wires=w))
                continue
            
            if op.basis == "Z":
                ops.append(op)
                for w in op.wires:
                    ops.append(qml.DepolarizingChannel(qubit_noise[w], wires=w))
                continue
            
            ops.append(op)
            for w in op.wires:
                ops.append(qml.DepolarizingChannel(qubit_noise[w], wires=w))    
                ops.append(qml.AmplitudeDamping(relaxation[w], wires=w))
                ops.append(qml.PhaseDamping(decoherence[w], wires=w))
        
        return type(tape)(ops, tape.measurements, tape.shots)
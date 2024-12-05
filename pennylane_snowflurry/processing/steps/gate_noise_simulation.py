from pennylane_snowflurry.processing.interfaces import PreProcStep
from pennylane_snowflurry.monarq_data import get_qubit_noise, get_coupler_noise, get_amplitude_damping, get_phase_damping
from pennylane_snowflurry.utility.noise import TypicalErrors, amplitude_damping, phase_damping
import pennylane as qml

class GateNoiseSimulation(PreProcStep):
    def __init__(self, use_benchmark = True):
        self.use_benchmark = use_benchmark
        
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
        
        qubit_noise, cz_noise = (get_qubit_noise(), get_coupler_noise()) if self.use_benchmark \
            else ([TypicalErrors.qubit for _ in range(24)], [TypicalErrors.cz for _ in range(35)])
            
        relaxation, decoherence = (get_amplitude_damping(), get_phase_damping()) if self.use_benchmark \
            else ([amplitude_damping(1E-6, TypicalErrors.t1) for _ in range(24)], [phase_damping(1E-6, TypicalErrors.t2Ramsey) for _ in range(24)])
                
        ops = []
        
        if any(op.name not in self.native_gates() for op in tape.operations):
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
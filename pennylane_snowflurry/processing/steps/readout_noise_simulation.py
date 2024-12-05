from pennylane_snowflurry.processing.interfaces import PostProcStep
from pennylane_snowflurry.monarq_data import get_readout_noise_matrices
import pennylane as qml
import numpy as np
from pennylane_snowflurry.utility.debug import get_labels
from pennylane_snowflurry.utility.noise import readout_error, TypicalErrors

class ReadoutNoiseSimulation(PostProcStep):
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
    
    def execute(self, tape, results):
        results = results[0] if not isinstance(results, dict) else results
        
        readout_error_matrices = get_readout_noise_matrices() \
            if self.use_benchmark \
            else [readout_error(TypicalErrors.readout0, TypicalErrors.readout1) for _ in range(24)]

        readout_matrix = np.identity(1)
        
        wires = []
        
        for mp in tape.measurements:
            for w in mp.wires:
                wires.append(w)
                readout_matrix = np.kron(readout_matrix, readout_error_matrices[w])

        # Apply the readout error matrix (dot product with the probability vector)
        probs = []
        all_labels = get_labels((1 << len(wires)) - 1)
        
        for label in all_labels:
            probs.append(results[label] / tape.shots.total_shots if label in results else 0)
            
        prob_after_error = np.dot(readout_matrix, probs)
        
        results_after_error = {label:np.round(prob_after_error[i] * tape.shots.total_shots) for i, label in enumerate(all_labels)}
    
        # Return the new measurement probabilities after applying the readout error
        return results_after_error
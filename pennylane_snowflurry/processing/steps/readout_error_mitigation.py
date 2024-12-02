from pennylane.tape import QuantumTape
from pennylane_snowflurry.utility.debug import get_labels
from pennylane_snowflurry.API.adapter import ApiAdapter
import json
from datetime import datetime, timedelta
import numpy as np
from pennylane_snowflurry.processing.interfaces import PostProcStep

class ReadoutErrorMitigation(PostProcStep):
    _a_normalized = None
    _a_reduced = None
    _a_reduced_inverted = None
    
    @property
    def all_combinations(self):
        return get_labels((2 ** self.num_qubits) - 1)
  
    def all_results(self, results):
        all_combs = self.all_combinations
        return {bitstring:(results[bitstring] if bitstring in results else 0) for bitstring in all_combs}
        
    def _get_readout_fidelities(self, myqubits):
        benchmark = ApiAdapter.get_qubits_and_couplers()
        complete_benchmark = ApiAdapter.get_benchmark()
        time_stamp = complete_benchmark["timeStamp"]
        time_stamp = datetime.strptime(time_stamp, '%Y-%m-%dT%H:%M:%S.%fZ')
        time_stamp = time_stamp - timedelta(hours=5)
        benchmark_time = time_stamp.strftime('%Y-%m-%d %H:%M')
        
        
        readout0 = {}
        readout1 = {}
        for i in myqubits:
            readout0[i] = benchmark["qubits"][str(i)]["readoutState0Fidelity"]
            readout1[i] = benchmark["qubits"][str(i)]["readoutState1Fidelity"]    

        readout0 = list(readout0.values())
        readout1 = list(readout1.values())
        
        return readout0, readout1

    def _tensor_product_calibration(self, calibration_matrices):
        # Initialize with the first qubit's calibration matrix
        A = calibration_matrices[0]
        
        # Tensor product of calibration matrices for all qubits
        for i in range(1, len(calibration_matrices)):
            A = np.kron(A, calibration_matrices[i])
            
        return A

    def _get_calibration_data(self, myqubits):
        state_0_readout_fidelity, state_1_readout_fidelity = self._get_readout_fidelities(myqubits)
        calibration_data = {
            k: np.array([
                [state_0_readout_fidelity[k], 1 - state_1_readout_fidelity[k]],
                [1 - state_0_readout_fidelity[k], state_1_readout_fidelity[k]]
            ]) for k in range(self.num_qubits)
        }
        return calibration_data

    def _get_reduced_a_matrix(self, A_full, observed_bit_strings, all_bit_strings):
        # Convert observed bit strings to their integer indices
        observed_indices = [all_bit_strings.index(bit_str) for bit_str in observed_bit_strings]
        
        # Extract the reduced A-matrix from the full A-matrix
        A_reduced = A_full[np.ix_(observed_indices, observed_indices)]
        
        return A_reduced

    def _get_inverted_reduced_a_matrix(self, myqubits, results):
        # Generate the full A-matrix
        if ReadoutErrorMitigation._a_normalized is None or ApiAdapter.is_last_update_expired():
            ReadoutErrorMitigation._a_reduced = None
            ReadoutErrorMitigation._a_reduced_inverted = None

            calibration_data = self._get_calibration_data(myqubits)
            ReadoutErrorMitigation._a_normalized = self._tensor_product_calibration(calibration_data)
            
            # normalize it
            for col in range(ReadoutErrorMitigation._a_normalized.shape[1]):
                col_sum= np.sum(ReadoutErrorMitigation._a_normalized[:, col])
                if col_sum > 1e-9:  # Threshold to handle potential near-zero sums
                    ReadoutErrorMitigation._a_normalized[:, col] /= col_sum


        observed_bit_string = list(self.all_results(results).keys())

        #Build the reduced A-matrix
        if ReadoutErrorMitigation._a_reduced is None:
            ReadoutErrorMitigation._a_reduced = self._get_reduced_a_matrix(ReadoutErrorMitigation._a_normalized, observed_bit_string, self.all_combinations)
            for col in range(ReadoutErrorMitigation._a_reduced.shape[1]):
                col_sum= np.sum(ReadoutErrorMitigation._a_reduced[:, col])
                if col_sum > 1e-9:  # Threshold to handle potential near-zero sums
                    ReadoutErrorMitigation._a_reduced[:, col] /= col_sum
            
        #Invert the reduced A-matrix
        if ReadoutErrorMitigation._a_reduced_inverted is None:
            try:
                ReadoutErrorMitigation._a_reduced_inverted = np.linalg.inv(ReadoutErrorMitigation._a_reduced)
            except np.linalg.LinAlgError:
                print("The reduced A-matrix is not invertible, using pseudo-inverse.")
                ReadoutErrorMitigation._a_reduced_inverted = np.linalg.pinv(ReadoutErrorMitigation._a_reduced)
        
        return ReadoutErrorMitigation._a_reduced_inverted

    def execute(self, tape : QuantumTape, results : dict[str, int]):
        # TODO : this method currently has edge effects. should probably make it purely functional by removing state from the class
        wires = [w for w in tape.wires]
            
        self.num_qubits = len(wires)
        real_counts = np.array([v for v in self.all_results(results).values()])
        
        A_reduced_inv = self._get_inverted_reduced_a_matrix(wires, results)
        
        # Correction
        corrected_counts = np.dot(A_reduced_inv, real_counts)
        corrected_counts = [np.round(v) for v in corrected_counts]
        # reconstruct counts dict
        return {k:v for k,v in zip(self.all_combinations, corrected_counts)}

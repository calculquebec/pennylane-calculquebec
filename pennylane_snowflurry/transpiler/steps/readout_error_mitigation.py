from pennylane.tape import QuantumTape
from pennylane.wires import Wires
from pennylane_snowflurry.utility.debug_utility import get_labels
from pennylane_snowflurry.API.api_adapter import ApiAdapter
import json
from datetime import datetime, timedelta
import numpy as np
from pennylane_snowflurry.transpiler.steps.interfaces.post_processing import PostProcStep

class ReadoutErrorMitigation(PostProcStep):
    @property
    def all_combinations(self):
        return get_labels((2 ** self.num_qubits) - 1)
    @property
    def all_results(self):
        all_combs = self.all_combinations
        return {bitstring:(self.results[bitstring] if bitstring in self.results else 0) for bitstring in all_combs}
        

    def _get_readout_fidelities(self, myqubits):
        benchmark = ApiAdapter.get_qubits_and_couplers()
        complete_benchmark = json.loads(ApiAdapter.get_benchmark().text)
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

    def _get_inverted_reduced_a_matrix(self, myqubits):
        # Generate the full A-matrix
        calibration_data = self._get_calibration_data(myqubits)
        A_full = self._tensor_product_calibration(calibration_data)

        # normalize it
        for col in range(A_full.shape[1]):
            col_sum= np.sum(A_full[:, col])
            if col_sum > 1e-9:  # Threshold to handle potential near-zero sums
                A_full[:, col] /= col_sum


        observed_bit_string = list(self.all_results.keys())

        #Build the reduced A-matrix
        A_reduced = self._get_reduced_a_matrix(A_full, observed_bit_string, self.all_combinations)
        for col in range(A_reduced.shape[1]):
            col_sum= np.sum(A_reduced[:, col])
            if col_sum > 1e-9:  # Threshold to handle potential near-zero sums
                A_reduced[:, col] /= col_sum
            
        #Invert the reduced A-matrix
        try:
            A_reduced_inv = np.linalg.inv(A_reduced)
        except np.linalg.LinAlgError:
            print("The reduced A-matrix is not invertible, using pseudo-inverse.")
            A_reduced_inv = np.lina
        
        return A_reduced_inv

    def execute(self, tape : QuantumTape, results : dict[str, int]):
        # TODO : this method currently has edge effects. should probably make it purely functional by removing state from the class
        wires = [w for w in tape.wires]
        self.results = results
        self.num_wires = len(wires)
            
        self.num_qubits = len(wires)
        real_counts = np.array([v for v in self.all_results.values()])
        
        A_reduced_inv = self._get_inverted_reduced_a_matrix(wires)
        
        # Correction
        corrected_counts = np.dot(A_reduced_inv, real_counts)
        corrected_counts = [np.round(v) for v in corrected_counts]
        # reconstruct counts dict
        return {k:v for k,v in zip(self.all_combinations, corrected_counts)}

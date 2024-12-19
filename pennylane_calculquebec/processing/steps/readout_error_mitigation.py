"""
Contains readout error mitigation post-processing steps
"""

from pennylane.tape import QuantumTape
from pennylane_calculquebec.utility.debug import get_labels
from pennylane_calculquebec.API.adapter import ApiAdapter
import json
import numpy as np
from pennylane_calculquebec.processing.interfaces import PostProcStep

class IBUReadoutMitigation(PostProcStep):
    """a mitigation method that uses iterative bayesian unfolding to mitigate readout errors on a circuit's results
    
    Args:
        * initial_guess (list[float]) : a probability distribution representing a starting point for the results (defaults to None)
    """
    def __init__(self, initial_guess = None):
        self._initial_guess = initial_guess
    
    
    def all_combinations(self, num_qubits):
        return get_labels((2 ** num_qubits) - 1)
  
    def all_results(self, results, num_qubits):
        """counts for all bitstring combinations

        Args:
            results (dict[str, int]): counts for possibilities that are not 0

        Returns:
            dict[str, int]: counts for all bitstring combinations
        """
        all_combs = self.all_combinations(num_qubits)
        return {bitstring:(results[bitstring] if bitstring in results else 0) for bitstring in all_combs}
        
    def get_readout_fidelities(self, chosen_qubits):
        """
        what are the readout 0 and 1 fidelities for given qubits?

        Args
            chosen_qubits (list[int]) : qubits from the circuit
        
        Returns
            a tuple appending readouts on 0 and 1 for given qubits
        """
        benchmark = ApiAdapter.get_qubits_and_couplers()
        
        readout0 = {}
        readout1 = {}
        for qubit in chosen_qubits:
            readout0[qubit] = benchmark["qubits"][str(qubit)]["readoutState0Fidelity"]
            readout1[qubit] = benchmark["qubits"][str(qubit)]["readoutState1Fidelity"]    

        readout0 = list(readout0.values())
        readout1 = list(readout1.values())
        return readout0, readout1
    
    def get_calibration_data(self, chosen_qubits):
        """gets fidelities for the observed qubits

        Args:
            chosen_qubits (list[int]) : which qubits are observed

        Returns:
            dict[int, float], dict[int, float] : readout fidelities for state 0 and 1
        """
        num_qubits = len(chosen_qubits)
        readout0, readout1 = self.get_readout_fidelities(chosen_qubits)
        calibration_data = {
            qubit: np.array([
                [readout0[qubit], 1 - readout1[qubit]],
                [1 - readout0[qubit], readout1[qubit]]
            ]) for qubit in range(num_qubits)
        }
        return calibration_data
    
    def tensor_product_calibration(self, calibration_matrices):
        """
        Initialize with the first qubit's calibration matrix
        """
        readout_matrix = calibration_matrices[0]
        
        # Tensor product of calibration matrices for all qubits
        for i in range(1, len(calibration_matrices)):
            readout_matrix = np.kron(readout_matrix, calibration_matrices[i])
            
        return readout_matrix
    
    def get_full_readout_matrix(self, chosen_qubits):
        calibration_data = self.get_calibration_data(chosen_qubits)
        full_readout_matrix = self.tensor_product_calibration(calibration_data)

        # normalize it
        for column in range(full_readout_matrix.shape[1]):
            column_sum= np.sum(full_readout_matrix[:, column])
            if column_sum > 1e-9:  # Threshold to handle potential near-zero sums
                full_readout_matrix[:, column] /= column_sum
        return full_readout_matrix

    def initial_guess(self, num_qubits):
        """
        returns a uniform probability vector if initial guess is not set. Returns initial guess otherwise
        """
        count_probabilities = 1 << num_qubits
        return [1/count_probabilities for _ in range(count_probabilities)]  \
            if self._initial_guess is None else self._initial_guess
    
    def iterative_bayesian_unfolding(self, 
                                     readout_matrix, 
                                     noisy_probs, 
                                     initial_guess, 
                                     max_iterations=1000, 
                                     tolerance=1e-6):
        """
        Iterative Bayesian unfolding to correct measurement errors.
        
        Args:
            readout_matrix (numpy.ndarray): Response matrix (2^n x 2^n).
            noisy_probs (numpy.ndarray): Noisy measured probability distribution.
            initial_guess (numpy.ndarray): Initial guess for the true distribution.
            max_iterations (int): Maximum number of iterations.
            tolerance (float): Convergence tolerance.
        
        Returns:
            final probabilities (numpy.ndarray): The final estimate of the true distribution.
            num_iterations (int): The number of iterations it took to converge.
        """
        current_probs = initial_guess.copy()
        
        for iteration in range(max_iterations):
            next_probs = np.zeros_like(current_probs)
            
            for true_prob in range(len(current_probs)):  # Loop over true states
                numerator = 0
                for measured_probs in range(len(noisy_probs)):  # Loop over measured states
                    mitigated_current_prob = np.dot(readout_matrix[measured_probs, :], current_probs)  # Compute sum_m R_im * theta_m
                    if mitigated_current_prob != 0:  # Avoid division by zero
                        numerator += noisy_probs[measured_probs] * readout_matrix[measured_probs, true_prob] * current_probs[true_prob] / mitigated_current_prob
                
                next_probs[true_prob] = numerator
            
            # Check for convergence
            if np.linalg.norm(next_probs - current_probs) < tolerance:
                return next_probs, iteration + 1
            
            current_probs = next_probs
        
        return current_probs, max_iterations
    
    
    def execute(self, tape, results):
        chosen_qubits = [wire for wire in tape.wires]
        num_qubits = len(chosen_qubits)
        shots = tape.shots.total_shots
        
        readout_matrix = self.get_full_readout_matrix(chosen_qubits)
        all_results = self.all_results(results, num_qubits)
        probs = [v/shots for _,v in all_results.items()]
        
        result, _ = self.iterative_bayesian_unfolding(readout_matrix, 
                                                   probs, 
                                                   self.initial_guess(num_qubits))
        result_dict = {key:round(shots * prob) for key, prob in zip(all_results.keys(), result)}
        return result_dict
    

class MatrixReadoutMitigation(PostProcStep):
    """
    a post-processing step that applies error mitigation based on the readout fidelities
    """
    _readout_matrix_normalized = None
    _readout_matrix_reduced = None
    _readout_matrix_reduced_inverted = None
    
    def all_combinations(self, num_qubits):
        return get_labels((2 ** num_qubits) - 1)
  
    def all_results(self, results, num_qubits):
        """counts for all bitstring combinations

        Args:
            results (dict[str, int]): counts for possibilities that are not 0

        Returns:
            dict[str, int]: counts for all bitstring combinations
        """
        all_combs = self.all_combinations(num_qubits)
        return {bitstring:(results[bitstring] if bitstring in results else 0) for bitstring in all_combs}
        
    def _get_readout_fidelities(self, chosen_qubits):
        """gets fidelities for the observed qubits

        Args:
            chosen_qubits (list[int]) : which qubits are observed

        Returns:
            dict[int, float], dict[int, float] : readout fidelities for state 0 and 1
        """
        benchmark = ApiAdapter.get_qubits_and_couplers()
        
        
        readout0 = {}
        readout1 = {}
        for qubit in chosen_qubits:
            readout0[qubit] = benchmark["qubits"][str(qubit)]["readoutState0Fidelity"]
            readout1[qubit] = benchmark["qubits"][str(qubit)]["readoutState1Fidelity"]    

        readout0 = list(readout0.values())
        readout1 = list(readout1.values())
        
        return readout0, readout1

    def _tensor_product_calibration(self, calibration_matrices):
        """
        creates a matrix out of calibration matrices for each observed qubits using kronecker product
        """
        # Initialize with the first qubit's calibration matrix
        readout_matrix = calibration_matrices[0]
        
        # Tensor product of calibration matrices for all qubits
        for qubit in range(1, len(calibration_matrices)):
            readout_matrix = np.kron(readout_matrix, calibration_matrices[qubit])
            
        return readout_matrix

    def _get_calibration_data(self, chosen_qubits):
        """create calibration matrices for each observed qubits

        Args:
            chosen_qubits (list[int]): observed qubits
        """
        state_0_readout_fidelity, state_1_readout_fidelity = self._get_readout_fidelities(chosen_qubits)
        calibration_data = {
            qubit: np.array([
                [state_0_readout_fidelity[qubit], 1 - state_1_readout_fidelity[qubit]],
                [1 - state_0_readout_fidelity[qubit], state_1_readout_fidelity[qubit]]
            ]) for qubit in range(len(chosen_qubits))
        }
        return calibration_data

    def _get_reduced_a_matrix(self, readout_matrix, observed_bit_strings, all_bit_strings):
        """
        keep only observe qubit lines and columns from A matrix
        """
        # Convert observed bit strings to their integer indices
        observed_indices = [all_bit_strings.index(bit_str) for bit_str in observed_bit_strings]
        
        # Extract the reduced A-matrix from the full A-matrix
        reduced_readout_matrix = readout_matrix[np.ix_(observed_indices, observed_indices)]
        
        return reduced_readout_matrix

    def _get_inverted_reduced_a_matrix(self, chosen_qubits, results):
        """
        create iverted reduced A matrix and cache it
        """
        num_qubits = len(chosen_qubits)
        # Generate the full A-matrix
        if MatrixReadoutMitigation._readout_matrix_normalized is None or ApiAdapter.is_last_update_expired():
            MatrixReadoutMitigation._readout_matrix_reduced = None
            MatrixReadoutMitigation._readout_matrix_reduced_inverted = None

            calibration_data = self._get_calibration_data(chosen_qubits)
            MatrixReadoutMitigation._readout_matrix_normalized = self._tensor_product_calibration(calibration_data)
            
            # normalize it
            for column in range(MatrixReadoutMitigation._readout_matrix_normalized.shape[1]):
                column_sum= np.sum(MatrixReadoutMitigation._readout_matrix_normalized[:, column])
                if column_sum > 1e-9:  # Threshold to handle potential near-zero sums
                    MatrixReadoutMitigation._readout_matrix_normalized[:, column] /= column_sum


        observed_bit_string = list(self.all_results(results, num_qubits).keys())

        #Build the reduced A-matrix
        if MatrixReadoutMitigation._readout_matrix_reduced is None:
            MatrixReadoutMitigation._readout_matrix_reduced = self._get_reduced_a_matrix(MatrixReadoutMitigation._readout_matrix_normalized, observed_bit_string, self.all_combinations(num_qubits))
            for column in range(MatrixReadoutMitigation._readout_matrix_reduced.shape[1]):
                column_sum= np.sum(MatrixReadoutMitigation._readout_matrix_reduced[:, column])
                if column_sum > 1e-9:  # Threshold to handle potential near-zero sums
                    MatrixReadoutMitigation._readout_matrix_reduced[:, column] /= column_sum
            
        #Invert the reduced A-matrix
        if MatrixReadoutMitigation._readout_matrix_reduced_inverted is None:
            try:
                MatrixReadoutMitigation._readout_matrix_reduced_inverted = np.linalg.inv(MatrixReadoutMitigation._readout_matrix_reduced)
            except np.linalg.LinAlgError:
                print("The reduced A-matrix is not invertible, using pseudo-inverse.")
                MatrixReadoutMitigation._readout_matrix_reduced_inverted = np.linalg.pinv(MatrixReadoutMitigation._readout_matrix_reduced)
        
        return MatrixReadoutMitigation._readout_matrix_reduced_inverted

    def execute(self, tape : QuantumTape, results : dict[str, int]):
        """
        mitigates readout errors from results using state 0 and 1 readouts
        """
        wires = [wire for wire in tape.wires]
        num_qubits = len(wires)
        real_counts = np.array([v for v in self.all_results(results, num_qubits).values()])
        
        inverted_reduced_readout_matrix = self._get_inverted_reduced_a_matrix(wires, results)
        
        # Correction
        corrected_counts = np.dot(inverted_reduced_readout_matrix, real_counts)
        corrected_counts = [np.round(count) for count in corrected_counts]
        # reconstruct counts dict
        return {key:count for key,count in zip(self.all_combinations(num_qubits), corrected_counts)}
        
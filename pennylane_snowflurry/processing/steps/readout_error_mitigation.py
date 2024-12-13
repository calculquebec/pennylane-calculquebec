"""
Contains readout error mitigation post-processing steps
"""

from pennylane.tape import QuantumTape
from pennylane_snowflurry.utility.debug import get_labels
from pennylane_snowflurry.API.adapter import ApiAdapter
import json
import numpy as np
from pennylane_snowflurry.processing.interfaces import PostProcStep

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
        
    def get_readout_fidelities(self, myqubits):
        benchmark = ApiAdapter.get_qubits_and_couplers()
        
        readout0 = {}
        readout1 = {}
        for i in myqubits:
            readout0[i] = benchmark["qubits"][str(i)]["readoutState0Fidelity"]
            readout1[i] = benchmark["qubits"][str(i)]["readoutState1Fidelity"]    

        readout0 = list(readout0.values())
        readout1 = list(readout1.values())
        return readout0, readout1
    
    def get_calibration_data(self, myqubits):
        """gets fidelities for the observed qubits

        Args:
            myqubits (list[int]) : which qubits are observed

        Returns:
            dict[int, float], dict[int, float] : readout fidelities for state 0 and 1
        """
        num_qubits = len(myqubits)
        readout0, readout1 = self.get_readout_fidelities(myqubits)
        calibration_data = {
            k: np.array([
                [readout0[k], 1 - readout1[k]],
                [1 - readout0[k], readout1[k]]
            ]) for k in range(num_qubits)
        }
        return calibration_data
    
    def tensor_product_calibration(self, calibration_matrices):
        """
        Initialize with the first qubit's calibration matrix
        """
        A = calibration_matrices[0]
        
        # Tensor product of calibration matrices for all qubits
        for i in range(1, len(calibration_matrices)):
            A = np.kron(A, calibration_matrices[i])
            
        return A
    
    def get_A_Full(self, myqubits):
        calibration_data = self.get_calibration_data(myqubits)
        A_full = self.tensor_product_calibration(calibration_data)

        # normalize it
        for col in range(A_full.shape[1]):
            col_sum= np.sum(A_full[:, col])
            if col_sum > 1e-9:  # Threshold to handle potential near-zero sums
                A_full[:, col] /= col_sum
        return A_full

    def initial_guess(self, num_qubits):
        return [1/(1 << num_qubits) for _ in range(1 << num_qubits)] if self._initial_guess is None else self._initial_guess
    
    def iterative_bayesian_unfolding(self, A_full, p_noisy, theta_0, max_iterations=1000, tol=1e-6):
        """
        Iterative Bayesian unfolding to correct measurement errors.
        
        Args:
            A_full (numpy.ndarray): Response matrix (2^n x 2^n).
            p_noisy (numpy.ndarray): Noisy measured probability distribution.
            theta_0 (numpy.ndarray): Initial guess for the true distribution.
            max_iterations (int): Maximum number of iterations.
            tol (float): Convergence tolerance.
        
        Returns:
            theta_final (numpy.ndarray): The final estimate of the true distribution.
            num_iterations (int): The number of iterations it took to converge.
        """
        theta_current = theta_0.copy()
        
        for iteration in range(max_iterations):
            theta_next = np.zeros_like(theta_current)
            
            for j in range(len(theta_current)):  # Loop over true states
                numerator = 0
                for i in range(len(p_noisy)):  # Loop over measured states
                    sum_m = np.dot(A_full[i, :], theta_current)  # Compute sum_m R_im * theta_m
                    if sum_m != 0:  # Avoid division by zero
                        numerator += p_noisy[i] * A_full[i, j] * theta_current[j] / sum_m
                
                theta_next[j] = numerator
            
            # Check for convergence
            if np.linalg.norm(theta_next - theta_current) < tol:
                print(f"Converged after {iteration + 1} iterations.")
                return theta_next, iteration + 1
            
            theta_current = theta_next
        
        print(f"Did not converge after {max_iterations} iterations.")
        return theta_current, max_iterations
    
    
    def execute(self, tape, results):
        myqubits = [w for w in tape.wires]
        num_qubits = len(myqubits)
        shots = tape.shots.total_shots
        
        a_full = self.get_A_Full(myqubits)
        all_results = self.all_results(results, num_qubits)
        probs = [v/shots for _,v in all_results.items()]
        
        result, _ = self.iterative_bayesian_unfolding(a_full, 
                                                   probs, 
                                                   self.initial_guess(num_qubits))
        result_dict = {k:round(shots * v) for k, v in zip(all_results.keys(), result)}
        return result_dict
    

class MatrixReadoutMitigation(PostProcStep):
    """
    a post-processing step that applies error mitigation based on the readout fidelities
    """
    _a_normalized = None
    _a_reduced = None
    _a_reduced_inverted = None
    
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
        
    def _get_readout_fidelities(self, myqubits):
        """gets fidelities for the observed qubits

        Args:
            myqubits (list[int]) : which qubits are observed

        Returns:
            dict[int, float], dict[int, float] : readout fidelities for state 0 and 1
        """
        benchmark = ApiAdapter.get_qubits_and_couplers()
        
        
        readout0 = {}
        readout1 = {}
        for i in myqubits:
            readout0[i] = benchmark["qubits"][str(i)]["readoutState0Fidelity"]
            readout1[i] = benchmark["qubits"][str(i)]["readoutState1Fidelity"]    

        readout0 = list(readout0.values())
        readout1 = list(readout1.values())
        
        return readout0, readout1

    def _tensor_product_calibration(self, calibration_matrices):
        """
        creates a matrix out of calibration matrices for each observed qubits using kronecker product
        """
        # Initialize with the first qubit's calibration matrix
        A = calibration_matrices[0]
        
        # Tensor product of calibration matrices for all qubits
        for i in range(1, len(calibration_matrices)):
            A = np.kron(A, calibration_matrices[i])
            
        return A

    def _get_calibration_data(self, myqubits):
        """create calibration matrices for each observed qubits

        Args:
            myqubits (list[int]): observed qubits
        """
        state_0_readout_fidelity, state_1_readout_fidelity = self._get_readout_fidelities(myqubits)
        calibration_data = {
            k: np.array([
                [state_0_readout_fidelity[k], 1 - state_1_readout_fidelity[k]],
                [1 - state_0_readout_fidelity[k], state_1_readout_fidelity[k]]
            ]) for k in range(len(myqubits))
        }
        return calibration_data

    def _get_reduced_a_matrix(self, A_full, observed_bit_strings, all_bit_strings):
        """
        keep only observe qubit lines and columns from A matrix
        """
        # Convert observed bit strings to their integer indices
        observed_indices = [all_bit_strings.index(bit_str) for bit_str in observed_bit_strings]
        
        # Extract the reduced A-matrix from the full A-matrix
        A_reduced = A_full[np.ix_(observed_indices, observed_indices)]
        
        return A_reduced

    def _get_inverted_reduced_a_matrix(self, myqubits, results):
        """
        create iverted reduced A matrix and cache it
        """
        num_qubits = len(myqubits)
        # Generate the full A-matrix
        if MatrixReadoutMitigation._a_normalized is None or ApiAdapter.is_last_update_expired():
            MatrixReadoutMitigation._a_reduced = None
            MatrixReadoutMitigation._a_reduced_inverted = None

            calibration_data = self._get_calibration_data(myqubits)
            MatrixReadoutMitigation._a_normalized = self._tensor_product_calibration(calibration_data)
            
            # normalize it
            for col in range(MatrixReadoutMitigation._a_normalized.shape[1]):
                col_sum= np.sum(MatrixReadoutMitigation._a_normalized[:, col])
                if col_sum > 1e-9:  # Threshold to handle potential near-zero sums
                    MatrixReadoutMitigation._a_normalized[:, col] /= col_sum


        observed_bit_string = list(self.all_results(results, num_qubits).keys())

        #Build the reduced A-matrix
        if MatrixReadoutMitigation._a_reduced is None:
            MatrixReadoutMitigation._a_reduced = self._get_reduced_a_matrix(MatrixReadoutMitigation._a_normalized, observed_bit_string, self.all_combinations(num_qubits))
            for col in range(MatrixReadoutMitigation._a_reduced.shape[1]):
                col_sum= np.sum(MatrixReadoutMitigation._a_reduced[:, col])
                if col_sum > 1e-9:  # Threshold to handle potential near-zero sums
                    MatrixReadoutMitigation._a_reduced[:, col] /= col_sum
            
        #Invert the reduced A-matrix
        if MatrixReadoutMitigation._a_reduced_inverted is None:
            try:
                MatrixReadoutMitigation._a_reduced_inverted = np.linalg.inv(MatrixReadoutMitigation._a_reduced)
            except np.linalg.LinAlgError:
                print("The reduced A-matrix is not invertible, using pseudo-inverse.")
                MatrixReadoutMitigation._a_reduced_inverted = np.linalg.pinv(MatrixReadoutMitigation._a_reduced)
        
        return MatrixReadoutMitigation._a_reduced_inverted

    def execute(self, tape : QuantumTape, results : dict[str, int]):
        """
        mitigates readout errors from results using state 0 and 1 readouts
        """
        wires = [w for w in tape.wires]
        num_qubits = len(wires)
        real_counts = np.array([v for v in self.all_results(results, num_qubits).values()])
        
        A_reduced_inv = self._get_inverted_reduced_a_matrix(wires, results)
        
        # Correction
        corrected_counts = np.dot(A_reduced_inv, real_counts)
        corrected_counts = [np.round(v) for v in corrected_counts]
        # reconstruct counts dict
        return {k:v for k,v in zip(self.all_combinations(num_qubits), corrected_counts)}
        
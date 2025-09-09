"""
Contains a post-processing step for adding noise to the results of a circuit using a noise model.
"""

from pennylane_calculquebec.processing.interfaces import PostProcStep
from pennylane_calculquebec.monarq_data import get_readout_noise_matrices
import pennylane_calculquebec.monarq_data as data
import pennylane as qml
import numpy as np
from pennylane_calculquebec.utility.debug import get_labels, get_measurement_wires
from pennylane_calculquebec.utility.noise import readout_error, TypicalBenchmark
from pennylane_calculquebec.logger import logger


class ReadoutNoiseSimulation(PostProcStep):
    """Adds readout noise on the results"""

    def __init__(self, machine_name: str, use_benchmark=True):
        self.machine_name = machine_name
        self.use_benchmark = use_benchmark

    def execute(self, tape, results):
        """adds readout noise to the results of a circuit

        Args:
            tape (QuantumTape): the tape where the results come from
            results (dict[str, int]) : the results from the circuit

        Returns:
            dict[str, int]: results with readout noise added to it
        """
        try:
            qubit_count = len(
                set(
                    [
                        a
                        for b in data.get_connectivity(
                            self.machine_name, False
                        ).values()
                        for a in b
                    ]
                )
            )
            coupler_count = len(data.get_connectivity(self.machine_name, False))

            results = results[0] if not isinstance(results, dict) else results

            readout_error_matrices = (
                get_readout_noise_matrices(self.machine_name)
                if self.use_benchmark
                else [
                    readout_error(TypicalBenchmark.readout0, TypicalBenchmark.readout1)
                    for _ in range(qubit_count)
                ]
            )

            readout_matrix = np.identity(1)

            wires = get_measurement_wires(tape)

            for wire in wires:
                readout_matrix = np.kron(readout_matrix, readout_error_matrices[wire])

            # Apply the readout error matrix (dot product with the probability vector)
            probs = []
            all_labels = get_labels((1 << len(wires)) - 1)

            for label in all_labels:
                probs.append(
                    results[label] / tape.shots.total_shots if label in results else 0
                )

            prob_after_error = np.dot(readout_matrix, probs)

            results_after_error = {
                label: np.round(prob_after_error[i] * tape.shots.total_shots)
                for i, label in enumerate(all_labels)
            }

            # Return the new measurement probabilities after applying the readout error
            return results_after_error
        except Exception as e:
            logger.error(
                "Error %s in execute located in ReadoutNoiseSimulation: %s",
                type(e).__name__,
                e,
            )
            return results

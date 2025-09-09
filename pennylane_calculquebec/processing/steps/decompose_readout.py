"""
contains a pre-processing step for decomposing readouts that are not observed from the computational basis\n
does not take into account multiple measurement yet.
"""

from pennylane_calculquebec.processing.interfaces import PreProcStep
from pennylane.tape import QuantumTape
import pennylane as qml
import pennylane.math as math
from pennylane_calculquebec.exceptions import ProcessingError


class DecomposeReadout(PreProcStep):

    def execute(self, tape: QuantumTape):
        """
        implementation of the execution method from pre-processing steps. \n
        for each observable, if it is a product, decompose it. \n
        if it is a single observable, add the right rotation before the readout,
        and change the observable to computational basis

        Args:
            tape (QuantumTape): the tape with the readouts to decompose

        Raises:
            ValueError: risen if an observable is not supported

        Returns:
            QuantumTape: a readout with only computational basis observables
        """
        operations = tape.operations.copy()
        measurements = []

        for measurement in tape.measurements:
            if measurement.obs is None:
                measurements.append(measurement)
                continue

            mat = qml.matrix(measurement.obs)

            if not math.allclose(mat.conj().T, mat):
                raise ProcessingError(
                    f"The observable {measurement.obs} is not supported"
                )

            operations += measurement.obs.diagonalizing_gates()
            measurements.append(type(measurement)(wires=measurement.wires))

        return type(tape)(operations, measurements, shots=tape.shots)

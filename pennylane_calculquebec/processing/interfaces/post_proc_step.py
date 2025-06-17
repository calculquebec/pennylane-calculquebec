"""
Contains a base class that can be implemented for creating new post-processing steps
"""

from pennylane_calculquebec.processing.interfaces.base_step import BaseStep
from pennylane.tape import QuantumTape


class PostProcStep(BaseStep):
    """a base class that represents post-processing steps that apply on quantum circuits' results"""

    def execute(self, tape: QuantumTape, results: dict[str, int]) -> dict[str, int]:
        """
        applies processing on a quantum circuit's results after the execution

        Args:
            tape (QuantumTape) : the tape representation of the quantum circuit
            results (dict[str, int]) : the results of the execution represented as counts

        Returns:
            dict[str, int] : the processed results represented as counts
        """

        return results

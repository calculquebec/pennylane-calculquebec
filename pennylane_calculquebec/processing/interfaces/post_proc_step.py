"""
Contains a base class that can be implemented for creating new post-processing steps
"""

from pennylane_calculquebec.processing.interfaces.base_step import BaseStep
from pennylane.tape import QuantumTape

class PostProcStep(BaseStep):
    """a base class that represents post-processing steps that apply on quantum circuits' results
    """
    
    def execute(self, tape : QuantumTape, results : dict[str, int]):
        return results
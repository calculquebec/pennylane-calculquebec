from pennylane.tape import QuantumTape
from pennylane_snowflurry.processing.interfaces.base_step import BaseStep

class PreProcStep(BaseStep):
    """a base class that represents pre-processing steps that apply on quantum circuit operations
    """
    def execute(self, tape : QuantumTape):
        return tape
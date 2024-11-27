from pennylane.tape import QuantumTape
from pennylane_snowflurry.processing.interfaces.base_step import BaseStep

class PreProcStep(BaseStep):
    def execute(self, tape : QuantumTape):
        return tape
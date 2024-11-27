from pennylane_snowflurry.processing.interfaces.base_step import BaseStep
from pennylane.tape import QuantumTape

class PostProcStep(BaseStep):
    results : dict[str, int]
    
    def execute(self, tape : QuantumTape, results : dict[str, int]):
        return results
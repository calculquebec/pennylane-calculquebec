from pennylane_snowflurry.processing.interfaces.base_step import BaseStep
from pennylane.tape import QuantumTape

class PostProcStep(BaseStep):
    """a base class that represents post-processing steps that apply on quantum circuits' results
    """
    results : dict[str, int]
    
    def execute(self, tape : QuantumTape, results : dict[str, int]):
        return results
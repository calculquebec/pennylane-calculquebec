from pennylane.tape import QuantumTape

class BaseStep:
    """
    a step that can be inherited in order to be appended to a TranspilerConfig. 
    Adding a step to a TranspilerConfig means it will be applied in the preprocess step of MonarqDevice.
    """
    def execute(self, tape : QuantumTape) -> QuantumTape:
        return tape
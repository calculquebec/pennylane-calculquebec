from pennylane.tape import QuantumTape

class BaseStep:
    def execute(self, tape : QuantumTape) -> QuantumTape:
        return tape
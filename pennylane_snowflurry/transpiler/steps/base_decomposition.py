from pennylane.tape import QuantumTape
from pennylane.operation import Operation
from pennylane_snowflurry.transpiler.steps.base_step import BaseStep
import pennylane.transforms as transforms


class BaseDecomposition(BaseStep):
    @property
    def base_gates(self):
        return []
    
    def execute(self, tape : QuantumTape) -> QuantumTape:
        def stop_at(op : Operation):
            # TODO : voir quelles portes on veut stop at
            return op.name in self.base_gates

        # pennylane create_expand_fn does the job for us 
        custom_expand_fn = transforms.create_expand_fn(depth=9, stop_at=stop_at)
        tape = custom_expand_fn(tape)
        return tape

class CliffordTDecomposition(BaseDecomposition):
    @property
    def base_gates(self):
        return ["Adjoint(T)", "Adjoint(S)", "SX", "Adjoint(SX)", 
                  "T", "PauliX", "PauliY", "PauliZ", "S", "Hadamard", 
                  "CZ", "CNOT", "RZ", "RX", "RY"]
        
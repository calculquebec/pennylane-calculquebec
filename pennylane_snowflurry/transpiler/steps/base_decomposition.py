from pennylane.tape import QuantumTape
from pennylane.operation import Operation
from pennylane_snowflurry.transpiler.steps.interfaces.pre_processing import PreProcStep
import pennylane.transforms as transforms


class BaseDecomposition(PreProcStep):
    """The purpose of this transpiler step is to turn the gates in a circuit to a simpler, more easily usable set of gates
    """
    @property
    def base_gates(self):
        """the base set of gates the circuit should be turned into
        """
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
        
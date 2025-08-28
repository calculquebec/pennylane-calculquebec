"""
Contains native decomposition pre-processing steps
"""

from pennylane.tape import QuantumTape
import pennylane as qml
import pennylane_calculquebec.processing.decompositions.native_decomp_functions as decomp_funcs
import numpy as np
from pennylane.ops.op_math import SProd
from pennylane_calculquebec.processing.interfaces import PreProcStep
from pennylane_calculquebec.monarq_data import monarq_native_gates
from pennylane_calculquebec.logger import logger


class NativeDecomposition(PreProcStep):
    """
    the purpose of this transpiler step is to turn the gates in the circuit into a set of gate that's readable by a specific machine
    """

    def native_gates(self):
        return []


class MonarqDecomposition(NativeDecomposition):
    """a decomposition process for turing all operations in a quantum tape to MonarQ-native ones

    Raises:
        ValueError: will be raised if an operation is not supported
    """

    _decomp_map = {
        "Adjoint(T)": decomp_funcs._custom_tdag,
        "S": decomp_funcs._custom_s,
        "Adjoint(S)": decomp_funcs._custom_sdag,
        "SX": decomp_funcs._custom_sx,
        "Adjoint(SX)": decomp_funcs._custom_sxdag,
        "Hadamard": decomp_funcs._custom_h,
        "CNOT": decomp_funcs._custom_cnot,
        "CY": decomp_funcs._custom_cy,
        "RZ": decomp_funcs._custom_rz,
        "RX": decomp_funcs._custom_rx,
        "RY": decomp_funcs._custom_ry,
        "SWAP": decomp_funcs._custom_swap,
    }

    def native_gates(self):
        """
        the set of monarq-native gates

        Returns:
            list[str] : the name of the gates that MonarQ can execute
        """
        return monarq_native_gates()

    def execute(self, tape: QuantumTape) -> QuantumTape:
        """Turns all gates in a tape to native gates

        Args:
            tape (QuantumTape): the tape to act on

        Raises:
            ValueError: Raised if the gate is not decomposable

        Returns:
            QuantumTape: The processed quantum tape
        """
        new_operations = []

        with qml.QueuingManager.stop_recording():
            for operation in tape.operations:
                if operation.name in MonarqDecomposition._decomp_map:
                    if operation.num_params > 0:
                        new_operations.extend(
                            MonarqDecomposition._decomp_map[operation.name](
                                angle=operation.data[0], wires=operation.wires
                            )
                        )
                    else:
                        new_operations.extend(
                            MonarqDecomposition._decomp_map[operation.name](
                                wires=operation.wires
                            )
                        )
                else:
                    if operation.name in self.native_gates():
                        new_operations.append(operation)
                    else:
                        raise ValueError(
                            f"gate {operation.name} is not handled by the native decomposition step. Did you bypass the base decomposition step?"
                        )

        new_operations = [
            operation.data[0][0] if isinstance(operation, SProd) else operation
            for operation in new_operations
        ]
        new_tape = type(tape)(new_operations, tape.measurements, shots=tape.shots)

        return new_tape

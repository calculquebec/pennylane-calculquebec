"""
Contains native decomposition pre-processing steps
"""

from pennylane.tape import QuantumTape
import pennylane as qml
import pennylane_snowflurry.processing.decompositions.native_decomp_functions as decomp_funcs
import numpy as np
from pennylane.ops.op_math import SProd
from pennylane_snowflurry.processing.interfaces import PreProcStep

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
        "Adjoint(T)" : decomp_funcs._custom_tdag,
        "S" : decomp_funcs._custom_s,
        "Adjoint(S)" : decomp_funcs._custom_sdag,
        "SX" : decomp_funcs._custom_sx,
        "Adjoint(SX)" : decomp_funcs._custom_sxdag,
        "Hadamard" : decomp_funcs._custom_h,
        "CNOT" : decomp_funcs._custom_cnot,
        "CY" : decomp_funcs._custom_cy,
        "RZ" : decomp_funcs._custom_rz,
        "RX" : decomp_funcs._custom_rx,
        "RY" : decomp_funcs._custom_ry,
        "SWAP" : decomp_funcs._custom_swap
    }
    
    def native_gates(self):
        """the set of monarq-native gates"""
        return  [
            "T", "TDagger",
            "PauliX", "PauliY", "PauliZ", 
            "X90", "Y90", "Z90",
            "XM90", "YM90", "ZM90",
            "PhaseShift", "CZ", "RZ"
        ]
    
    def execute(self, tape : QuantumTape):
        """
        decomposes all non-native gate to an equivalent set of native gates
        """
        new_operations = []

        with qml.QueuingManager.stop_recording():
            for op in tape.operations:
                if op.name in MonarqDecomposition._decomp_map:
                    if op.num_params > 0:
                        new_operations.extend(MonarqDecomposition._decomp_map[op.name](angle=op.data[0], wires=op.wires))
                    else:
                        new_operations.extend(MonarqDecomposition._decomp_map[op.name](wires=op.wires))
                else:
                    if op.name in self.native_gates():
                        new_operations.append(op)
                    else:
                        raise ValueError(f"gate {op.name} is not handled by the native decomposition step. Did you bypass the base decomposition step?")

        new_operations = [n.data[0][0] if isinstance(n, SProd) else n for n in new_operations]
        new_tape = type(tape)(new_operations, tape.measurements, shots=tape.shots)

        return new_tape
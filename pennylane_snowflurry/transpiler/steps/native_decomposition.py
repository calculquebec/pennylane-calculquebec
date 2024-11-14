from pennylane.tape import QuantumTape
import pennylane as qml
import pennylane_snowflurry.custom_gates as custom
import numpy as np
from pennylane.ops.op_math import SProd
from pennylane_snowflurry.transpiler.steps.base_step import BaseStep

class NativeDecomposition(BaseStep):
    pass

class MonarqDecomposition(NativeDecomposition):
    def _custom_tdag(wires):
        """
        a native implementation of the adjoint(T) operation
        """
        return [custom.TDagger(wires)]

    def _custom_s(wires):
        """
        a native implementation of the S operation
        """
        return [custom.Z90(wires)]

    def _custom_sdag(wires):
        """
        a native implementation of the adjoint(S) operation
        """
        return [custom.ZM90(wires)]

    def _custom_h(wires):
        """
        a native implementation of the Hadamard operation
        """
        return [custom.Z90(wires), custom.X90(wires), custom.Z90(wires)]

    def _custom_cnot(wires):
        """
        a native implementation of the CNOT operation
        """
        return MonarqDecomposition._custom_h(wires[1]) + [qml.CZ(wires)] + MonarqDecomposition._custom_h(wires[1])

    def _custom_cy(wires):
        """
        a native implementation of the CY operation
        """
        return MonarqDecomposition._custom_h(wires[1]) \
            + MonarqDecomposition._custom_s(wires[1]) \
            + MonarqDecomposition._custom_cnot(wires) \
            + MonarqDecomposition._custom_sdag(wires[1]) \
            + MonarqDecomposition._custom_h(wires[1])

    def _custom_rz(angle : float, wires, epsilon = 1E-8):
        """
        a native implementation of the RZ operation
        """
        while angle < 0: angle += np.pi * 2
        angle %= np.pi * 2
        is_close_enough_to = lambda other_angle: np.abs(angle - other_angle) < epsilon

        if is_close_enough_to(0): return []
        elif is_close_enough_to(np.pi/4): return [qml.T(wires = wires)]
        elif is_close_enough_to(-np.pi/4): return [custom.TDagger(wires = wires)]
        elif is_close_enough_to(np.pi/2): return [custom.Z90(wires = wires)]
        elif is_close_enough_to(-np.pi/2): return [custom.ZM90(wires = wires)]
        elif is_close_enough_to(np.pi): return [qml.PauliZ(wires = wires)]
        else: return [qml.PhaseShift(angle, wires)]

    def _custom_rx(angle : float, wires, epsilon = 1E-8):
        """
        a native implementation of the RX operation
        """
        while angle < 0: angle += np.pi * 2
        angle %= np.pi * 2
        is_close_enough_to = lambda other_angle: np.abs(angle - other_angle) < epsilon

        if is_close_enough_to(0): return []
        elif is_close_enough_to(np.pi/2): return [custom.X90(wires = wires)]
        elif is_close_enough_to(-np.pi/2): return [custom.XM90(wires = wires)]
        elif is_close_enough_to(np.pi): return [qml.PauliX(wires = wires)]
        else: return MonarqDecomposition._custom_h(wires) + [qml.PhaseShift(angle, wires)] + MonarqDecomposition._custom_h(wires)

    def _custom_ry(angle : float, wires, epsilon = 1E-8):
        """
        a native implementation of the RY operation
        """
        while angle < 0: angle += np.pi * 2
        angle %= np.pi * 2
        is_close_enough_to = lambda other_angle: np.abs(angle - other_angle) < epsilon

        if is_close_enough_to(0): return []
        elif is_close_enough_to(np.pi/2): return [custom.Y90(wires = wires)]
        elif is_close_enough_to(-np.pi/2): return [custom.YM90(wires = wires)]
        elif is_close_enough_to(np.pi): return [qml.PauliY(wires = wires)]
        else: return MonarqDecomposition._custom_s(wires) + MonarqDecomposition._custom_h(wires) \
                    + [qml.PhaseShift(angle, wires = wires)] + MonarqDecomposition._custom_h(wires) + MonarqDecomposition._custom_s(wires)

    def _custom_swap(wires):
        """
        a native implementation of the SWAP operation
        """
        return MonarqDecomposition._custom_cnot(wires) + MonarqDecomposition._custom_h(wires[0]) + MonarqDecomposition._custom_h(wires[1]) \
             + MonarqDecomposition._custom_cnot(wires) + MonarqDecomposition._custom_h(wires[0]) + MonarqDecomposition._custom_h(wires[1]) \
             + MonarqDecomposition._custom_cnot(wires)

    _decomp_map = {
        "Adjoint(T)" : _custom_tdag,
        "S" : _custom_s,
        "Adjoint(S)" : _custom_sdag,
        "Hadamard" : _custom_h,
        "CNOT" : _custom_cnot,
        "CY" : _custom_cy,
        "RZ" : _custom_rz,
        "RX" : _custom_rx,
        "RY" : _custom_ry,
        "SWAP" : _custom_swap
    }
    
    _native_gates = [
        "T", 
        "TDagger",
        "PauliX",
        "PauliY",
        "PauliZ", 
        "X90",
        "Y90",
        "Z90",
        "XM90",
        "YM90",
        "ZM90",
        "PhaseShift",
        "CZ"
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
                    if op.name in MonarqDecomposition._native_gates:
                        new_operations.append(op)
                    else:
                        raise Exception(f"gate {op.name} is not handled by the native decomposition step. Did you bypass the base decomposition step?")

        new_operations = [n.data[0][0] if isinstance(n, SProd) else n for n in new_operations]
        new_tape = type(tape)(new_operations, tape.measurements, shots=tape.shots)

        return new_tape
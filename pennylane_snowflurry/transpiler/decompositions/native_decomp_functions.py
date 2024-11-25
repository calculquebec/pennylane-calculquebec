import pennylane_snowflurry.custom_gates as custom
import numpy as np
import pennylane as qml

def _custom_tdag(wires):
    """
    a MonarQ native implementation of the adjoint(T) operation
    """
    return [custom.TDagger(wires)]

def _custom_sx(wires):
    """
    a MonarQ native implementation of the SX operation
    """
    return [custom.X90(wires)]

def _custom_sxdag(wires):
    """
    a MonarQ native implementation of the adjoint(SX) operation 
    """
    return [custom.XM90(wires)]

def _custom_s(wires):
    """
    a MonarQ native implementation of the S operation
    """
    return [custom.Z90(wires)]

def _custom_sdag(wires):
    """
    a MonarQ native implementation of the adjoint(S) operation
    """
    return [custom.ZM90(wires)]

def _custom_h(wires):
    """
    a MonarQ native implementation of the Hadamard operation
    """
    return [custom.Z90(wires), custom.X90(wires), custom.Z90(wires)]

def _custom_cnot(wires):
    """
    a MonarQ native implementation of the CNOT operation
    """
    return _custom_h(wires[1]) + [qml.CZ(wires)] + _custom_h(wires[1])

def _custom_cy(wires):
    """
    a MonarQ native implementation of the CY operation
    """
    return _custom_h(wires[1]) \
        + _custom_s(wires[1]) \
        + _custom_cnot(wires) \
        + _custom_sdag(wires[1]) \
        + _custom_h(wires[1])

def _custom_rz(angle : float, wires, epsilon = 1E-8):
    """
    a MonarQ native implementation of the RZ operation
    """
    while angle < 0: angle += np.pi * 2
    angle %= np.pi * 2
    is_close_enough_to = lambda other_angle: np.abs(angle - other_angle) < epsilon

    if is_close_enough_to(0): return []
    elif is_close_enough_to(np.pi/4): return [qml.T(wires = wires)]
    elif is_close_enough_to(7 * np.pi/4): return [custom.TDagger(wires = wires)]
    elif is_close_enough_to(np.pi/2): return [custom.Z90(wires = wires)]
    elif is_close_enough_to(3 * np.pi/2): return [custom.ZM90(wires = wires)]
    elif is_close_enough_to(np.pi): return [qml.PauliZ(wires = wires)]
    else: return [qml.PhaseShift(angle, wires)]

def _custom_rx(angle : float, wires, epsilon = 1E-8):
    """
    a MonarQ native implementation of the RX operation
    """
    while angle < 0: angle += np.pi * 2
    angle %= np.pi * 2
    is_close_enough_to = lambda other_angle: np.abs(angle - other_angle) < epsilon

    if is_close_enough_to(0): 
        return []
    elif is_close_enough_to(np.pi/2): 
        return [custom.X90(wires = wires)]
    elif is_close_enough_to(3 * np.pi/2): 
        return [custom.XM90(wires = wires)]
    elif is_close_enough_to(np.pi): 
        return [qml.PauliX(wires = wires)]
    else: 
        return _custom_h(wires) + [qml.PhaseShift(angle, wires)] + _custom_h(wires)

def _custom_ry(angle : float, wires, epsilon = 1E-8):
    """
    a MonarQ native implementation of the RY operation
    """
    while angle < 0: angle += np.pi * 2
    angle %= np.pi * 2
    is_close_enough_to = lambda other_angle: np.abs(angle - other_angle) < epsilon

    if is_close_enough_to(0): return []
    elif is_close_enough_to(np.pi/2): return [custom.Y90(wires = wires)]
    elif is_close_enough_to(3 * np.pi/2): return [custom.YM90(wires = wires)]
    elif is_close_enough_to(np.pi): return [qml.PauliY(wires = wires)]
    else: return _custom_s(wires) + _custom_h(wires) \
                + [qml.PhaseShift(angle, wires = wires)] + _custom_h(wires) + _custom_s(wires)

def _custom_swap(wires):
    """
    a MonarQ native implementation of the SWAP operation
    """
    return _custom_cnot([wires[0], wires[1]]) + _custom_cnot([wires[1], wires[0]]) + _custom_cnot([wires[0], wires[1]])

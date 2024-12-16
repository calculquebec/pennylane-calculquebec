"""
Contains debug utility functions
"""

from functools import partial
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operation
from pennylane.tape import QuantumTape
import pennylane as qml
import numpy as np
import pennylane_snowflurry.processing.custom_gates as custom
import random

def remove_global_phase(matrix):
    """
    Removes the global phase from a matrix by normalizing its determinant to 1.

    Parameters:
        matrix (ndarray): The input matrix (square and complex).

    Returns:
        ndarray: The matrix with the global phase removed.
    """
    # Compute the determinant of the matrix
    determinant = np.linalg.det(matrix)

    # Compute the global phase factor
    global_phase = np.exp(1j * np.angle(determinant))

    # Normalize the matrix to remove the global phase
    normalized_matrix = matrix / global_phase

    return normalized_matrix

# TOFIX : this method doesnt work for some matrices
def are_matrices_equivalent(matrix1, matrix2, tolerance=1e-9):
    """
    Checks if two matrices are equal up to a complex multiplicative factor.

    Args:
        matrix1 (ndarray): First matrix.
        matrix2 (ndarray): Second matrix.
        tolerance (float): Numerical tolerance for comparison.

    Returns:
        bool: True if the matrices are equal up to a complex factor, False otherwise.
    """
    det = np.linalg.det(matrix2)
    
    if abs(det) < tolerance:
        raise ValueError("cannot invert matrix. cannot check equivalence")
    
    if matrix1.shape != matrix2.shape:
        return False

    matrix1_normalized = remove_global_phase(matrix1)
    matrix2_normalized = remove_global_phase(matrix2)

    matrix2_inv = np.linalg.inv(matrix2_normalized)
    id = np.round(matrix1_normalized @ matrix2_inv, 4)
    value = id[0][0]
    
    # check if matrix is diagonal
    for i in range(id.shape[0]):
        for j in range(id.shape[1]):
            # if non-diagonal position is > 0, not equivalent
            if i != j and abs(id[i][j]) > tolerance:
                return False
            # if diagonal positions are not equal, not equivalent
            if i == j and abs(id[i][j] - value) > tolerance:
                return False
    return True


def add_noise(tape : QuantumTape, t = 0.005):
    """
    adds noise for each gate present in a quantum tape

    Args:
        tape (QuantumTape): the tape you want to noisify
        t (float, optional): the noise quantity from 0 to 1. Defaults to 0.005.
    """
    new_operations = []
    for op in tape.operations:
        new_operations += [op] + [qml.DepolarizingChannel(random.random() * t, w) for w in op.wires]
    
    new_tape = type(tape)(ops=new_operations, measurements=tape.measurements, shots=tape.shots)
    return [new_tape], lambda results : results[0]

def to_qasm(tape : QuantumTape) -> str:
    """
    turns a quantum tape into a qasm string
    """
    eq = {
        "PauliX" : "x", "PauliY" : "y", "PauliZ" : "z", "Identity" : "id",
        "RX" : "rx", "RY" : "ry", "RZ" : "rz", "PhaseShift" : "p", "Hadamard" : "h",
        "S" : "s", "Adjoint(S)" : "sdg", "SX" : "sx", "Adjoint(SX)" : "sxdg", "T" : "t", "Adjoint(T)" : "tdg", 
        "CNOT" : "cx", "CY" : "cy", "CZ" : "cz", "SWAP" : "swap",
        "Z90" : "s", "ZM90" : "sdg", "X90" : "sx", "XM90" : "sxdg", "Y90" : "ry(pi/2)", "YM90" : "ry(3*pi/2)",
        "TDagger" : "tdg", "CRY" : "cry"
    }
    total_string = ""
    for op in tape.operations:
        string = eq[op.name]
        if len(op.parameters) > 0:            
            string += "(" + str(op.parameters[0]) + ")" 
        string += " "
        string += ", ".join([f"q[{w}]" for w in op.wires])
        string += ";"
        total_string += string + "\n"
    return total_string

def arbitrary_circuit(tape : QuantumTape, measurement = qml.counts):
    """
    create a quantum function out of a tape and a default measurement to use (overrides the measurements in the qtape)
    """
    def _arbitrary_circuit(operations : list[Operation], measurements : list[MeasurementProcess]):
        for op in operations:
            if len(op.parameters) > 0:
                qml.apply(op)
            else:
                qml.apply(op)
        
        def get_wires(mp : MeasurementProcess):
            return [w for w in mp.wires] if mp is not None and mp.wires is not None and len(mp.wires) > 0 else tape.wires

        # retourner une liste de mesures si on a plusieurs mesures, sinon retourner une seule mesure
        return [measurement(wires=get_wires(meas)) for meas in measurements] if len(measurements) > 1 \
            else measurement(wires=get_wires(measurements[0] if len(measurements) > 0 else None))
    return _arbitrary_circuit(tape.operations, tape.measurements)

def get_labels(up_to : int):
    """
    gets bitstrings from 0 to "up_to" value
    """
    
    if not isinstance(up_to, int): 
        raise ValueError("up_to must be an int")
    if up_to < 0:
        raise ValueError("up_to must be >= 0")
    
    num = int(np.log2(up_to)) + 1
    return [format(i, f"0{num}b") for i in range(up_to + 1)]

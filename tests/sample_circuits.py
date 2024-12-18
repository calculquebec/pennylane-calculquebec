"""
Contains a couple of quantum circuit implementations\n
Those circuits are not used in any features of the plugin.
"""

import pennylane as qml
import numpy as np

def add_k_fourier(k, wires):
    """applies phaseshifts on each wires for phase embedding k
    """
    for j in range(len(wires)):
        qml.PhaseShift(k * np.pi / (2 ** j), wires=wires[j])

def U(wires, angle = 2 * np.pi / 5):
    """arbitrary qubit unitary given an angle using a phaseshift
    """
    return qml.PhaseShift(angle, wires=wires)

def sum_m_k(m, k, num_wires):
    """add two numbers using n qubits
    """
    wires = range(num_wires)
    qml.BasisEmbedding(m, wires=wires)
    qml.QFT(wires=wires)
    add_k_fourier(k, wires)
    qml.adjoint(qml.QFT)(wires=wires)
    return qml.counts(wires=wires)

def grover4():
    """
    an implementation of grover that should find 0110 and 1001
    """
    num_qubits = 4
    for i in range(num_qubits):
        qml.Hadamard(i)
        
    qml.CNOT([0, 1])
    qml.CNOT([2, 3])
    qml.CNOT([0, 2])
    qml.CCZ([1,2,3])
    qml.CNOT([0, 2])
    qml.CNOT([0, 1])
    qml.CNOT([2, 3])
        
    for i in range(num_qubits):
        qml.Hadamard(i)
        qml.PauliX(i)
    qml.ControlledQubitUnitary(qml.Z(0), [1, 2, 3])
    for i in range(num_qubits):
        qml.PauliX(i)
        qml.Hadamard(i)
    return qml.counts(wires = [0, 1, 2, 3])


def circuit_qpe(num_wires = 5, angle = 2 * np.pi / 5, measurement = qml.counts):
    """quantum phase estimation algorithm using given angle and number of qubits
    """
    wires = [i for i in range(num_wires)]
    estimation_wires = wires[:-1]
    # initialize to state |1>
    qml.PauliX(wires=num_wires - 1)

    for wire in estimation_wires:
        qml.Hadamard(wires=wire)    

    qml.ControlledSequence(U(num_wires - 1, angle), control=estimation_wires)
    qml.adjoint(qml.QFT)(wires=estimation_wires)

    return measurement(wires=estimation_wires)

def GHZ(num_wires, measurement=qml.counts):
    """ghz on given number of qubits
    """
    qml.Hadamard(0)
    [qml.CNOT([0, i]) for i in range(1, num_wires)]
    return measurement(wires = range(num_wires))

def Toffoli():
    """toffoli with hadamards on control wires
    """
    qml.Hadamard(0)
    qml.Hadamard(1)
    [qml.Toffoli([0, 1, 2])]
    return qml.probs(wires = [0, 1, 2])

def bernstein_vazirani(number : int, num_qubits : int, measurement = qml.counts):
    """
    a general implementation of bernstein vazirani

    Args:
        number (int): a number which will be translated to a bit string to find
        num_qubits (int): the amount of qubits to use
        measurement (optional): the measurement method that should be used. Defaults to qml.counts.
    """
    value = []
    for i in range(num_qubits - 1):
        value.insert(0, (number & (1 << i)) != 0)

    qml.X(num_qubits-1)
        
    [qml.Hadamard(i) for i in range(num_qubits)]
    
    # Uf
    [qml.CNOT([i, num_qubits - 1]) for i, should in enumerate(value) if should]
        
    [qml.Hadamard(i) for i in range(num_qubits - 1)]
    
    wires = [i for i in range(num_qubits - 1)]
    return measurement(wires=wires)
    
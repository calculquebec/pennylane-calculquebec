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

def circuit_qpe(num_wires = 5, angle = 2 * np.pi / 5):
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

    return qml.counts(wires=estimation_wires)

def GHZ(num_wires):
    """ghz on given number of qubits
    """
    qml.Hadamard(0)
    [qml.CNOT([0, i]) for i in range(1, num_wires)]
    return qml.counts(wires = range(num_wires))

def Toffoli():
    """toffoli with hadamards on control wires
    """
    qml.Hadamard(0)
    qml.Hadamard(1)
    [qml.Toffoli([0, 1, 2])]
    return qml.probs(wires = [0, 1, 2])

def bernstein_vazirani(number : int):
    """bernstein vazirani for encoding given number
    """
    value = []
    while number > 0:
        value.insert(0, (number & 1) != 0)
        number = number >> 1

    num_wires = len(value) + 1
    [qml.Hadamard(i) for i in range(num_wires)]
    qml.Z(num_wires-1)
        
    # Uf
    [qml.CNOT([i, num_wires - 1]) for i, should in enumerate(value) if should]
        
    [qml.Hadamard(i) for i in range(num_wires - 1)]
    return qml.counts(wires=[i for i in range(num_wires - 1)])
    
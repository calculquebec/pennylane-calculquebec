"""
Contains debug utility functions
"""

from functools import partial
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operation
from pennylane.tape import QuantumTape
import pennylane as qml
import numpy as np
import pennylane_calculquebec.processing.custom_gates as custom
import random


def compute_expval(probabilities: list[float]) -> float:
    """Compute the expectation value using the parity of each outcome

    Args:
        probabilities (list[float]): the results of a circuit execution represented as probabilities

    Returns:
        float: the expectation value of a probability distribution
    """

    if isinstance(probabilities, dict):
        probabilities = counts_to_probs(probabilities)

    expval = 0
    for i, prob in enumerate(probabilities):
        hamming_weight = bin(i).count("1")  # Count the 1s in the binary representation
        parity = (-1) ** hamming_weight  # +1 for even parity, -1 for odd parity
        expval += prob * parity
    return expval


def probs_to_counts(probs: list, count: int) -> dict[str, int]:
    """turns probabilities into counts

    Args:
        probs (list): the probability distribution
        count (int): the amount of shots

    Returns:
        dict[str, int]: the counts
    """
    bit_length = np.log2(len(probs))
    return {label_from(i, bit_length): round(p * count) for i, p in enumerate(probs)}


def counts_to_probs(counts: dict) -> list[float]:
    """converts counts into probabilities

    Args:
        counts (dict): the results of a circuit execution as counts

    Returns:
        list[float]: probabilities for a circuit
    """
    max_count = sum(counts.values())
    all_labels = get_labels(2 ** len(list(counts.keys())[0]) - 1)
    return [
        (counts[label] if label in counts else 0) / max_count for label in all_labels
    ]


def are_tape_same_probs(tape1, tape2):
    """execute two tapes and check if they yield the same probabilities

    Args:
        tape1 (QuantumTape): a quantum tape with counts in it
        tape2 (QuantumTape): a quantum tape with counts in it

    Returns:
        bool: do the two tapes have the same probabilities?
    """
    tolerance_place = 5

    dev = qml.device("default.qubit")
    results1 = qml.execute([tape1], dev)[0]
    results2 = qml.execute([tape2], dev)[0]

    if isinstance(results1, dict):
        results1 = counts_to_probs(results1)

    if isinstance(results2, dict):
        results2 = counts_to_probs(results2)

    results1 = np.round(results1, tolerance_place)
    results2 = np.round(results2, tolerance_place)

    return np.array_equal(results1, results2)


def to_qasm(tape: QuantumTape) -> str:
    """turns a quantum tape into a qasm string

    Args:
        tape (QuantumTape): a quantum tape you want to turn into qasm2

    Returns:
        str: the resulting qasm2 string
    """
    eq = {
        "PauliX": "x",
        "PauliY": "y",
        "PauliZ": "z",
        "Identity": "id",
        "RX": "rx",
        "RY": "ry",
        "RZ": "rz",
        "PhaseShift": "p",
        "Hadamard": "h",
        "S": "s",
        "Adjoint(S)": "sdg",
        "SX": "sx",
        "Adjoint(SX)": "sxdg",
        "T": "t",
        "Adjoint(T)": "tdg",
        "CNOT": "cx",
        "CY": "cy",
        "CZ": "cz",
        "SWAP": "swap",
        "Z90": "s",
        "ZM90": "sdg",
        "X90": "sx",
        "XM90": "sxdg",
        "Y90": "ry(pi/2)",
        "YM90": "ry(3*pi/2)",
        "TDagger": "tdg",
        "CRY": "cry",
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


def get_labels(up_to: int):
    """gets bitstrings from 0 to "up_to" value

    Args:
        up_to (int): the upper bound for the labels

    Raises:
        ValueError: upper bound must be an int
        ValueError: upper bound must be greater than 0

    Returns:
        list[str]: bitstrings from 0 to the upper bounds
    """

    if not isinstance(up_to, int):
        raise ValueError("up_to must be an int")
    if up_to < 0:
        raise ValueError("up_to must be >= 0")

    num = int(np.log2(up_to)) + 1
    return [format(i, f"0{num}b") for i in range(up_to + 1)]


def get_measurement_wires(tape: QuantumTape):
    """returns the wires that are used for measurement

    Args:
        tape (QuantumTape): the tape from which to find the measurement wires

    Returns:
        list[int]: the measurement wires
    """
    measurement_wires = []
    for mp in tape.measurements:
        measurement_wires += list(mp.wires)
    return set(measurement_wires)


def label_from(number: int, binary_places: int):
    """get a bitstring out of a number. ie 27 with binary place 8 would be 00011011

    Args:
        number (int): the number to turn into a bitstring
        binary_places (int): the number of bits in the string

    Returns:
        str: the bitstring
    """
    return format(number, f"0{binary_places}b")

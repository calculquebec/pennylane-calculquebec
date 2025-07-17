"""
Contains utility functions for optimization
"""

from typing import TypeVar, Callable
from pennylane.operation import Operation
from pennylane.ops import ControlledOp
import pennylane as qml
from pennylane import math
from pennylane.tape import QuantumTape
from pennylane.wires import Wires
import numpy as np
from pennylane_calculquebec.calcul_quebec_error.utility_error import UtilityError

T = TypeVar("T")
U = TypeVar("U")


def expand(
    tape: QuantumTape,
    decomps: dict[str, Callable[[Wires], list[Operation]]],
    iterations=1,
) -> QuantumTape:
    """Decomposes a quantum tape's operations into mapped sets of given operations

    Args:
        tape (QuantumTape): the tape you want to decompose
        decomps (dict[str, Callable[[Wires], list[Operation]]]): a gate name, and the resulting operations
        iterations (int, optional): the number of time it should be ran. Defaults to 1.

    Returns:
        QuantumTape: the quantum tape that results from the decomposition
    """
    # Custom error: tape.operations should be a list of Operations (Pennylane may not always guarantee this)
    if not isinstance(tape.operations, list):
        # Prevents silent failures if tape structure is corrupted or unexpected
        raise UtilityError("tape.operations must be a list of Operations.")
    # Custom error: decomps values should be callable (to avoid silent logic errors if user passes wrong mapping)
    for name, fn in decomps.items():
        if not callable(fn):
            # Prevents silent logic errors if a decomposition is not a function
            raise UtilityError(f"Decomposition for gate '{name}' is not callable.")

    list_copy = tape.operations.copy()
    for _ in range(iterations):
        new_operations = []
        for op in list_copy:
            new_operations += decomps[op.name](op.wires) if op.name in decomps else [op]
        if list_copy == new_operations:
            list_copy = new_operations.copy()
            break
        list_copy = new_operations.copy()
    return type(tape)(list_copy, tape.measurements, tape.shots)


def find_previous_gate(index: int, wires, op_list: list[Operation]) -> int:
    """find first operation that shares a list of wires prior to an index in a list

    Args:
        index (int): the index from which to start
        wires: the wires on which to act (can be list[int] or Wires)
        op_list (list[Operation]): the list of operation to consider

    Returns:
        int: the index of the previous operation. None if none were found
    """
    # Accept Wires object or list, convert to list of ints
    wires = list(wires)
    if not all(isinstance(w, int) for w in wires):
        raise UtilityError("wires must be a list of integers.")

    for i in reversed(range(0, index)):
        if any(w in op_list[i].wires for w in wires):
            return i
    return None


def find_next_gate(index: int, wires, op_list: list[Operation]) -> int:
    """find first operation that shares a list of wires after an index in a list

    Args:
        index (int): the index from which to start
        wires: the wires on which to act (can be list[int] or Wires)
        op_list (list[Operation]): the list of operation to consider

    Returns:
        int: the index of the next operation. None if none were found
    """
    # Accept Wires object or list, convert to list of ints
    wires = list(wires)
    if not all(isinstance(w, int) for w in wires):
        raise UtilityError("wires must be a list of integers.")

    for i in range(index + 1, len(op_list)):
        if any(w in op_list[i].wires for w in wires):
            return i
    return None


def is_single_axis_gate(op: Operation, axis: str):
    """check if given operation is on given basis

    Args:
        op (Operation): the operation to check
        axis (str): the basis to check

    Returns:
        bool: true if operation is on given basis. false otherwise
    """
    # Custom error: op should have a 'basis' attribute (not all Operations may have this)
    if not hasattr(op, "basis"):
        # Prevents attribute errors and clarifies the source of the problem
        raise UtilityError("Operation does not have a 'basis' attribute.")

    if op.num_wires != 1:
        return False
    return op.basis == axis

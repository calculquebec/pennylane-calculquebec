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

T = TypeVar("T")
U = TypeVar("U")
            
def expand(tape : QuantumTape, decomps : dict[str, Callable[[Wires], list[Operation]]], iterations = 1) -> QuantumTape:
    """Decomposes a quantum tape's operations into mapped sets of given operations 

    Args:
        tape (QuantumTape): the tape you want to decompose
        decomps (dict[str, Callable[[Wires], list[Operation]]]): a gate name, and the resulting operations
        iterations (int, optional): the number of time it should be ran. Defaults to 1.

    Returns:
        QuantumTape: the quantum tape that results from the decomposition
    """
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

def find_previous_gate(index : int, wires : list[int], op_list : list[Operation]) -> int:
    """
    find first operation that shares a list of wires prior to an index in a list
    """
    for i in reversed(range(0, index)):
        if any(w in op_list[i].wires for w in wires):
            return i
    return None

def find_next_gate(index : int, wires : list[int], op_list : list[Operation]) -> int:
    """
    find first operation that shares a list of wires after an index in a list
    """
    for i in range(index+1, len(op_list)):
        if any(w in op_list[i].wires for w in wires):
            return i
    return None

def is_single_axis_gate(op : Operation, axis : str):
    if op.num_wires != 1: return False
    return op.basis == axis

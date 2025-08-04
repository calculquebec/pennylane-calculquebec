"""
Contains utility classes for the iterative commute and merge optimization pre-processing step.
"""

from pennylane.tape import QuantumTape
from pennylane_calculquebec.utility.optimization import (
    find_previous_gate,
    find_next_gate,
)
import pennylane.transforms as transforms
import numpy as np
from autograd.numpy.numpy_boxes import ArrayBox
from pennylane.ops.op_math.adjoint import adjoint, Adjoint
from pennylane_calculquebec.logger import logger


def remove_root_zs(tape: QuantumTape, iterations=3) -> QuantumTape:
    """
    removes all heading z operations (the ones on the first layer of a tape)

    Args:
        - tape (QuantumTape) : the tape to act on
        - iteration (int) : the maximum number of times to repeat the operation

    Returns :
        - (QuantumTape) : the resulting quantum tape
    """
    try:
        new_operations = tape.operations.copy()
        for _ in range(iterations):
            list_copy = new_operations.copy()
            new_operations = []

            for i, op in enumerate(list_copy):
                previous = find_previous_gate(i, op.wires, list_copy)
                if op.basis != "Z" or previous is not None:
                    new_operations.append(op)

            if new_operations == list_copy:
                break
        return type(tape)(new_operations, tape.measurements, tape.shots)
    except Exception as e:
        logger.error(
            "Error %s in remove_root_zs located in iterative_commute_and_merge: %s",
            type(e).__name__,
            e,
        )
        return tape


def remove_leaf_zs(tape: QuantumTape, iterations=3) -> QuantumTape:
    """
    removes all tailing z operations (the ones just before a measure with observable Z)

    Args:
        - tape (QuantumTape) : the tape to act on
        - iteration (int) : the maximum number of times to repeat the operation

    Returns :
        - (QuantumTape) : the resulting quantum tape
    """
    try:
        new_operations = tape.operations.copy()
        for _ in range(iterations):
            list_copy = new_operations.copy()
            new_operations = []
            for i in reversed(range(len(list_copy))):
                op = list_copy[i]

                next = find_next_gate(i, op.wires, list_copy)
                if op.basis != "Z" or next is not None:
                    new_operations.insert(0, op)
                    continue

            if new_operations == list_copy:
                break
        return type(tape)(new_operations, tape.measurements, tape.shots)
    except Exception as e:
        logger.error(
            "Error %s in remove_leaf_zs located in iterative_commute_and_merge: %s",
            type(e).__name__,
            e,
        )
        return tape


def _get_adjoint_base(op):
    """
    returns the base of an Adjoint operation

    Args:
        - op (Operation) : a quantum operation

    Returns:
        - Tuple[Operation, bool] : the base operation, and a bool telling us if it was an adjoint or not
    """

    isAdjoint = False
    while isinstance(op, Adjoint):
        isAdjoint = not isAdjoint
        op = op.base
    return op, isAdjoint


def _remove_trivials(tape: QuantumTape, iteration=3, epsilon=1e-8) -> QuantumTape:
    """
    removes 0rad rotations and identities

    Args:
        - tape (QuantumTape) : the tape to act on
        - iteration (int) : the maximum number of times to repeat the operation
        - epsilon (float) : up to which precision do we wish to detect 0 rad rotations

    Returns :
        - (QuantumTape) : the resulting quantum tape
    """
    try:
        new_operations = []
        for op in tape.operations:
            op, isAdjoint = _get_adjoint_base(op)

            if len(op.parameters) > 0:
                angle = op.parameters[0]
                while angle > 2 * np.pi - epsilon:
                    angle -= 2 * np.pi
                while angle < 0:
                    angle += 2 * np.pi
                if abs(angle) > epsilon:
                    op = (type(op) if not isAdjoint else adjoint(type(op)))(
                        angle, wires=op.wires
                    )
                    new_operations.append(op)
            elif op.name == "Identity":
                continue
            else:
                new_operations.append(op)
        return type(tape)(new_operations, tape.measurements, tape.shots)
    except Exception as e:
        logger.error(
            "Error %s in _remove_trivials located in iterative_commute_and_merge: %s",
            type(e).__name__,
            e,
        )
        return tape


def commute_and_merge(tape: QuantumTape) -> QuantumTape:
    """
    applies commutations, rotation merges and inverses/trivial gates cancellations

    Args:
        tape (QuantumTape) : the tape to act on

    Returns :
        QuantumTape : the resulting quantum tape
    """
    try:
        iterations = 3

        for _ in range(iterations):
            new_tape = tape
            new_tape = transforms.commute_controlled(new_tape)[0][0]
            new_tape = remove_root_zs(new_tape)
            new_tape = remove_leaf_zs(new_tape)
            new_tape = transforms.cancel_inverses(new_tape)[0][0]
            new_tape = transforms.merge_rotations(new_tape)[0][0]
            new_tape = _remove_trivials(new_tape)

            new_tape = transforms.commute_controlled(new_tape, "left")[0][0]
            new_tape = remove_root_zs(new_tape)
            new_tape = remove_leaf_zs(new_tape)
            new_tape = transforms.cancel_inverses(new_tape)[0][0]
            new_tape = transforms.merge_rotations(new_tape)[0][0]
            new_tape = _remove_trivials(new_tape)
            if tape.operations == new_tape.operations:
                tape = new_tape
                break
            else:
                tape = new_tape
        return tape
    except Exception as e:
        logger.error(
            "Error %s in commute_and_merge located in iterative_commute_and_merge: %s",
            type(e).__name__,
            e,
        )

        return tape

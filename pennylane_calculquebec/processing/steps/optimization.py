"""
Contains optimization pre-processing steps
"""

import numpy as np
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane_calculquebec.utility.optimization import expand, is_single_axis_gate
import pennylane.transforms as transforms
from pennylane_calculquebec.processing.optimization_methods.iterative_commute_and_merge import commute_and_merge
from pennylane_calculquebec.processing.interfaces import PreProcStep

class Optimize(PreProcStep):
    """
    Optimization step base class.
    """
    pass

class IterativeCommuteAndMerge(Optimize):
    """
    Decomposes iteratively until the circuit contains only rotations. For each decomposition step, applies commutations, merges and cancellations
    """
    def execute(self, tape):
        """
        decomposes swaps, cnots and hadamards iteratively. Then turns everything to Z and X rotations. \n
        at each iteration, apply commutations, rotation merges, and trivial and inverse gates cancellations

        Args:
            tape (QuantumTape): the tape to optimize

        Returns:
            QuantumTape: an optimized QuantumTape
        """
        tape = commute_and_merge(tape)

        tape = expand(tape, { "SWAP" : IterativeCommuteAndMerge.swap_cnot})
        tape = commute_and_merge(tape)
        
        tape = expand(tape, { "CNOT" : IterativeCommuteAndMerge.HCZH_cnot })
        tape = commute_and_merge(tape)
        
        tape = expand(tape, { "Hadamard" : IterativeCommuteAndMerge.ZXZ_Hadamard })
        tape = commute_and_merge(tape)
        
        tape = transforms.create_expand_fn(depth=3, stop_at=lambda operation: operation.name in ["RZ", "RX", "RY", "CZ"])(tape)
        tape = commute_and_merge(tape)
        
        tape = IterativeCommuteAndMerge.get_rid_of_y_rotations(tape)
        tape = commute_and_merge(tape)
        return tape    
    
    @staticmethod
    def swap_cnot(wires):
        """turns swaps into cnots
        """
        if len(wires) != 2:
            raise ValueError("SWAPs must be given two wires")
        
        return [ 
            qml.CNOT([wires[0], wires[1]]),
            qml.CNOT([wires[1], wires[0]]),
            qml.CNOT([wires[0], wires[1]])
        ]
            
    @staticmethod
    def HCZH_cnot(wires):
        """turns cnots into H - CZ - H
        """
        if len(wires) != 2:
            raise ValueError("cnots must be given two wires")
        
        return [
            qml.Hadamard(wires[1]),
            qml.CZ(wires),
            qml.Hadamard(wires[1])
        ]

    @staticmethod
    def ZXZ_Hadamard(wires):
        """turns H into S - SX - S
        """
        
        if len(wires) != 1:
            raise ValueError("Hadamards must be given one wire")
        
        return [
            qml.S(wires),
            qml.SX(wires),
            qml.S(wires)
        ]

    @staticmethod
    def Y_to_ZXZ(operation):
        """turns RY into RZ - RX - RZ"""
        if len(operation.wires) != 1:
            raise ValueError("Single qubit rotations must be given one wire")
        
        if operation.basis != "Y":
            raise ValueError("Operation must be in the Y basis")
        rot_angles = operation.single_qubit_rot_angles()
        return [qml.RZ(np.pi/2, operation.wires), qml.RX(rot_angles[1], operation.wires), qml.RZ(-np.pi/2, operation.wires)]

    @staticmethod
    def get_rid_of_y_rotations(tape : QuantumTape):
        """removes all Y rotations"""
        list_copy = tape.operations.copy()
        new_operations = []
        for operation in list_copy:
            if not is_single_axis_gate(operation, "Y"): 
                new_operations += [operation]
            else:
                new_operations += IterativeCommuteAndMerge.Y_to_ZXZ(operation)
        return type(tape)(new_operations, tape.measurements, tape.shots)

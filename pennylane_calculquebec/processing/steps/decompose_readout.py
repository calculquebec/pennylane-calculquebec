"""
contains a pre-processing step for decomposing readouts that are not observed from the computational basis
"""

from pennylane_calculquebec.processing.interfaces import PreProcStep
from pennylane.tape import QuantumTape
import pennylane as qml
import numpy as np
class DecomposeReadout(PreProcStep):
    """
    a pre-processing step that decomposes readouts so they are all made in the computational basis \n
    Supported observables are : X, Y, Z and H
    """
    def get_ops_for_product(self, observable):
        """
        decomposes a product of observables to separated operations on different wires

        Args:
            obs : a product of observables

        Raises:
            ValueError: will be risen if the observable is not supported. 

        Returns:
            A list of operations
        """
        operations = []
        for operation in observable.operands:
            if operation.name in self.observable_dict:
                for wire in operation.wires:
                    operations.append(self.observable_dict[operation.name](wire))
                continue
            
            raise ValueError("this readout observable is not supported")
        return operations
        
    @property
    def observable_dict(self):
        """a dictionary of observables and their respective rotation operations
        """
        return {
            "PauliZ" : lambda wire : qml.Identity(wire),
            "PauliX" : lambda wire : qml.RY(np.pi / 2, wire),
            "PauliY" : lambda wire : qml.RX(-np.pi / 2, wire),
            "Hadamard" : lambda wire : qml.RY(np.pi / 4, wire)
        }
        
    def execute(self, tape : QuantumTape):
        """
        implementation of the execution method from pre-processing steps. \n
        for each observable, if it is a product, decompose it. \n
        if it is a single observable, add the right rotation before the readout, 
        and change the observable to computational basis

        Args:
            tape (QuantumTape): the tape with the readouts to decompose

        Raises:
            ValueError: risen if an observable is not supported

        Returns:
            _type_: a readout with only computational basis observables
        """
        operations = tape.operations.copy()
        measurements = []
        for measurement in tape.measurements:
            # if there is no obs, skip
            if measurement.obs is None:
                measurements.append(measurement)
                continue
            
            # if op is supported, apply rotation and change mp's observable to Z
            if measurement.obs.name in self.observable_dict:
                wires = [wire for wire in measurement.obs.wires]
                for wire in wires:
                    operations.append(self.observable_dict[measurement.obs.name](wire))
                measurements.append(type(measurement)(wires=wires))
                continue
            
            # if op is a product, get the list of rotations that represent this product, and change mp's observable to Z
            if measurement.obs.name == "Prod":
                wires = [wire for wire in measurement.obs.wires]
                for operation in self.get_ops_for_product(measurement.obs):
                    operations.append(operation)
                measurements.append(type(measurement)(wires=wires))
                continue
                
                
            # if we reach this point, it means that we can't readout on this observable
            raise ValueError("this readout observable is not supported")
        
        return type(tape)(operations, measurements, shots=tape.shots)
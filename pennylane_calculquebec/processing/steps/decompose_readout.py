"""
contains a pre-processing step for decomposing readouts that are not observed from the computational basis
"""

from pennylane_calculquebec.processing.interfaces import PreProcStep
from pennylane.tape import QuantumTape
import pennylane as qml
from pennylane.operation import Observable
from pennylane.ops import Prod
import numpy as np
from pennylane_calculquebec.processing.processing_exception import ProcessingException
class DecomposeReadout(PreProcStep):
    """
    a pre-processing step that decomposes readouts so they are all made in the computational basis
    """
    def get_ops_for_product(self, observable : Prod):
        """
        decomposes a product of observables to separated operations on different wires

        Args:
            obs : a product of observables

        Raises:
            ValueError: will be risen if the observable is not supported. 

        Returns:
            A list of operations
        """
        if not observable.is_hermitian:
            raise ProcessingException(f"The observable {observable} is not supported")
        
        operations = []
        for operation in observable.operands:
            if isinstance(operation, Observable):
                operations += operation.diagonalizing_gates()
                continue
            
            raise ProcessingException("The observable {observable} is not supported")
        return operations

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

            if not measurement.obs.is_hermitian:
                raise ProcessingException(f"The observable {measurement.obs} is not supported")
            
            # if op is supported, apply rotation and change mp's observable to Z
            if isinstance(measurement.obs, Observable):
                wires = [wire for wire in measurement.obs.wires]
                operations += measurement.obs.diagonalizing_gates()
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
            raise ProcessingException(f"The observable {measurement.obs} is not supported")
        
        return type(tape)(operations, measurements, shots=tape.shots)

import pennylane_snowflurry
import pennylane as qml
import numpy as np
import logging
from pennylane.tape import QuantumTape

class Test_PennylaneConverterClass:

    def test_quantumTape(self):
        ops = [qml.BasisState(np.array([1,1]), wires=(0,"a"))]
        quantumTape = QuantumTape(ops, [qml.expval(qml.PauliZ(0))])
        converter = pennylane_snowflurry.PennylaneConverter(quantumTape)
        assert isinstance(converter.pennylane_circuit, QuantumTape)
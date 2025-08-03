import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane_calculquebec.utility.api import ApiUtility, keys

def test_measurement_bits_are_sequential_and_start_at_zero():
    wires = [3, 7, 1]
    tape = QuantumTape(ops=[], measurements=[qml.counts(wires=wires)], shots=1000)
    result = ApiUtility.convert_circuit(tape)
    # Find all readout operations
    readouts = [op for op in result[keys.OPERATIONS] if op[keys.TYPE] == "readout"]
    # Bits should be [0, 1, 2] in order
    bits = [op[keys.BITS][0] for op in readouts]
    assert bits == list(range(len(wires)))
    # Qubits should match the wires order
    qubits = [op[keys.QUBITS][0] for op in readouts]
    assert qubits == wires
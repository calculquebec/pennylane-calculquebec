from pennylane.tape import QuantumTape
import pennylane as qml
import numpy as np

def get_label(value, num_wires):
    result = []
    for _ in range(int(num_wires)):
        if value <= 0:
            result.insert(0, 0)
        else:
            result.insert(0, 1 if (value & 1 == 1) else 0)
        value >>= 1
    return result

def same(a, b, tol = 1e-5):
    return abs(a - b) < tol


def compare_result(result1, result2):
    num_wires = np.log2(len(result1))
    dissimilar_results = []

    for i, (a, b) in enumerate(zip(result1, result2)):
        if a == 0 and b == 0:
            continue
        if same(a, b):
            continue
        
        dissimilar_results.append(i)
    
    connected = [0 for _ in range(int(num_wires))]
    for dissimilar in dissimilar_results:
        label = get_label(dissimilar, num_wires)
        connected = [a ^ b for a, b in zip(label, connected)]

    return dissimilar_results

def get_groups(tape : QuantumTape):
    dev = qml.device("default.qubit")

    result = qml.execute([tape], dev)[0]
    
    groups = {}

    for wire in tape.wires:
        test_tape = type(tape)([qml.PauliX(wire)] + tape.operations, tape.measurements)
        test_result = qml.execute([test_tape], dev)[0]
        connected = compare_result(result, test_result)
        groups[wire] = connected
    
    return groups

tape = QuantumTape([qml.Hadamard(0), qml.PauliX(1), qml.CNOT([0, 2])], [qml.probs()])

groups = get_groups(tape)
print(groups)
import pennylane as qml
from pennylane_calculquebec.processing.steps import MonarqDecomposition
from pennylane.workflow import construct_tape
from pennylane.drawer import tape_text

  # Circuit using non-native ops
dev = qml.device("default.qubit", wires=2)
@qml.qnode(dev)
def circuit():
    qml.SWAP(wires=[0, 1])  # placeholder non-native
    return qml.expval(qml.PauliZ(0))

  # Convert to MonarQ native gates
step = MonarqDecomposition()
native = step.execute(construct_tape(circuit)())
print(tape_text(native))  # only MonarQ primitive gates
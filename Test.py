import pennylane as qml
from pennylane.workflow import construct_tape
from pennylane.drawer import tape_text, tape_mpl
from pennylane_calculquebec.processing.steps import CliffordTDecomposition

# 1. define device & QNode
dev = qml.device("default.qubit", wires=1)
@qml.qnode(dev)
def circuit():
    qml.T(wires=0)
    qml.RZ(0.5, wires=0)
    return qml.expval(qml.PauliZ(0))

# 2. build raw tape (user ops only)
tape = construct_tape(circuit, level="top")()

# 3. apply your CalculQuébec decomposition
step       = CliffordTDecomposition()
decomposed = step.execute(tape)

# 4a. text drawing
print(tape_text(decomposed, show_wire_labels=True))

# 4b. matplotlib drawing
fig, ax = tape_mpl(decomposed)
ax.set_title("Clifford+T Decomposed Circuit")
fig.savefig("decomposed_circuit.png")  # or plt.show()

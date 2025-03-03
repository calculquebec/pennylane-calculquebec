import pennylane as qml
from pennylane_calculquebec.API.client import MonarqClient

client = MonarqClient("https://manager.anyonlabs.com", "boucherf", "jtl00AlxQsmOPaJsKYJP2+AF8E27rzS0")
dev = qml.device("monarq.backup", client = client, shots=1000)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(0)
    qml.CNOT([0, 1])
    return qml.counts()

print(circuit())
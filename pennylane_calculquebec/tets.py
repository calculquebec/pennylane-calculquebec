import pennylane as qml
from dotenv import dotenv_values
from pennylane_calculquebec.API.client import MonarqClient

client = MonarqClient("https://manager.anyonlabs.com", "boucherf", "jtl00AlxQsmOPaJsKYJP2+AF8E27rzS0")

dev = qml.device("monarq.backup", wires=[0, 1, 2, 3], shots=1000, client=client)

dev.circuit_name = "mon_circuit"
dev.project_name = "default"

@qml.qnode(dev)
def circuit():
    qml.H(0)
    qml.CNOT([0, 1])
    qml.CNOT([1, 2])
    qml.CNOT([2, 3])

    return qml.counts()

print(circuit())
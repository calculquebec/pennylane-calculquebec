import pennylane as qml
import numpy as np
from pennylane_calculquebec.processing.config import MonarqDefaultConfig
from pennylane_calculquebec.processing.steps import PrintTape
from pennylane_calculquebec.API.client import MonarqClient
from pennylane_calculquebec.API.adapter import ApiAdapter

client = MonarqClient("https://manager.anyonlabs.com", "boucherf", "jtl00AlxQsmOPaJsKYJP2+AF8E27rzS0")
dev = qml.device("monarq.default", machine_name = "yukon", client = client, processing_config = config, shots=1000)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(0)
    qml.CNOT([0, 1])
    return qml.probs()

print(circuit())
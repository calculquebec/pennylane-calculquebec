import pennylane as qml
from pennylane_calculquebec.API.client import MonarqClient
import pytest
from unittest.mock import patch
from pennylane_calculquebec.processing.config import MonarqDefaultConfig

config = MonarqDefaultConfig(False)
client = MonarqClient("test", "test", "test")
dev = qml.device("monarq.default", wires=[0, 1, 2, 3], client = client, shots = 1000, processing_config=config)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=2)
    qml.Hadamard(wires=3)
    qml.PauliZ(wires=3)
    qml.CNOT(wires=[1, 3])
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=2)
    return qml.counts(wires=[0, 1, 2])

class Response:
    def __init__(self, content):
        self.status_code = 200
        self.text = content

def test_integration():
    with patch("requests.post") as post:
        post.return_value = Response('{"job" : {"id" : 1}}')
        with patch("requests.get") as get:
            get.return_value = Response('{"job" : {"status" : {"type" : "SUCCEEDED"}}, "result":{"histogram": {"0":500, "1":500}}}')
            results = circuit()
            assert results is not None
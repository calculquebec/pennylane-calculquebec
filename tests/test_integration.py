import pennylane as qml
from pennylane_calculquebec.API.client import MonarqClient
import pytest
from unittest.mock import patch
from pennylane_calculquebec.processing.config import MonarqDefaultConfig

def circuit():
    qml.Hadamard(0)
    return qml.counts(wires=[0])

class Response:
    def __init__(self, content):
        self.status_code = 200
        self.text = content

def test_monarq_default():
    config = MonarqDefaultConfig(False)
    client = MonarqClient("test", "test", "test")
    dev = qml.device("monarq.default", wires=[0], client = client, shots = 1000, processing_config=config)

    qnode = qml.QNode(circuit, dev)
    with patch("requests.post") as post:
        post.return_value = Response('{"job" : {"id" : 1}}')
        with patch("requests.get") as get:
            get.return_value = Response('{"job" : {"status" : {"type" : "SUCCEEDED"}}, "result":{"histogram": {"0":500, "1":500}}}')
            results = qnode()
            assert results is not None
            assert len(results) == 2 and all(result in results for result in ["0", "1"])

def test_monarq_sim():
    config = MonarqDefaultConfig(False)
    client = MonarqClient("test", "test", "test")
    dev = qml.device("monarq.sim", wires=[0], client = client, shots = 1000, processing_config=config)

    qnode = qml.QNode(circuit, dev)
    results = qnode()
    assert results is not None
    assert len(results) == 2 and all(result in results for result in ["0", "1"])

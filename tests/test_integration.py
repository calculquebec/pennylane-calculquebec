import pennylane as qml
from pennylane_calculquebec.API.client import CalculQuebecClient
import pytest
from unittest.mock import patch
import pennylane_calculquebec.monarq_data as data
from pennylane_calculquebec.processing.config import MonarqDefaultConfig


def circuit():
    qml.Hadamard(0)
    return qml.counts(wires=[0])


class Response:
    def __init__(self, content):
        self.status_code = 200
        self.text = content


@pytest.fixture
def mock_get_connectivity():
    with patch("pennylane_calculquebec.utility.graph.get_connectivity") as mock:
        mock.side_effect = lambda machine_name, use_benchmark: data.cache["yamaska"][
            data.Cache.OFFLINE_CONNECTIVITY
        ]
        yield mock


def test_monarq_default(mock_get_connectivity):
    config = MonarqDefaultConfig("yamaska", False)
    client = CalculQuebecClient(
        "test",
        "test",
        "test",
        project_id="test_project_id",
        circuit_name="test_circuit",
    )
    dev = qml.device(
        "monarq.default", wires=[0], client=client, processing_config=config
    )
    assert dev._client is client
    assert dev._client.machine_name == dev.machine_name
    qnode = qml.QNode(circuit, dev)
    qml.set_shots(qnode, 1000)
    with patch("requests.post") as post:
        post.return_value = Response('{"job" : {"id" : 1}}')
        with patch("requests.get") as get:
            get.return_value = Response(
                '{"items" : [{"id" : 0}], "job" : {"status" : {"type" : "SUCCEEDED"}}, "result":{"histogram": {"0":500, "1":500}}}'
            )

            results = qnode()
            assert results is not None
            assert len(results) == 2 and all(result in results for result in ["0", "1"])


def test_monarq_sim(mock_get_connectivity):
    config = MonarqDefaultConfig("yamaska", False)
    dev = qml.device("monarq.sim", wires=[0], processing_config=config)

    qnode = qml.QNode(circuit, dev)
    qml.set_shots(qnode, 1000)
    results = qnode()
    assert results is not None
    assert len(results) == 2 and all(result in results for result in ["0", "1"])

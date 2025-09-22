from pennylane_calculquebec.API.adapter import (
    ApiAdapter,
    ApiException,
    MultipleProjectsException,
    NoProjectFoundException,
)
from pennylane_calculquebec.API.client import CalculQuebecClient
import pytest
from unittest.mock import patch
from pennylane_calculquebec.utility.api import ApiUtility, keys
from datetime import datetime, timedelta

client = CalculQuebecClient("test", "test", "test", project_id="123")


# ------------ MOCKS ----------------------


def raise_exception(*args, **kwargs):
    raise Exception()


@pytest.fixture
def mock_list_machines():
    with patch("pennylane_calculquebec.API.adapter.ApiAdapter.list_machines") as mock:
        yield mock


@pytest.fixture
def mock_get_project_id_by_name():
    with patch(
        "pennylane_calculquebec.API.adapter.ApiAdapter.get_project_id_by_name"
    ) as mock:
        yield mock


@pytest.fixture
def mock_requests_get():
    with patch("requests.get") as requests_get:
        yield requests_get


@pytest.fixture
def mock_requests_post():
    with patch("requests.post") as request_post:
        yield request_post


@pytest.fixture
def mock_get_benchmark():
    with patch(
        "pennylane_calculquebec.API.adapter.ApiAdapter.get_benchmark"
    ) as get_benchmark:
        yield get_benchmark


@pytest.fixture
def mock_is_last_update_expired():
    with patch(
        "pennylane_calculquebec.API.adapter.ApiAdapter.is_last_update_expired"
    ) as is_last_update_expired:
        yield is_last_update_expired


@pytest.fixture
def mock_get_machine_by_name():
    with patch(
        "pennylane_calculquebec.API.adapter.ApiAdapter.get_machine_by_name"
    ) as get_machine_by_name:
        return get_machine_by_name


@pytest.fixture
def mock_job_body():
    with patch("pennylane_calculquebec.utility.api.ApiUtility.job_body") as job_body:
        return job_body


class Res:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text


@pytest.fixture(autouse=True)
def setup():
    ApiAdapter._instance = None


# ------------- TESTS ---------------------


def test_exception():
    code = 42
    message = "test"
    ex = ApiException(code, message)
    assert str(code) in ex.message and message in ex.message


def test_initialize():
    with pytest.raises(Exception):
        ApiAdapter()

    assert ApiAdapter.instance() is None
    ApiAdapter.initialize(client)
    assert ApiAdapter.instance() is not None

    headers = ApiUtility.headers("test", "test", "calculqc")
    assert ApiAdapter.instance().headers == headers


def test_is_last_update_expired():
    ApiAdapter._last_update = datetime.now() - timedelta(hours=25)
    assert ApiAdapter.is_last_update_expired()
    ApiAdapter._last_update = datetime.now() - timedelta(hours=5)
    assert not ApiAdapter.is_last_update_expired()


def test_get_qubits_and_couplers(mock_get_benchmark):
    ApiAdapter.clean_cache()
    ApiAdapter.initialize(client)
    mock_get_benchmark.return_value = {keys.RESULTS_PER_DEVICE: True}
    result = ApiAdapter.get_qubits_and_couplers("yamaska")
    assert result


def test_get_benchmark(
    mock_is_last_update_expired, mock_get_machine_by_name, mock_requests_get
):
    test_benchmark_str = '{"test" : "im a benchmark"}'
    test_benchmark_str2 = '{"test" : "im a benchmark2"}'

    test_machine_str = '{"items" : [{"id" : "3"}]}'
    mock_get_machine_by_name.return_value = {keys.ITEMS: [{keys.ID: "3"}]}
    mock_is_last_update_expired.return_value = False

    test_benchmark = {"test": "im a benchmark"}
    ApiAdapter.clean_cache()
    ApiAdapter.initialize(client)

    # test 200 and cache is None
    mock_requests_get.side_effect = lambda route, headers: (
        Res(200, test_machine_str)
        if "benchmark" not in route
        else Res(200, test_benchmark_str)
    )

    benchmark = ApiAdapter.get_benchmark("yamaska")
    assert all(test_benchmark[k] == benchmark[k] for k in benchmark)

    # test last_update < 24 h
    mock_requests_get.side_effect = lambda route, headers: (
        Res(400, test_machine_str)
        if "benchmark" not in route
        else Res(400, test_benchmark_str2)
    )
    benchmark = ApiAdapter.get_benchmark("yamaska")
    assert all(test_benchmark[k] == benchmark[k] for k in benchmark)

    # test 400 and last_update > 24 h
    mock_is_last_update_expired.return_value = True
    with pytest.raises(Exception):
        benchmark = ApiAdapter.get_benchmark("yamaska")


def test_post_job(mock_job_body, mock_get_project_id_by_name, mock_requests_post):
    ApiAdapter.initialize(client)

    mock_get_project_id_by_name.return_value = 0

    mock_requests_post.return_value = Res(200, "{'jobID': 'a_job_uuid'}")
    assert ApiAdapter.post_job(circuit={}).text == "{'jobID': 'a_job_uuid'}"

    mock_requests_post.return_value = Res(400, '{"error" : 42}')
    with pytest.raises(Exception):
        ApiAdapter.post_job(circuit={})


def test_list_jobs(mock_requests_get):
    ApiAdapter.clean_cache()
    ApiAdapter.initialize(client)

    mock_requests_get.return_value = Res(200, 42)
    assert ApiAdapter.list_jobs().text == 42
    mock_requests_get.return_value = Res(400, 42)

    with pytest.raises(Exception):
        ApiAdapter.list_jobs()


def test_job_by_id(mock_requests_get):
    ApiAdapter.clean_cache()
    ApiAdapter.initialize(client)

    mock_requests_get.return_value = Res(200, 42)
    assert ApiAdapter.job_by_id(None).text == 42

    mock_requests_get.return_value = Res(400, 42)
    with pytest.raises(Exception):
        ApiAdapter.job_by_id(None)


def test_list_machines(mock_requests_get):
    ApiAdapter.clean_cache()
    ApiAdapter.initialize(client)
    mock_requests_get.side_effect = raise_exception

    with pytest.raises(Exception):
        ApiAdapter.list_machines()

    mock_requests_get.side_effect = lambda route, headers: Res(
        400,
        '{"items" : [{"status" : "online", "answer" : 42}, {"status" : "offline", "answer" : 42}]}',
    )

    with pytest.raises(Exception):
        ApiAdapter.list_machines()

    mock_requests_get.side_effect = lambda route, headers: Res(
        200,
        '{"items" : [{"status" : "online", "answer" : 42}, {"status" : "offline", "answer" : 42}]}',
    )
    assert len(ApiAdapter.list_machines()) == 2
    assert len(ApiAdapter.list_machines(True)) == 1


def test_get_connectivity_for_machine(mock_list_machines):
    ApiAdapter.clean_cache()
    ApiAdapter.initialize(client)

    mock_list_machines.side_effect = raise_exception

    with pytest.raises(Exception):
        ApiAdapter.get_connectivity_for_machine("a")

    mock_list_machines.side_effect = lambda: [
        {"name": "a", "couplerToQubitMap": 1},
        {"name": "b", "couplerToQubitMap": 2},
    ]

    assert ApiAdapter.get_connectivity_for_machine("a") == 1
    assert ApiAdapter.get_connectivity_for_machine("b") == 2

    with pytest.raises(Exception):
        ApiAdapter.get_connectivity_for_machine("c")


def test_get_machine_by_name(mock_requests_get):
    ApiAdapter.clean_cache()
    ApiAdapter.initialize(client)

    mock_requests_get.side_effect = raise_exception
    with pytest.raises(Exception):
        ApiAdapter.get_machine_by_name("yamaska")

    mock_requests_get.side_effect = lambda body, headers: Res(
        400, '{"name" : "yamaska", "status" : "online", "answer" : 42}'
    )

    with pytest.raises(Exception):
        ApiAdapter.get_machine_by_name("yamaska")

    mock_requests_get.side_effect = lambda body, headers: Res(
        200, '{"name" : "yamaska", "status" : "online", "answer" : 42}'
    )

    assert ApiAdapter.get_machine_by_name("yamaska")["name"] == "yamaska"


def test_get_project_id_by_name(mock_requests_get):
    """
    Test the get_project_id_by_name method of ApiAdapter.

    The method should handle various scenarios including:

    - Raising an ApiException when the request fails.
    - Raising an ApiException when the response is not as expected.
    - Raising an MultipleProjectsException when multiple projects with the same name are found.
    - Raising an NoProjectFoundException when no projects are found for the given name.
    - Successfully returning the project ID when a single project is found with the given name.
    """

    ApiAdapter.clean_cache()

    ApiAdapter.initialize(client)

    # Request fails
    mock_requests_get.side_effect = raise_exception
    with pytest.raises(Exception):
        ApiAdapter.get_project_id_by_name("project0")

    # Response is not as expected
    mock_requests_get.side_effect = lambda body, headers: Res(
        400, '{"name" : "project0", "status" : "online", "answer" : 42}'
    )
    with pytest.raises(ApiException):
        ApiAdapter.get_project_id_by_name("project0")

    # Multiple projects with the same name
    mock_requests_get.side_effect = lambda body, headers: Res(
        200,
        '{"items" : [{"id" : 42, "name" : "project0"}, {"id" : 43, "name" : "project0"}]}',
    )
    with pytest.raises(MultipleProjectsException):
        ApiAdapter.get_project_id_by_name("project0")

    # No projects found for the given name
    mock_requests_get.side_effect = lambda body, headers: Res(
        200, '{"items" : [{"id" : 42, "name" : "project1"}]}'
    )
    with pytest.raises(NoProjectFoundException):
        ApiAdapter.get_project_id_by_name("project0")

    # Single project found with the given name
    mock_requests_get.side_effect = lambda body, headers: Res(
        200,
        '{"items" : [{"id" : 42, "name" : "project0"}, {"id" : 43, "name" : "project1"}]}',
    )
    assert ApiAdapter.get_project_id_by_name("project0") == 42

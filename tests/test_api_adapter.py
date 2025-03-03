from pennylane_calculquebec.API.adapter import ApiAdapter, ApiException
from pennylane_calculquebec.API.client import MonarqClient
import pytest
from unittest.mock import patch
from pennylane_calculquebec.utility.api import ApiUtility, keys
from datetime import datetime, timedelta
client = MonarqClient("test", "test", "test")

# ------------ MOCKS ----------------------

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
    with patch("pennylane_calculquebec.API.adapter.ApiAdapter.get_benchmark") as get_benchmark:
        yield get_benchmark
    
@pytest.fixture
def mock_is_last_update_expired():
    with patch("pennylane_calculquebec.API.adapter.ApiAdapter.is_last_update_expired") as is_last_update_expired:
        yield is_last_update_expired

@pytest.fixture
def mock_get_machine_by_name():
    with patch("pennylane_calculquebec.API.adapter.ApiAdapter.get_machine_by_name") as get_machine_by_name:
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


def test_initialize():
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

def test_get_machine_by_name(mock_requests_get):
    machine_test_str = '{"test" : "im a machine"}'
    machine_test = {"test" : "im a machine"}
    mock_requests_get.return_value.status_code = 200
    mock_requests_get.return_value.text = machine_test_str
    
    ApiAdapter.clean_cache()
    ApiAdapter.initialize(client)
    
    machine = ApiAdapter.get_machine_by_name("yamaska")
    
    assert all([machine[k] == machine_test[k] for k in machine])
    
    mock_requests_get.return_value.status_code = 400
    # test cache 
    machine = ApiAdapter.get_machine_by_name("yamaska")
    assert all([machine[k] == machine_test[k] for k in machine])
    
    ApiAdapter._machine = None  
    
    with pytest.raises(ApiException):
        machine = ApiAdapter.get_machine_by_name("yamaska")

def test_get_qubits_and_couplers(mock_get_benchmark):
    ApiAdapter.clean_cache()
    ApiAdapter.initialize(client)
    mock_get_benchmark.return_value = {keys.RESULTS_PER_DEVICE : True}
    result = ApiAdapter.get_qubits_and_couplers("yamaska")
    assert result

def test_get_benchmark(mock_is_last_update_expired, mock_get_machine_by_name, mock_requests_get):
    test_benchmark_str = '{"test" : "im a benchmark"}'
    test_benchmark_str2 = '{"test" : "im a benchmark2"}'

    test_machine_str = '{"items" : [{"id" : "3"}]}'
    mock_get_machine_by_name.return_value = {keys.ITEMS : [{keys.ID : "3"}]}
    mock_is_last_update_expired.return_value = False
    
    test_benchmark = {"test" : "im a benchmark"}
    ApiAdapter.clean_cache()
    ApiAdapter.initialize(client)
    
    # test 200 and cache is None
    mock_requests_get.side_effect = lambda route, headers: \
        Res(200, test_machine_str) if "benchmark" not in route \
            else Res(200, test_benchmark_str)
            
    benchmark = ApiAdapter.get_benchmark("yamaska")
    assert all(test_benchmark[k] == benchmark[k] for k in benchmark)

    # test last_update < 24 h
    mock_requests_get.side_effect = lambda route, headers: \
        Res(400, test_machine_str) if "benchmark" not in route \
            else Res(400, test_benchmark_str2)
    benchmark = ApiAdapter.get_benchmark("yamaska")
    assert all(test_benchmark[k] == benchmark[k] for k in benchmark)    

    # test 400 and last_update > 24 h
    mock_is_last_update_expired.return_value = True
    with pytest.raises(ApiException):
        benchmark = ApiAdapter.get_benchmark("yamaska")

def test_create_job(mock_job_body, mock_requests_post):
    ApiAdapter.initialize(client)
    
    mock_requests_post.return_value = Res(200, 42)
    assert ApiAdapter.create_job(None, "yamaska").text == 42
    
    mock_requests_post.return_value = Res(400, 42)
    with pytest.raises(ApiException):
        ApiAdapter.create_job(None, "yamaska")
    

def test_list_jobs(mock_requests_get):
    ApiAdapter.clean_cache()
    ApiAdapter.initialize(client)
    
    mock_requests_get.return_value = Res(200, 42)
    assert ApiAdapter.list_jobs().text == 42
    mock_requests_get.return_value = Res(400, 42)
    with pytest.raises(ApiException):
        ApiAdapter.list_jobs()

def test_job_by_id(mock_requests_get):
    ApiAdapter.clean_cache()
    ApiAdapter.initialize(client)
    
    mock_requests_get.return_value = Res(200, 42)
    assert ApiAdapter.job_by_id(None).text == 42
    
    mock_requests_get.return_value = Res(400, 42)
    with pytest.raises(ApiException):
        ApiAdapter.job_by_id(None)

def test_list_machines(mock_requests_get):
    ApiAdapter.clean_cache()
    ApiAdapter.initialize(client)

    mock_requests_get.return_value = Res(200, '{"items" : [{"status" : "online", "answer" : 42}, {"status" : "offline", "answer" : 42}]}')
    assert len(ApiAdapter.list_machines()) == 2
    assert len(ApiAdapter.list_machines(True)) == 1

def test_get_machine_by_name(mock_requests_get):
    ApiAdapter.clean_cache()
    ApiAdapter.initialize(client)

    mock_requests_get.return_value = Res(200, '{"name" : "yamaska", "status" : "online", "answer" : 42}')
    assert ApiAdapter.get_machine_by_name("yamaska")["name"] == "yamaska"
from pennylane_calculquebec.API.adapter import ApiAdapter
from pennylane_calculquebec.API.client import MonarqClient
import pytest
from unittest.mock import patch
from pennylane_calculquebec.API.job import Job, JobException
import json

client = MonarqClient("test", "test", "test")

class Response_JobById:
    def __init__(self, status_code, job_status):
        dict = lambda status : {"job" : {"status" : {"type" : str(status)}}, \
            "result" : {"histogram" : 42}}
        self.status_code = status_code
        
        if status_code == 400:
            self.text = '{"code" : 400, "error" : "this is an error"}'
        elif status_code == 200:
            self.text = json.dumps(dict(job_status))
  
class Response_Error:
    def __init__(self):
        self.text = '{"code":400, "error": "this is an error"}'
        
class Circuit:
    i = 0
    class Shots:
        def __init__(self):
            self.total_shots = 10
            
    def __init__(self):
        self.shots = Circuit.Shots()
        
        self.operations = []
        self.measurements = []
        self.wires = []

@pytest.fixture(name="mock_convert_circuit")
def mock_convert_circut():
    with patch("pennylane_calculquebec.utility.api.ApiUtility.convert_circuit") as convert_circuit:
        yield convert_circuit

@pytest.fixture(name="mock_create_job")
def mock_create_job():
    with patch("pennylane_calculquebec.API.adapter.ApiAdapter.create_job") as create_job:
        yield create_job

@pytest.fixture(name="mock_job_by_id")
def mock_job_by_id():
    with patch("pennylane_calculquebec.API.adapter.ApiAdapter.job_by_id") as job_by_id:
        yield job_by_id


def test_run(mock_convert_circuit, mock_create_job, mock_job_by_id):    
    test_job_str = '{"job" : {"id" : 3}}'
    test_error_str = '{"code" : 400, "error" : "this is an error"}'

    def side_effect_generator(status_code):
        def side_effect(_):
            Circuit.i += 1
            if Circuit.i % 3 != 0:
                return Response_JobById(status_code, "no")
            return Response_JobById(status_code, "SUCCEEDED")
        return side_effect
    
    ApiAdapter.initialize(client)
    
    mock_create_job.return_value.status_code = 400
    mock_create_job.return_value.text = test_error_str
    
    # create job => code 400
    Circuit.i = 0
    with pytest.raises(JobException):
        result = Job(Circuit()).run()
        
    mock_create_job.return_value.status_code = 200    
    mock_create_job.return_value.text = test_job_str
    mock_job_by_id.side_effect = side_effect_generator(200)
    
    # typical flow
    Circuit.i = 0
    result = Job(Circuit()).run()
    assert result == 42
    assert Circuit.i == 3
    
    # job_by_id => code 400
    mock_job_by_id.side_effect = side_effect_generator(400)
    
    Circuit.i = 0
    with pytest.raises(JobException):
        result = Job(Circuit()).run()
        
    # runs past iteration limit
    mock_job_by_id.side_effect = side_effect_generator(200)
    Circuit.i = 0
    with pytest.raises(JobException):
        Job(Circuit()).run(2)
    Circuit.i == 2
    
def test_raise_api_error():
    response = Response_Error()
    with pytest.raises(JobException):
        Job(Circuit()).raise_api_error(response)
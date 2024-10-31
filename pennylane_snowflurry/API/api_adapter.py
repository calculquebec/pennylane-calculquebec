from pennylane.tape import QuantumTape
from pennylane.operation import Operation
from pennylane.measurements import MeasurementProcess
from pennylane_snowflurry.API.api_utility import ApiUtility
from dotenv import dotenv_values
import requests
import json
from pennylane_snowflurry.device_configuration import Client
class ApiAdapter(object):
    """
    a wrapper around Thunderhead. Provide a host, user, access token and realm, and you can :
    - create job with circuit dict, circuit name, project id, machine name and shots count
    - get benchmark by machine name
    - get machine id by name
    """
    def __init__(self):
        raise Exception("Call instance() instead")
    
    client : Client
    headers : dict[str, str]
    _instance : "ApiAdapter"
    
    @classmethod
    def instance(cls):
        return cls._instance
    
    @classmethod
    def initialize(cls, client : Client):
        cls._instance = cls.__new__(cls)
        cls._instance.headers = ApiUtility.headers(client.user, client.access_token, client.realm)
        cls.client = client
    
    @staticmethod
    def get_machine_id_by_name():
        route = ApiAdapter.instance().client.host + ApiUtility.routes.machines + ApiUtility.routes.machineName + "=" + ApiAdapter.instance().client.machine_name
        return requests.get(route, headers=ApiAdapter.instance().headers)
    
    @staticmethod
    def get_qubits_and_couplers() -> dict[str, any] | None:
        res = ApiAdapter.instance().get_benchmark()
        if res.status_code != 200:
            return None
        return json.loads(res.text)[ApiUtility.keys.resultsPerDevice]

    @staticmethod
    def get_benchmark():
        res = ApiAdapter.instance().get_machine_id_by_name()
        if res.status_code != 200:
            return None
        result = json.loads(res.text)
        machine_id = result[ApiUtility.keys.items][0][ApiUtility.keys.id]
    
        route = ApiAdapter.instance().client.host + ApiUtility.routes.machines + "/" + machine_id + ApiUtility.routes.benchmarking
        return requests.get(route, headers=ApiAdapter.instance().headers)
    
    @staticmethod
    def create_job(circuit : dict[str, any], 
                   circuit_name: str = "default",
                   shot_count : int = 1) -> requests.Response:
        body = ApiUtility.job_body(circuit, circuit_name, ApiAdapter.instance().client.project_name, ApiAdapter.instance().client.machine_name, shot_count)
        return requests.post(ApiAdapter.instance().client.host + ApiUtility.routes.jobs, data=json.dumps(body), headers=ApiAdapter.instance().headers)

    @staticmethod
    def list_jobs() -> requests.Response:
        return requests.get(ApiAdapter.instance().client.host + ApiUtility.routes.jobs, headers=ApiAdapter.instance().headers)

    @staticmethod
    def job_by_id(id : str) -> requests.Response:
        return requests.get(ApiAdapter.instance().client.host + ApiUtility.routes.jobs + f"/{id}", headers=ApiAdapter.instance().headers)

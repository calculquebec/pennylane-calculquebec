from pennylane_snowflurry.utility.api import ApiUtility, routes, keys
import requests
import json
from pennylane_snowflurry.API.client import ApiClient


class ApiAdapter(object):
    """
    a wrapper around Thunderhead. Provide a host, user, access token and realm, and you can :
    - create jobs with circuit dict, circuit name, project id, machine name and shots count
    - get benchmark by machine name
    - get machine id by name
    """
    def __init__(self):
        raise Exception("Call ApiAdapter.initialize(ApiClient) and ApiAdapter.instance() instead")
    
    client : ApiClient
    headers : dict[str, str]
    _instance : "ApiAdapter"
    
    @classmethod
    def instance(cls):
        """
        unique ApiAdapter instance
        """
        return cls._instance
    
    @classmethod
    def initialize(cls, client : ApiClient):
        """
        create a unique ApiAdapter instance
        """
        cls._instance = cls.__new__(cls)
        cls._instance.headers = ApiUtility.headers(client.user, client.access_token, client.realm)
        cls.client = client
    
    @staticmethod
    def get_machine_id_by_name():
        """
        get the id of a machine by using the machine's name stored in the client
        """
        route = ApiAdapter.instance().client.host + routes.machines + routes.machineName + "=" + ApiAdapter.instance().client.machine_name
        return requests.get(route, headers=ApiAdapter.instance().headers)
    
    @staticmethod
    def get_qubits_and_couplers() -> dict[str, any] | None:
        """
        get qubits and couplers informations from latest benchmark
        """
        res = ApiAdapter.instance().get_benchmark()
        if res.status_code != 200:
            return None
        return json.loads(res.text)[keys.resultsPerDevice]

    @staticmethod
    def get_benchmark():
        """
        get latest benchmark for a given machine (machine name stored in client)
        """
        res = ApiAdapter.instance().get_machine_id_by_name()
        if res.status_code != 200:
            return None
        result = json.loads(res.text)
        machine_id = result[keys.items][0][keys.id]
    
        route = ApiAdapter.instance().client.host + routes.machines + "/" + machine_id + routes.benchmarking
        return requests.get(route, headers=ApiAdapter.instance().headers)
    
    @staticmethod
    def create_job(circuit : dict[str, any], 
                   circuit_name: str = "default",
                   shot_count : int = 1) -> requests.Response:
        """
        post a new job for running a specific circuit a certain amount of times on given machine (machine name stored in client)
        """
        body = ApiUtility.job_body(circuit, circuit_name, ApiAdapter.instance().client.project_name, ApiAdapter.instance().client.machine_name, shot_count)
        return requests.post(ApiAdapter.instance().client.host + routes.jobs, data=json.dumps(body), headers=ApiAdapter.instance().headers)

    @staticmethod
    def list_jobs() -> requests.Response:
        """
        get all jobs for a given user (user stored in client)
        """
        return requests.get(ApiAdapter.instance().client.host + routes.jobs, headers=ApiAdapter.instance().headers)

    @staticmethod
    def job_by_id(id : str) -> requests.Response:
        """
        get a job for a given user by providing its id (user stored in client)
        """
        return requests.get(ApiAdapter.instance().client.host + routes.jobs + f"/{id}", headers=ApiAdapter.instance().headers)

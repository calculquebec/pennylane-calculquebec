"""Contains the ApiAdapter singleton class, which wraps every API call necessary for communicating with MonarQ
"""
from pennylane_calculquebec.utility.api import ApiUtility, routes, keys
import requests
import json
from pennylane_calculquebec.API.client import ApiClient
from datetime import datetime, timedelta

class ApiException(Exception):
    def __init__(self, code, message):
        self.message = f"API ERROR : {code}, {message}"
    
    def __str__(self):
        return self.message

    def __repr__(self):
        return self.message

class ApiAdapter(object):
    _qubits_and_couplers = None
    _machine = None
    _benchmark = None
    _last_update = None
    
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
    _instance : "ApiAdapter" = None
    
    @staticmethod
    def clean_cache():
        ApiAdapter._qubits_and_couplers = None
        ApiAdapter._machine = None
        ApiAdapter._benchmark = None
        ApiAdapter._last_update = None

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
        cls._instance.client = client

        cls._qubits_and_couplers = None
        cls._machine = None
        cls._benchmark = None
        cls._last_update = None
    
    @staticmethod
    def is_last_update_expired():
        return datetime.now() - ApiAdapter._last_update > timedelta(hours=24)
    
    @staticmethod
    def get_machine_by_name(machine_name):
        """
        get the id of a machine by using the machine's name stored in the client
        """
        # put machine in cache
        if ApiAdapter._machine is None:
            route = ApiAdapter.instance().client.host + routes.MACHINES + routes.MACHINE_NAME + "=" + machine_name
            res = requests.get(route, headers=ApiAdapter.instance().headers)
            if res.status_code != 200:
                ApiAdapter.raise_exception(res)
            ApiAdapter._machine = json.loads(res.text)
            
        return ApiAdapter._machine
    
    @staticmethod
    def get_qubits_and_couplers(machine_name : str) -> dict[str, any] | None:
        """
        get qubits and couplers informations from latest benchmark for given machine
        """
        
        benchmark = ApiAdapter.get_benchmark(machine_name)
        return benchmark[keys.RESULTS_PER_DEVICE]

    @staticmethod
    def get_benchmark(machine_name):
        """
        get latest benchmark for a given machine
        """

        # put benchmark in cache
        if ApiAdapter._benchmark is None or ApiAdapter.is_last_update_expired():
            machine = ApiAdapter.get_machine_by_name(machine_name)
            machine_id = machine[keys.ITEMS][0][keys.ID]

            route = ApiAdapter.instance().client.host + routes.MACHINES + "/" + machine_id + routes.BENCHMARKING
            res = requests.get(route, headers=ApiAdapter.instance().headers)
            if res.status_code != 200:
                ApiAdapter.raise_exception(res)
            ApiAdapter._benchmark = json.loads(res.text)
            ApiAdapter._last_update = datetime.now()
            
        return ApiAdapter._benchmark
    
    @staticmethod
    def create_job(circuit : dict[str, any], 
                   machine_name : str,
                   circuit_name: str = "default",
                   shot_count : int = 1) -> requests.Response:
        """
        post a new job for running a specific circuit a certain amount of times on given machine (machine name stored in client)
        """
        body = ApiUtility.job_body(circuit, circuit_name, ApiAdapter.instance().client.project_name, machine_name, shot_count)
        res = requests.post(ApiAdapter.instance().client.host + routes.JOBS, data=json.dumps(body), headers=ApiAdapter.instance().headers)
        if res.status_code != 200:
            ApiAdapter.raise_exception(res)
        return res

    @staticmethod
    def list_jobs() -> requests.Response:
        """
        get all jobs for a given user (user stored in client)
        """
        res = requests.get(ApiAdapter.instance().client.host + routes.JOBS, headers=ApiAdapter.instance().headers)
        if res.status_code != 200:
            ApiAdapter.raise_exception(res)
        return res

    @staticmethod
    def job_by_id(id : str) -> requests.Response:
        """
        get a job for a given user by providing its id (user stored in client)
        """
        res = requests.get(ApiAdapter.instance().client.host + routes.JOBS + f"/{id}", headers=ApiAdapter.instance().headers)
        if res.status_code != 200:
            ApiAdapter.raise_exception(res)
        return res

    @staticmethod
    def list_machines(online_only : bool = False):
        """
        get a list of available machines
        """
        res = requests.get(ApiAdapter.instance().client.host + routes.MACHINES, headers=ApiAdapter.instance().headers)
        if res.status_code != 200:
            ApiAdapter.raise_exception(res)
        return [m for m in json.loads(res.text)[keys.ITEMS] if not online_only or m[keys.STATUS] == keys.ONLINE]

    def get_connectivity_for_machine(machine_name):
        """
        get connectivity of a machine (given its name)
        """
        machines = ApiAdapter.instance().list_machines()
        target = [m for m in machines if m[keys.NAME] == machine_name]
        if len(target) < 1:
            raise ApiException(f"No machine available with name {machine_name}")
        
        return target[0][keys.COUPLER_TO_QUBIT_MAP]

    @staticmethod
    def raise_exception(res):
        raise ApiException(res.status_code, res.text)
"""Contains the ApiAdapter singleton class, which wraps every API call necessary for communicating with MonarQ"""

from pennylane_calculquebec.utility.api import ApiUtility, routes, keys, queries
import requests
import json
from pennylane_calculquebec.API.client import ApiClient
from datetime import datetime, timedelta
from pennylane_calculquebec.API.retry_decorator import retry


class ApiException(Exception):
    """
    an exception that is thrown when something goes wrong in the ApiAdapter

    Args:
        - code (int) : the http status code that represents the error
        - message (str) : the message for the error
    """

    def __init__(self, code: int, message: str):
        self.message = f"API ERROR : {code}, {message}"
        super().__init__(self.message)


# TODO : Move this exception to a separate module
class ProjectException(Exception):
    """
    An exception that is thrown when something goes wrong the parsing of a project

    Args:
        - message (str) : the message for the error
    """

    def __init__(self, message: str):
        self.message = f"PROJECT ERROR : {message}"
        super().__init__(self.message)


class MultipleProjectsException(ProjectException):
    """
    An exception that is thrown when multiple projects with the same name are found.
    It displays the projects names and corresponding IDs.

    Args:
        - projects (list) : the list of projects that were found with the same name
    """

    def __init__(self, projects: list):

        message = (
            f"Multiple projects found with the same name. When creating client, "
            f"please use the project ID instead of the name.\n"
            "Projects found:\n"
        )
        for project in projects:
            message += (
                f"Project Name: {project[keys.NAME]}, Project ID: {project[keys.ID]}\n"
            )
        super().__init__(message)


class NoProjectFoundException(ProjectException):
    """
    An exception that is thrown when no project with the given name is found.
    It displays the name of the project that was not found.

    Args:
        - message (str) : the message for the error
    """

    def __init__(self, project_name: str):
        message = f"No project found with name: {project_name}"
        super().__init__(message)


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
        raise Exception(
            "Call ApiAdapter.initialize(ApiClient) and ApiAdapter.instance() instead"
        )

    client: ApiClient
    headers: dict[str, str]
    _instance: "ApiAdapter" = None

    @staticmethod
    def clean_cache():
        """
        Cleans all cache values
        """
        """
        Cleans all cache values
        """
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
    def initialize(cls, client: ApiClient):
        """
        Create a unique ApiAdapter instance

        Args :
            client (ApiClient) : The client to initialize ApiAdapter with
        Create a unique ApiAdapter instance

        Args :
            client (ApiClient) : The client to initialize ApiAdapter with
        """
        cls._instance = cls.__new__(cls)
        cls._instance.headers = ApiUtility.headers(
            client.user, client.access_token, client.realm
        )
        cls._instance.client = client
        if client.project_name != "":
            cls._instance.client.project_id = ApiAdapter.get_project_id_by_name(
                client.project_name
            )

        cls._qubits_and_couplers: dict = None
        cls._machine: dict = None
        cls._benchmark: dict = None
        cls._last_update: datetime = None

    @staticmethod
    def is_last_update_expired():
        """
        Checks if the last update has been done more than 24 h ago

        Returns:
            bool : Was the last update more than 24 h ago?
        """
        """
        Checks if the last update has been done more than 24 h ago

        Returns:
            bool : Was the last update more than 24 h ago?
        """
        return datetime.now() - ApiAdapter._last_update > timedelta(hours=24)

    @staticmethod
    @retry(3)
    def get_project_id_by_name(project_name: str = "default") -> str:
        """
        Get the id of a project by using the project's name stored in the client

        Args:
            project_name (str) : The name of the project you want to fetch. Defaults to "default"

        Returns:
            project_id (str) : The id of the project
        """
        res = requests.get(
            ApiAdapter.instance().client.host
            + routes.PROJECTS
            + queries.NAME
            + "="
            + project_name,
            headers=ApiAdapter.instance().headers,
        )

        if res.status_code != 200:
            ApiAdapter.raise_exception(res)

        converted = json.loads(res.text)

        projects = converted.get(keys.ITEMS, [])
        matching_projects = [
            project for project in projects if project.get(keys.NAME) == project_name
        ]

        if len(matching_projects) > 1:
            raise MultipleProjectsException(matching_projects)

        if len(matching_projects) == 1:
            return matching_projects[0][keys.ID]

        raise NoProjectFoundException(project_name)

    @staticmethod
    @retry(3)
    def get_machine_by_name(machine_name: str) -> dict:
        """
        Get the id of a machine by using the machine's name stored in the client

        Args:
            machine_name (str) : The name of the machine you want to fetch

        Returns:
            dict : The machine information in a dictionary
        """
        # put machine in cache
        if ApiAdapter._machine is None:
            route = (
                ApiAdapter.instance().client.host
                + routes.MACHINES
                + queries.MACHINE_NAME
                + "="
                + machine_name
            )

            res = requests.get(route, headers=ApiAdapter.instance().headers)

            if res.status_code != 200:
                ApiAdapter.raise_exception(res)
            ApiAdapter._machine = json.loads(res.text)

        return ApiAdapter._machine

    @staticmethod
    @retry(3)
    def get_qubits_and_couplers(machine_name: str) -> dict:
        """
        Get qubits and couplers informations from latest benchmark for given machine

        Args:
            machine_name (str) : The name of the machine you want to fetch

        Return :
            dict : A dictionary with fidelity values (T1, T2, Q1 fidelities, Q2 fidelities, readout 1, readout 0)
        """

        benchmark = ApiAdapter.get_benchmark(machine_name)
        return benchmark[keys.RESULTS_PER_DEVICE]

    @staticmethod
    @retry(3)
    @retry(3)
    def get_benchmark(machine_name):
        """
        get latest benchmark for a given machine

        Args:
            machine_name (str) : the name of the machine you want to fetch

        Return :
            dict : a dictionary all benchmark information for the machine
        """

        # put benchmark in cache
        if ApiAdapter._benchmark is None or ApiAdapter.is_last_update_expired():
            machine = ApiAdapter.get_machine_by_name(machine_name)
            machine_id = machine[keys.ITEMS][0][keys.ID]

            route = (
                ApiAdapter.instance().client.host
                + routes.MACHINES
                + "/"
                + machine_id
                + routes.BENCHMARKING
            )
            res = requests.get(route, headers=ApiAdapter.instance().headers)
            if res.status_code != 200:
                ApiAdapter.raise_exception(res)
            ApiAdapter._benchmark = json.loads(res.text)
            ApiAdapter._last_update = datetime.now()

        return ApiAdapter._benchmark

    @staticmethod
    @retry(3)
    def post_job(
        circuit: dict,
        shot_count: int = 1,
    ) -> requests.Response:
        """
        Post a new job for running a specific circuit a certain number of times on given machine (machine name stored in client)

        Args:
            circuit (dict) : The dictionary representation of a circuit
            shot_count (int) : The number of shots. default is 1

        Returns:
            Response : The response of the /job post request
        """
        project_id = ApiAdapter.instance().client.project_id
        circuit_name = ApiAdapter.instance().client.circuit_name
        machine_name = ApiAdapter.instance().client.machine_name
        body = ApiUtility.job_body(
            circuit, circuit_name, project_id, machine_name, shot_count
        )
        res = requests.post(
            ApiAdapter.instance().client.host + routes.JOBS,
            data=json.dumps(body),
            headers=ApiAdapter.instance().headers,
        )
        if res.status_code != 200:
            ApiAdapter.raise_exception(res)
        return res

    @staticmethod
    @retry(3)
    @retry(3)
    def list_jobs() -> requests.Response:
        """
        get all jobs for a given user (user stored in client)

        Returns:
            Response : the response of the /jobs get request

        Returns:
            Response : the response of the /jobs get request
        """
        res = requests.get(
            ApiAdapter.instance().client.host + routes.JOBS,
            headers=ApiAdapter.instance().headers,
        )
        if res.status_code != 200:
            ApiAdapter.raise_exception(res)
        return res

    @staticmethod
    @retry(3)
    def job_by_id(id: str) -> requests.Response:
        """
        Get a job for a given user by providing its id (user stored in client)

        Args:
            id (str) : The id of the job you want to get

        Returns:
            Response : The response of the /job/id get request
        """
        res = requests.get(
            ApiAdapter.instance().client.host + routes.JOBS + f"/{id}",
            headers=ApiAdapter.instance().headers,
        )
        if res.status_code != 200:
            ApiAdapter.raise_exception(res)
        return res

    @staticmethod
    @retry(3)
    def list_machines(online_only: bool = False) -> list[dict]:
        """
        Get a list of available machines

        Args:
            online_only (bool) : Only return machines that are online. Defaults to False

        Returns:
            list[dict] : The list of dictionaries representing machines
        """
        res = requests.get(
            ApiAdapter.instance().client.host + routes.MACHINES,
            headers=ApiAdapter.instance().headers,
        )
        if res.status_code != 200:
            ApiAdapter.raise_exception(res)
        return [
            m
            for m in json.loads(res.text)[keys.ITEMS]
            if not online_only or m[keys.STATUS] == keys.ONLINE
        ]

    def get_connectivity_for_machine(machine_name: str) -> dict:
        """
        Get connectivity of a machine (given its name)

        Args:
            machine_name (str) : The name of the machine you want to fetch

        Returns:
            dict : dictionary that represents the connectivity of the machine
        """
        machines = ApiAdapter.list_machines()
        target = [m for m in machines if m[keys.NAME] == machine_name]
        if len(target) < 1:
            raise ApiException(f"No machine available with name {machine_name}")

        return target[0][keys.COUPLER_TO_QUBIT_MAP]

    @staticmethod
    def raise_exception(res):
        message = res

        # try to fetch the text from the response
        if hasattr(message, "text"):
            message = message.text

        # try to deserialize the text (it might not be deserializable)
        try:
            message = json.loads(message)
            if "error" in message:
                message = message["error"]
        except Exception:
            pass

        raise ApiException(res.status_code, message)

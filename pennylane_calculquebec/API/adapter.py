"""Contains the ApiAdapter singleton class, which wraps every API call necessary for communicating with MonarQ"""

from pennylane_calculquebec.utility.api import ApiUtility, routes, keys, queries
import requests
import json
from pennylane_calculquebec.API.client import ApiClient
from datetime import datetime, timedelta
from pennylane_calculquebec.API.retry_decorator import retry


class ApiException(Exception):
    """An exception raised when an HTTP error is returned by the MonarQ API.

    Args:
        code (int): the HTTP status code that represents the error
        message (str): the error message
    """

    def __init__(self, code: int, message: str):
        self.message = f"API ERROR : {code}, {message}"
        super().__init__(self.message)


# TODO : Move this exception to a separate module
class ProjectException(Exception):
    """An exception raised when something goes wrong while parsing a project.

    Args:
        message (str): the error message
    """

    def __init__(self, message: str):
        self.message = f"PROJECT ERROR : {message}"
        super().__init__(self.message)


class MultipleProjectsException(ProjectException):
    """An exception raised when multiple projects with the same name are found.

    Displays all matching project names and IDs so the user can switch to
    project-ID-based authentication.

    Args:
        projects (list): the list of projects that share the same name
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
    """An exception raised when no project matching the given name is found.

    Args:
        project_name (str): the name of the project that was not found
    """

    def __init__(self, project_name: str):
        message = f"No project found with name: {project_name}"
        super().__init__(message)


class ApiAdapter(object):
    """A singleton wrapper around the Thunderhead REST API.

    Call :meth:`initialize` once with an :class:`~pennylane_calculquebec.API.client.ApiClient`
    before making any requests.  Afterwards, use :meth:`instance` to access the singleton.

    Cached values (machine, benchmark, qubits/couplers) are invalidated automatically
    after 24 hours via :meth:`is_last_update_expired`.
    """

    _qubits_and_couplers = None
    _machine = None
    _benchmark = None
    _last_update = None

    def __init__(self):
        raise Exception(
            "Call ApiAdapter.initialize(ApiClient) and ApiAdapter.instance() instead"
        )

    client: ApiClient
    headers: dict[str, str]
    _instance: "ApiAdapter" = None

    @staticmethod
    def clean_cache():
        """Invalidate all cached API values.

        Forces the next request for machine, benchmark, or qubit/coupler data
        to perform a fresh API call.
        """
        ApiAdapter._qubits_and_couplers = None
        ApiAdapter._machine = None
        ApiAdapter._benchmark = None
        ApiAdapter._last_update = None

    @classmethod
    def instance(cls):
        """Return the current singleton instance.

        Returns:
            ApiAdapter: the unique adapter instance created by :meth:`initialize`
        """
        return cls._instance

    @classmethod
    def initialize(cls, client: ApiClient):
        """Create (or replace) the singleton ApiAdapter instance.

        Builds the authentication headers from ``client`` and, when
        ``client.project_name`` is set, resolves it to a project ID via
        :meth:`get_project_id_by_name`.

        Args:
            client (ApiClient): client credentials and configuration used for
                every subsequent API request
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
        """Check whether the cached data is older than 24 hours.

        Returns:
            bool: ``True`` if the last successful cache update was more than
            24 hours ago, ``False`` otherwise.
        """
        return datetime.now() - ApiAdapter._last_update > timedelta(hours=24)

    @staticmethod
    @retry(3)
    def get_project_id_by_name(project_name: str = "default") -> str:
        """Resolve a project name to its unique project ID.

        Args:
            project_name (str): the name of the project to look up.
                Defaults to ``"default"``.

        Returns:
            str: the unique project ID

        Raises:
            MultipleProjectsException: if more than one project shares ``project_name``
            NoProjectFoundException: if no project matches ``project_name``
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
        """Fetch machine metadata by name, caching the result in memory.

        The result is cached for the lifetime of the adapter; call
        :meth:`clean_cache` to force a fresh lookup.

        Args:
            machine_name (str): the name of the machine to fetch (e.g. ``"yamaska"``)

        Returns:
            dict: raw machine metadata returned by the API
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
        """Return per-device fidelity data from the latest benchmark.

        Retrieves the ``resultsPerDevice`` section of the benchmark, which
        contains T1, T2, single-qubit gate fidelities, CZ gate fidelities,
        and readout state-0/state-1 fidelities for each qubit and coupler.

        Args:
            machine_name (str): the name of the machine to query

        Returns:
            dict: fidelity values keyed by qubit/coupler identifiers
        """

        benchmark = ApiAdapter.get_benchmark(machine_name)
        return benchmark[keys.RESULTS_PER_DEVICE]

    @staticmethod
    @retry(3)
    def get_benchmark(machine_name):
        """Fetch the latest calibration benchmark for a machine, caching it for 24 hours.

        Args:
            machine_name (str): the name of the machine to query (e.g. ``"yamaska"``)

        Returns:
            dict: full benchmark payload including per-device fidelity results
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
        """Submit a new circuit execution job to MonarQ.

        Args:
            circuit (dict): dictionary representation of the circuit
                (as produced by :func:`~pennylane_calculquebec.utility.api.ApiUtility.convert_circuit`)
            shot_count (int): number of shots to execute. Defaults to ``1``.

        Returns:
            requests.Response: the HTTP response from the job-creation endpoint
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
    def list_jobs() -> requests.Response:
        """Retrieve all jobs associated with the authenticated user.

        Returns:
            requests.Response: the HTTP response from the jobs listing endpoint
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
        """Retrieve a specific job by its unique identifier.

        Args:
            id (str): the unique job ID to look up

        Returns:
            requests.Response: the HTTP response from the job detail endpoint
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
        """Return a list of available machines.

        Args:
            online_only (bool): when ``True``, only machines whose status is
                ``"online"`` are returned. Defaults to ``False``.

        Returns:
            list[dict]: each element is a dictionary of machine metadata
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

    @staticmethod
    def get_connectivity_for_machine(machine_name: str) -> dict:
        """Return the coupler-to-qubit connectivity map for a machine.

        Args:
            machine_name (str): the name of the machine to query

        Returns:
            dict: mapping of coupler identifiers to the pair of connected qubits

        Raises:
            ApiException: if no machine with ``machine_name`` is available
        """
        machines = ApiAdapter.list_machines()
        target = [m for m in machines if m[keys.NAME] == machine_name]
        if len(target) < 1:
            raise ApiException(f"No machine available with name {machine_name}")

        return target[0][keys.COUPLER_TO_QUBIT_MAP]

    @staticmethod
    def raise_exception(res):
        """Parse an unsuccessful HTTP response and raise an :class:`ApiException`.

        Attempts to deserialize the response body as JSON and extract an
        ``"error"`` field; falls back to the raw text if deserialization fails.

        Args:
            res (requests.Response): the failed HTTP response

        Raises:
            ApiException: always raised with the extracted status code and message
        """
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

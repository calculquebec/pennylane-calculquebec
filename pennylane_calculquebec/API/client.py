"""
Contains base client class and implementations. \n
MonarQ users will mostly use MonarqClient.
"""


class ProjectParameterError(ValueError):
    """
    Exception raised when project parameter validation fails.

    This is a specialized ValueError that can be raised for various
    project parameter validation scenarios in client constructors.
    The specific error message is provided when the exception is raised.

    Args:
        message (str): A descriptive error message explaining the validation failure.

    Example:
        raise ProjectParameterError("Either project_name or project_id must be provided")
        raise ProjectParameterError("Invalid project_id format")
    """

    pass


class ApiClient:
    """
    data object that is used to pass client information to CalculQCDevice

    Args :
        host (str) : the server address for the machine
        user (str) : the users identifier
        access_token (str) : the unique access key provided to the user
        realm (str) : the organisational group associated with the machine
        machine_name (str) : the name of the machine
        project_name (str) : the name of the project
        project_id (str) : the ID of the project
    """

    @property
    def project_name(self):
        """Returns the project name."""
        return self._project_name

    @property
    def project_id(self):
        """Returns the project ID."""
        return self._project_id

    @project_id.setter
    def project_id(self, value: str):
        """Sets the project ID."""
        self._project_id = value

    @property
    def circuit_name(self):
        """Returns the circuit name."""
        return self._circuit_name

    def __init__(
        self,
        host: str,
        user: str,
        access_token: str,
        realm: str,
        machine_name: str,
        project_name: str = None,
        project_id: str = None,
        circuit_name: str = None,
    ):
        # Validation of project_name and project_id parameters
        if project_name is None and project_id is None:
            raise ProjectParameterError(
                "Either project_name or project_id must be provided"
            )

        if project_name is not None and project_id is not None:
            # If both are provided, use only project_id
            project_name = None

        self.host = host
        self.user = user
        self.access_token = access_token
        self.realm = realm
        self.machine_name = machine_name
        self._project_name = project_name or ""
        self._project_id = project_id or ""
        self._circuit_name = circuit_name or "none"


class CalculQuebecClient(ApiClient):
    """
    specialization of Client for Calcul Quebec infrastructures

    Args :
        host (str) : the server address for the machine
        user (str) : the users identifier
        access_token (str) : the unique access key provided to the user
        machine_name (str) : the name of the machine
        project_name (str) : the name of the project (exclusive with project_id)
        project_id (str) : the ID of the project (exclusive with project_name)

    """

    def __init__(
        self,
        host,
        user,
        token,
        machine_name,
        project_name=None,
        project_id=None,
        circuit_name=None,
    ):
        super().__init__(
            host,
            user,
            token,
            "calculqc",
            machine_name,
            project_name,
            project_id,
            circuit_name,
        )


class MonarqClient(CalculQuebecClient):
    """
    specialization of CalculQuebecClient for MonarQ infrastructure

    Args :
        host (str) : the server address for the machine
        user (str) : the users identifier
        access_token (str) : the unique access key provided to the user
        project_name (str) : the name of the project (exclusive with project_id)
        project_id (str) : the ID of the project (exclusive with project_name)

    """

    def __init__(
        self,
        host,
        user,
        access_token,
        project_name=None,
        project_id=None,
        circuit_name=None,
    ):
        super().__init__(
            host,
            user,
            access_token,
            "yamaska",
            project_name,
            project_id,
            circuit_name,
        )

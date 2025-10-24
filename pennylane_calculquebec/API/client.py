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
        circuit_name (str) : the name of the circuit to use for the job
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

    @circuit_name.setter
    def circuit_name(self, value: str):
        """Sets the circuit name."""
        self._circuit_name = value

    @property
    def machine_name(self):
        """Returns the machine name."""
        return self._machine_name

    @machine_name.setter
    def machine_name(self, value: str):
        """Sets the machine name."""
        self._machine_name = value

    def __init__(
        self,
        host: str,
        user: str,
        access_token: str,
        realm: str,
        project_name: str = "",
        project_id: str = "",
        circuit_name: str = "none",
    ):
        # Validation of project_name and project_id parameters
        if project_name == "" and project_id == "":
            raise ProjectParameterError(
                "Either project_name or project_id must be provided"
            )

        if project_name != "" and project_id != "":
            # If both are provided, use only project_id
            project_name = ""

        self.host = host
        self.user = user
        self.access_token = access_token
        self.realm = realm
        self._machine_name = ""
        self._project_name = project_name
        self._project_id = project_id
        self._circuit_name = circuit_name


class CalculQuebecClient(ApiClient):
    """
    specialization of Client for Calcul Quebec infrastructures

    Args :
        host (str) : the server address for the machine
        user (str) : the users identifier
        access_token (str) : the unique access key provided to the user
        project_name (str) : the name of the project (exclusive with project_id)
        project_id (str) : the ID of the project (exclusive with project_name)
        circuit_name (str) : the name of the circuit to use for the job

    """

    def __init__(
        self,
        host,
        user,
        access_token,
        project_name="",
        project_id="",
        circuit_name="none",
    ):
        super().__init__(
            host,
            user,
            access_token,
            "calculqc",
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
        circuit_name (str) : the name of the circuit to use for the job

    """

    # FIXME : deprecate this class in favor of CalculQuebecClient

    def __init__(
        self,
        host,
        user,
        access_token,
        project_name="",
        project_id="",
        circuit_name="none",
    ):
        super().__init__(
            host,
            user,
            access_token,
            project_name,
            project_id,
            circuit_name,
        )
        import warnings

        warnings.warn(
            "MonarqClient is deprecated and will be removed in a future release. Use CalculQuebecClient instead.",
            DeprecationWarning,
            stacklevel=2,
        ),

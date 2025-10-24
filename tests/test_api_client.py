from pennylane_calculquebec.API.client import (
    ApiClient,
    CalculQuebecClient,
    MonarqClient,
    ProjectParameterError,
)
import pytest


# Test fixtures for common test data
@pytest.fixture
def basic_client_params():
    """Basic parameters for client initialization."""
    return {
        "host": "https://example.com",
        "user": "test_user",
        "access_token": "test_token_123",
        "realm": "test_realm",
    }


@pytest.fixture
def project_name():
    """Standard project name for testing."""
    return "test_project"


@pytest.fixture
def project_id():
    """Standard project ID for testing."""
    return "project_123"


@pytest.fixture
def circuit_name():
    """Standard circuit name for testing."""
    return "test_circuit"


@pytest.fixture
def calcul_quebec_params():
    """Parameters specific to CalculQuebecClient."""
    return {
        "host": "https://calculquebec.example.com",
        "user": "cq_user",
        "access_token": "cq_token_456",
    }


@pytest.fixture
def monarq_params():
    """Parameters specific to MonarqClient."""
    return {
        "host": "https://monarq.example.com",
        "user": "monarq_user",
        "access_token": "monarq_token_789",
    }


@pytest.fixture
def all_client_classes():
    """List of all client classes for parametrized tests."""
    return [ApiClient, CalculQuebecClient, MonarqClient]


class TestApiClient:
    """Test cases for the base ApiClient class."""

    def test_api_client_initialization_with_project_name(
        self, basic_client_params, project_name
    ):
        """Test that ApiClient initializes correctly with project_name."""
        client = ApiClient(**basic_client_params, project_name=project_name)

        assert client.host == basic_client_params["host"]
        assert client.user == basic_client_params["user"]
        assert client.access_token == basic_client_params["access_token"]
        assert client.realm == basic_client_params["realm"]
        assert client.project_name == project_name
        assert client.project_id == ""
        assert client.circuit_name == "none"
        assert client.machine_name == ""  # Machine name is set later

    def test_api_client_initialization_with_project_id(
        self, basic_client_params, project_id
    ):
        """Test that ApiClient initializes correctly with project_id."""
        client = ApiClient(**basic_client_params, project_id=project_id)

        assert client.project_id == project_id
        assert client.project_name == ""

    def test_api_client_requires_project_parameter(self, basic_client_params):
        """Test that ApiClient raises error when neither project_name nor project_id is provided."""
        with pytest.raises(
            ProjectParameterError,
            match="Either project_name or project_id must be provided",
        ):
            ApiClient(**basic_client_params)

    def test_api_client_project_id_takes_precedence(
        self, basic_client_params, project_name, project_id
    ):
        """Test that when both project_name and project_id are provided, project_id is used."""
        client = ApiClient(
            **basic_client_params, project_name=project_name, project_id=project_id
        )

        assert client.project_id == project_id
        assert client.project_name == ""

    def test_api_client_attributes_are_strings(self, basic_client_params, project_name):
        """Test that all ApiClient attributes accept string values."""
        client = ApiClient(**basic_client_params, project_name=project_name)

        assert isinstance(client.host, str)
        assert isinstance(client.user, str)
        assert isinstance(client.access_token, str)
        assert isinstance(client.realm, str)
        assert isinstance(client.machine_name, str)
        assert isinstance(client.project_name, str)
        assert isinstance(client.project_id, str)
        assert isinstance(client.circuit_name, str)

    def test_api_client_circuit_name_initialization(
        self, basic_client_params, project_name, circuit_name
    ):
        """Test that ApiClient initializes correctly with circuit_name."""
        client = ApiClient(
            **basic_client_params, project_name=project_name, circuit_name=circuit_name
        )
        assert client.circuit_name == circuit_name

    def test_api_client_circuit_name_defaults_to_none_string(
        self, basic_client_params, project_name
    ):
        """Test that circuit_name defaults to empty string when not provided."""
        client = ApiClient(**basic_client_params, project_name=project_name)
        assert client.circuit_name == "none"


class TestCalculQuebecClient:
    """Test cases for the CalculQuebecClient class."""

    def test_calcul_quebec_client_initialization_with_project_name(
        self, calcul_quebec_params, project_name
    ):
        """Test that CalculQuebecClient initializes correctly with project_name."""
        client = CalculQuebecClient(**calcul_quebec_params, project_name=project_name)

        assert client.host == calcul_quebec_params["host"]
        assert client.user == calcul_quebec_params["user"]
        assert client.access_token == calcul_quebec_params["access_token"]
        assert client.realm == "calculqc"
        assert client.machine_name == ""
        assert client.project_name == project_name
        assert client.project_id == ""

    def test_calcul_quebec_client_initialization_with_project_id(
        self, calcul_quebec_params, project_id
    ):
        """Test that CalculQuebecClient initializes correctly with project_id."""
        client = CalculQuebecClient(**calcul_quebec_params, project_id=project_id)

        assert client.project_id == project_id
        assert client.project_name == ""
        assert client.realm == "calculqc"

    def test_calcul_quebec_client_requires_project_parameter(
        self, calcul_quebec_params
    ):
        """Test that CalculQuebecClient raises error when no project parameter is provided."""
        with pytest.raises(
            ProjectParameterError,
            match="Either project_name or project_id must be provided",
        ):
            CalculQuebecClient(**calcul_quebec_params)

    def test_calcul_quebec_client_project_id_precedence(
        self, calcul_quebec_params, project_name, project_id
    ):
        """Test that project_id takes precedence over project_name in CalculQuebecClient."""
        client = CalculQuebecClient(
            **calcul_quebec_params, project_name=project_name, project_id=project_id
        )

        assert client.project_id == project_id
        assert client.project_name == ""

    def test_calcul_quebec_client_inherits_from_api_client(
        self, calcul_quebec_params, project_name
    ):
        """Test that CalculQuebecClient is a subclass of ApiClient."""
        client = CalculQuebecClient(**calcul_quebec_params, project_name=project_name)
        assert isinstance(client, ApiClient)

    def test_calcul_quebec_client_realm_is_fixed(
        self, calcul_quebec_params, project_name
    ):
        """Test that CalculQuebecClient always sets realm to 'calculqc'."""
        client = CalculQuebecClient(**calcul_quebec_params, project_name=project_name)
        assert client.realm == "calculqc"

    def test_calcul_quebec_circuit_name_defaults_to_none_string(
        self, calcul_quebec_params, project_name
    ):
        """Test that circuit_name defaults to 'none' when not provided."""
        client = CalculQuebecClient(**calcul_quebec_params, project_name=project_name)
        assert client.circuit_name == "none"


class TestMonarqClient:
    """Test cases for the MonarqClient class."""

    def test_monarq_client_initialization_with_project_name(
        self, monarq_params, project_name
    ):
        """Test that MonarqClient initializes correctly with project_name."""
        client = MonarqClient(**monarq_params, project_name=project_name)

        assert client.host == monarq_params["host"]
        assert client.user == monarq_params["user"]
        assert client.access_token == monarq_params["access_token"]
        assert client.realm == "calculqc"
        assert client.machine_name == ""
        assert client.project_name == project_name
        assert client.project_id == ""

    def test_monarq_client_initialization_with_project_id(
        self, monarq_params, project_id
    ):
        """Test that MonarqClient initializes correctly with project_id."""
        client = MonarqClient(**monarq_params, project_id=project_id)

        assert client.project_id == project_id
        assert client.project_name == ""
        assert client.machine_name == ""
        assert client.realm == "calculqc"

    def test_monarq_client_requires_project_parameter(self, monarq_params):
        """Test that MonarqClient raises error when no project parameter is provided."""
        with pytest.raises(
            ProjectParameterError,
            match="Either project_name or project_id must be provided",
        ):
            MonarqClient(**monarq_params)

    def test_monarq_client_project_id_precedence(
        self, monarq_params, project_name, project_id
    ):
        """Test that project_id takes precedence over project_name in MonarqClient."""
        client = MonarqClient(
            **monarq_params, project_name=project_name, project_id=project_id
        )

        assert client.project_id == project_id
        assert client.project_name == ""

    def test_monarq_client_inherits_from_calcul_quebec_client(
        self, monarq_params, project_name
    ):
        """Test that MonarqClient is a subclass of CalculQuebecClient."""
        client = MonarqClient(**monarq_params, project_name=project_name)
        assert isinstance(client, CalculQuebecClient)
        assert isinstance(client, ApiClient)

    def test_monarq_client_inheritance_chain(self, monarq_params, project_name):
        """Test the complete inheritance chain for MonarqClient.
        i.e. MonarqClient --|> CalculQuebecClient --|> ApiClient
        """
        client = MonarqClient(**monarq_params, project_name=project_name)

        # Test that it has all the expected attributes from parent classes
        assert hasattr(client, "host")
        assert hasattr(client, "user")
        assert hasattr(client, "access_token")
        assert hasattr(client, "realm")
        assert hasattr(client, "machine_name")
        assert hasattr(client, "project_name")
        assert hasattr(client, "project_id")
        assert hasattr(client, "circuit_name")

    def test_deprecation_warning(self, monarq_params, project_name):
        """Test that MonarqClient raises a deprecation warning when initialized."""
        with pytest.warns(DeprecationWarning):
            MonarqClient(**monarq_params, project_name=project_name)


class TestProjectParameterValidation:
    """Test cases for project_name and project_id validation rules at Parent class level."""

    def test_project_name_is_readonly(self, basic_client_params, project_name):
        """Test that project_name is read-only after initialization."""
        client = ApiClient(**basic_client_params, project_name=project_name)
        with pytest.raises(AttributeError):
            client.project_name = "new_project"

    def test_project_id_is_writable(self, basic_client_params, project_id):
        """Test that project_id can be set after initialization."""
        client = ApiClient(**basic_client_params, project_id=project_id)
        assert client.project_id == project_id
        client.project_id = "new_project_id"
        assert client.project_id == "new_project_id"

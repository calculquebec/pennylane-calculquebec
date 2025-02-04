from unittest.mock import patch
import pytest

@pytest.fixture
def mock_snowflurry_device():
    with patch("pennylane_calculquebec.snowflurry_device.SnowflurryQubitDevice") as mock:
        yield mock

@pytest.fixture
def mock_find_spec():
    with patch("importlib.util.find_spec") as mock:
        yield mock

def test_snowflurry_device(mock_find_spec):
    mock_find_spec.return_value = None

    with pytest.raises(Exception):
        import pennylane_calculquebec.snowflurry_device
    
    old_call_count = mock_find_spec.call_count

    mock_find_spec.return_value = "not null"

    import pennylane_calculquebec.snowflurry_device

def test_pennylane_converter(mock_find_spec):
    mock_find_spec.return_value = None

    with pytest.raises(Exception):
        import pennylane_calculquebec.pennylane_converter
    
    old_call_count = mock_find_spec.call_count

    mock_find_spec.return_value = "not null"

    import pennylane_calculquebec.pennylane_converter

def test_pennylane_calculquebec_none_lib(mock_find_spec):
    import sys
    mock_find_spec.return_value = None

    import pennylane_calculquebec
    modules = sorted([k for k in sys.modules.keys() if "snowflurry_device" in k])
    assert len(modules) == 0

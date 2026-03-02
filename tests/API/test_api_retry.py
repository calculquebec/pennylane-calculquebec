from pennylane_calculquebec.API.retry_decorator import retry
import pytest
from unittest.mock import patch, Mock
import time
import warnings

# ------------ TEST FUNCTIONS ----------------------


def function_succeeds():
    """Function that always succeeds"""
    return "Success"


def function_always_fails():
    """Function that always fails with a ValueError"""
    raise ValueError("Test exception")


def function_fails_then_succeeds(max_failures):
    """Function that fails a number of times and then succeeds.

    The way it works is by initializing a static variable attribute `calls` on the func object.
    The attribute is incremented each time the function is called and raises a ValueError until
    it reaches `max_failures`.

    Args:
        max_failures (int): The number of times the function should fail before succeeding.
    """
    if hasattr(function_fails_then_succeeds, "calls"):
        function_fails_then_succeeds.calls += 1
    else:
        function_fails_then_succeeds.calls = 1

    if function_fails_then_succeeds.calls <= max_failures:
        raise ValueError(f"Failure {function_fails_then_succeeds.calls}")
    return "Success after failures"


# ------------ MOCKS ----------------------


@pytest.fixture
def reset_function_state():
    """Reset the state of test functions between tests"""
    if hasattr(function_fails_then_succeeds, "calls"):
        delattr(function_fails_then_succeeds, "calls")


@pytest.fixture
def mock_sleep():
    with patch("pennylane_calculquebec.API.retry_decorator.sleep") as mock:
        yield mock


@pytest.fixture
def mock_logger_error():
    with patch("pennylane_calculquebec.API.retry_decorator.logger.error") as mock:
        yield mock


@pytest.fixture
def mock_logger_warning():
    with patch("pennylane_calculquebec.API.retry_decorator.logger.warning") as mock:
        yield mock


# ------------- TESTS ---------------------


def test_successful_execution_no_retries(mock_sleep):
    """Test that a successful function executes without any retries"""
    decorated_func = retry()(function_succeeds)
    result = decorated_func()

    assert result == "Success"
    mock_sleep.assert_not_called()


def test_failed_execution_with_retries(
    mock_sleep, mock_logger_error, mock_logger_warning
):
    """Test that a failing function retries the correct number of times"""
    retries = 5
    expected_failed_attempt = retries - 1

    decorated_func = retry(retries=retries)(function_always_fails)

    with pytest.raises(ValueError, match="Test exception"):
        decorated_func()

    # Check the correct number of sleep calls (one less than retries since the last attempt doesn't sleep)
    assert mock_sleep.call_count == expected_failed_attempt

    # Check that we are logging warnings for each failed attempt except the last
    assert mock_logger_warning.call_count == expected_failed_attempt

    # Check final warning message
    mock_logger_error.assert_called_with(
        f"The request failed after {retries} retries. \nThis was caused by inner exception: \nTest exception"
    )


def test_exponential_backoff(mock_sleep):
    """Test that the delay increases exponentially with each retry"""
    initial_delay = 0.1
    backoff_factor = 2.0
    retries = 5

    decorated_func = retry(
        retries=retries, initial_delay=initial_delay, backoff_factor=backoff_factor
    )(function_always_fails)

    with pytest.raises(ValueError):
        decorated_func()

    # Check that sleep was called with exponentially increasing delays
    expected_delays = [
        initial_delay,
        initial_delay * backoff_factor,
        initial_delay * backoff_factor**2,
        initial_delay * backoff_factor**3,
    ]

    actual_calls = [args[0] for args, _ in mock_sleep.call_args_list]
    assert actual_calls == expected_delays


def test_eventual_success(mock_sleep, reset_function_state):
    """Test that the function eventually succeeds after a few failures"""
    failures = 3
    retries = 5

    decorated_func = retry(retries=retries)(
        lambda: function_fails_then_succeeds(failures)
    )

    result = decorated_func()

    assert result == "Success after failures"
    # Should have slept exactly the number of failures times
    assert mock_sleep.call_count == failures

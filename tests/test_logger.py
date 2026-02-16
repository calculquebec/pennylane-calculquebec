import importlib
import os
import logging
from unittest.mock import patch
import pytest
from pennylane_calculquebec import logger as logger_module


@pytest.fixture
def clean_logger():
    """Fixture to ensure logger is in a clean state and handlers are closed."""
    # Close existing handlers to release file locks
    for h in logger_module.logger.handlers[:]:
        logger_module.logger.removeHandler(h)
        h.close()
    yield
    # Cleanup after test
    for h in logger_module.logger.handlers[:]:
        logger_module.logger.removeHandler(h)
        h.close()


def test_logger_write_permission_denied(clean_logger, caplog):
    """Test that logger falls back to StreamHandler if FileHandler fails due to permissions."""
    with patch("logging.FileHandler", side_effect=PermissionError("Permission denied")):
        importlib.reload(logger_module)

    handlers = logger_module.logger.handlers
    # Check if a StreamHandler is present (and it's not a FileHandler which is a subclass)
    assert any(type(h) is logging.StreamHandler for h in handlers)
    assert "Unable to open log file" in caplog.text
    assert "PermissionError" in caplog.text


def test_logger_file_not_found(clean_logger, caplog):
    """Test that logger falls back to StreamHandler if FileHandler fails due to file not found."""
    with patch("logging.FileHandler", side_effect=FileNotFoundError("File not found")):
        importlib.reload(logger_module)

    handlers = logger_module.logger.handlers
    # Check if a StreamHandler is present (and it's not a FileHandler which is a subclass)
    assert any(type(h) is logging.StreamHandler for h in handlers)
    assert "Unable to open log file" in caplog.text
    assert "FileNotFoundError" in caplog.text


def test_logger_create_logfile_in_set_env_var(clean_logger, tmp_path):
    """Test that logger uses the path specified in PLCQ_LOG_PATH environment variable."""
    log_path = str(tmp_path / "custom_env.log")

    with patch.dict(os.environ, {"PLCQ_LOG_PATH": log_path}):
        importlib.reload(logger_module)

    logger_module.logger.info("test message env var")

    assert os.path.exists(log_path)
    with open(log_path, "r", encoding="utf-8") as f:
        assert "test message env var" in f.read()


def test_logger_create_logfile_in_cwd(clean_logger, tmp_path, monkeypatch):
    """Test that logger defaults to creating a log file in the current working directory."""
    # Use monkeypatch to safely change CWD and environment
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PLCQ_LOG_PATH", raising=False)

    importlib.reload(logger_module)

    expected_path = os.path.join(os.getcwd(), "pennylane_calculquebec.log")
    logger_module.logger.info("test message cwd")

    assert os.path.exists(expected_path)
    with open(expected_path, "r", encoding="utf-8") as f:
        assert "test message cwd" in f.read()

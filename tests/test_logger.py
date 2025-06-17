import os
import tempfile
import logging
from pennylane_calculquebec.logger import logger  # Replace with actual import


def test_logger_creates_virtual_folder_and_logs_message(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = os.path.join(tmp_dir, "virtual", "pennylane_calculquebec.log")

        # Patch environment variable
        monkeypatch.setenv("PLCQ_LOG_PATH", log_path)

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Remove any existing handlers
        for h in logger.handlers[:]:
            logger.removeHandler(h)
            h.close()  # <-- Close existing handlers

        # Set up new FileHandler
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        formatter = logging.Formatter("Test | %(asctime)s | Incident: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Log test message
        test_message = "Simulated error"
        logger.error(test_message)

        # Verify file created and message written
        assert os.path.exists(log_path)
        with open(log_path, "r", encoding="utf-8") as f:
            contents = f.read()
            assert test_message in contents

        # Explicitly close the handler to release the file lock
        logger.removeHandler(handler)
        handler.close()

import os
import sys
import tempfile
import importlib
import logging

def test_logger_writes_log(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "test.log")
        monkeypatch.setenv("PLCQ_LOG_PATH", log_path)

        # make sure the logger starts with no handlers
        log = logging.getLogger("pennylane_calculquebec")
        for h in log.handlers[:]:
            log.removeHandler(h)
            h.close()

        sys.modules.pop("pennylane_calculquebec.logger", None)
        logger_mod = importlib.import_module("pennylane_calculquebec.logger")
        logger = logger_mod.logger

        # ---- write -----------------------------------------------------
        test_message = "This is a test log message"
        logger.info(test_message)

        # ---- read & assert --------------------------------------------
        # (the file can be read even while it is still open for writing)
        with open(log_path, encoding="utf-8") as f:
            contents = f.read()

        assert test_message in contents
        assert f"The plugin version is {logger_mod.__version__}" in contents
        assert "Incident:" in contents

        # ---- now close the handlers so Windows can delete the file ----
        for h in logger.handlers[:]:
            logger.removeHandler(h)
            h.close()
        # or simply:
        # logging.shutdown()

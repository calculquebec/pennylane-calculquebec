import logging
import os
from pennylane_calculquebec._version import __version__

DEFAULT_LOG_PATH = os.path.join(
    os.getcwd(),
    "pennylane_calculquebec.log",
)
LOG_PATH = os.environ.get(
    "PLCQ_LOG_PATH",
    DEFAULT_LOG_PATH,
)

logger = logging.getLogger("pennylane_calculquebec")
logger.setLevel(logging.INFO)

try:
    handler = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")

except OSError as e:
    logging.warning(
        "Unable to open log file '%s' for writing due to a %s: %s. "
        "Falling back to console logging (StreamHandler).",
        LOG_PATH,
        type(e).__name__,
        e,
    )
    handler = logging.StreamHandler()

formatter = logging.Formatter(
    f"%(asctime)s [%(levelname)s] Version {__version__} | Message: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)

import logging
import os
from ._version import __version__
# Get the root directory (one level up from this file's directory)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(ROOT_DIR, "pennylane_calculquebec.log")

logger = logging.getLogger("pennylane_calculquebec")
logger.setLevel(logging.INFO)

handler = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
formatter = logging.Formatter(
    f'The plugin version is {__version__} | %(asctime)s | Incident: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)


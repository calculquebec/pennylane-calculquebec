"""Plugin-wide logger configuration for the PennyLane Calcul Québec plugin.

Configures a named :class:`logging.Logger` that writes timestamped entries
(including the plugin version) to a log file.  The log file path defaults to
``pennylane_calculquebec.log`` at the repository root but can be overridden
with the ``PLCQ_LOG_PATH`` environment variable.
"""

import logging
import os
from pennylane_calculquebec._version import __version__

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.environ.get(
    "PLCQ_LOG_PATH", os.path.join(ROOT_DIR, "pennylane_calculquebec.log")
)

logger = logging.getLogger("pennylane_calculquebec")
logger.setLevel(logging.INFO)

handler = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
formatter = logging.Formatter(
    f"%(asctime)s - The plugin version is {__version__} | Incident: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)

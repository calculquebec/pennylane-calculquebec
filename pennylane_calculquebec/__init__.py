"""Top-level package for the PennyLane plugin for Calcul Québec's MonarQ quantum computer.

Exposes the three main device classes:

* :class:`~pennylane_calculquebec.MonarqDevice` — executes circuits on MonarQ hardware (``monarq.default``)
* :class:`~pennylane_calculquebec.MonarqSim` — simulates MonarQ execution locally via ``default.mixed`` (``monarq.sim``)
* :class:`~pennylane_calculquebec.MonarqBackup` — targets the backup MonarQ machine (``monarq.backup``)
"""

import importlib.util


from pennylane_calculquebec.monarq_device import MonarqDevice
from pennylane_calculquebec.monarq_sim import MonarqSim
from pennylane_calculquebec.monarq_backup import MonarqBackup

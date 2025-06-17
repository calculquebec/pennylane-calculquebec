"""
Contains the Device implementation of monarq.default
"""

from pennylane.tape import QuantumTape
from pennylane_calculquebec.processing import PostProcessor
from pennylane_calculquebec.processing.config import (
    ProcessingConfig,
    MonarqDefaultConfig,
)
from pennylane_calculquebec.API.client import ApiClient
from pennylane_calculquebec.API.job import Job
from pennylane_calculquebec.device_exception import DeviceException
from pennylane_calculquebec.monarq_device import MonarqDevice


class MonarqBackup(MonarqDevice):
    """Backup device for interfacing with Anyon's quantum Hardware.

    * Extends the PennyLane :class:`~.pennylane.Device` class.
    * Batching is not supported yet.

    Args:
        wires (int, Iterable[Number, str]): Number of wires present on the device, or iterable that
            contains unique labels for the wires as numbers (i.e., ``[-1, 0, 2]``) or strings
            (``['ancilla', 'q1', 'q2']``). Default ``None`` if not specified.
        shots (int, Sequence[int], Sequence[Union[int, Sequence[int]]]): The default number of shots
            to use in executions involving this device.
        client (Client) : client information for connecting to MonarQ
        behaviour_config (Config) : behaviour changes to apply to the transpiler
    """

    name = "MonarqBackup"
    short_name = "monarq.backup"

    def __init__(self, wires=None, shots=None, client=None, processing_config=None):
        super().__init__(wires, shots, client, processing_config)

    @property
    def machine_name(self):
        return "yukon"

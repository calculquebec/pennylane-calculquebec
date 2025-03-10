"""
Contains the Device implementation of monarq.default
"""

from pennylane.tape import QuantumTape
from pennylane_calculquebec.processing import PostProcessor
from pennylane_calculquebec.processing.config import ProcessingConfig, MonarqDefaultConfig
from pennylane_calculquebec.API.client import ApiClient
from pennylane_calculquebec.API.job import Job
from pennylane_calculquebec.device_exception import DeviceException
from pennylane_calculquebec.base_device import BaseDevice

class MonarqDevice(BaseDevice):
    """PennyLane device for interfacing with Anyon's quantum Hardware.

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

    name = "MonarqDevice"
    short_name = "monarq.default"

    def __init__(self, 
                 wires = None, 
                 shots = None,  
                 client : ApiClient = None,
                 processing_config : ProcessingConfig = None) -> None:

        if processing_config is None:
            processing_config = MonarqDefaultConfig(self.machine_name)
        
        super().__init__(wires, shots, client, processing_config)

        if isinstance(shots, int) and (shots < 1 or shots > 1000) or isinstance(shots, list) and (len(shots) < 1 or len(shots) > 1000) or shots == None:
            raise DeviceException("The number of shots must be contained between 1 and 1000")
        
        if client is None:
            raise DeviceException("The client has not been defined. Cannot establish connection with MonarQ.")

    
    @property
    def machine_name(self):
        return "yamaska"

    @property
    def name(self):
        return MonarqDevice.short_name
   
    def _measure(self, tape : QuantumTape):
        """
        runs job to Monarq and returns value, converted to required measurement type

        Args : 
            tape (QuantumTape) : the tape from which to get results
        
        Returns :
            a result, which format can change according to the measurement process
        """
        if len(tape.measurements) != 1:
            raise DeviceException("Multiple measurements not supported")
        meas = type(tape.measurements[0]).__name__

        if not any(meas == measurement for measurement in MonarqDevice.measurement_methods.keys()):
            raise DeviceException("Measurement not supported")

        results = Job(tape, self.machine_name).run()
        results = PostProcessor.get_processor(self._processing_config, self.wires)(tape, results)
        measurement_method = MonarqDevice.measurement_methods[meas]

        return measurement_method(results)
    
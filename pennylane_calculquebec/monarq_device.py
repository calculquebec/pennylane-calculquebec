"""
Contains the Device implementation of monarq.default
"""

from typing import Tuple
from pennylane.devices import Device
from pennylane.transforms import transform
from pennylane.transforms.core import TransformProgram
from pennylane.tape import QuantumScript, QuantumTape
from pennylane_calculquebec.execution_config import DefaultExecutionConfig, ExecutionConfig
from pennylane_calculquebec.API.adapter import ApiAdapter
from pennylane_calculquebec.processing import PreProcessor, PostProcessor
from pennylane_calculquebec.processing.config import ProcessingConfig, MonarqDefaultConfig
from pennylane_calculquebec.API.client import ApiClient
from pennylane_calculquebec.API.job import Job
import pennylane.measurements as measurements


class DeviceException(Exception):
    def __init__(self, message):
        super().__init__(message)
        

class MonarqDevice(Device):
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

    name = "CalculQCDevice"
    short_name = "monarq.default"
    pennylane_requires = ">=0.30.0"
    author = "CalculQuebec"
    
    realm = "calculqc"
    circuit_name = "test circuit"
    project_id = ""
    machine_name = "yamaska"

    observables = {
        "PauliZ"
    }
    
    _processing_config : ProcessingConfig
    
    def __init__(self, 
                 wires = None, 
                 shots = None,  
                 client : ApiClient = None,
                 processing_config : ProcessingConfig = None) -> None:

        if isinstance(shots, int) and (shots < 1 or shots > 1000) or isinstance(shots, list) and (len(shots) < 1 or len(shots) > 1000) or shots == None:
            raise DeviceException("The number of shots must be contained between 1 and 1000")
            
        super().__init__(wires=wires, shots=shots)
        if client is None:
            raise DeviceException("The client has not been defined. Cannot establish connection with MonarQ.")
        
        self.client = client
        if not processing_config:
            processing_config = MonarqDefaultConfig()
        
        ApiAdapter.initialize(client)
        
        self._processing_config = processing_config
    

    @property
    def name(self):
        return MonarqDevice.short_name
    
    
    def preprocess(
        self,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Tuple[TransformProgram, ExecutionConfig]:
        """This function defines the device transfrom program to be applied and an updated execution config.

        Args:
            execution_config (Union[ExecutionConfig, Sequence[ExecutionConfig]]): A data structure describing the
            parameters needed to fully describe the execution.

        Returns:
            TransformProgram: A transform program that when called returns QuantumTapes that the device
            can natively execute.
            ExecutionConfig: A configuration with unset specifications filled in.
        """
        config = execution_config

        transform_program = TransformProgram()
        processor = PreProcessor.get_processor(self._processing_config, self.wires)
        transform_program.add_transform(transform=transform(processor))
        return transform_program, config


    def execute(self, circuits: QuantumTape | list[QuantumTape], execution_config : ExecutionConfig = DefaultExecutionConfig):
        """
        This function runs provided quantum circuit on MonarQ
        A job is first created, and then ran. 
        Results are then post-processed and returned to the user.
        """
        is_single_circuit : bool = isinstance(circuits, QuantumScript)
        if is_single_circuit:
            circuits = [circuits]
        
         # Check if execution_config is an instance of ExecutionConfig
        if isinstance(execution_config, ExecutionConfig):
            interface = (
                execution_config.interface
                if execution_config.gradient_method in {"backprop", None}
                else None
            )
        else:
            # Fallback or default behavior if execution_config is not an instance of ExecutionConfig
            interface = None
        
        results = [self._measure(tape) for tape in circuits]
        post_processed_results = [PostProcessor.get_processor(self._processing_config, self.wires)(circuits[i], res) for i, res in enumerate(results)]
       
        return post_processed_results if not is_single_circuit else post_processed_results[0]


    def _measure(self, tape : QuantumTape):
        meas = set([type(measure).__name__ for measure in tape.measurements])
        if len(meas) != 1:
            raise DeviceException("Multiple measurements not supported")
        meas = tape.measurements[0]
        if isinstance(meas, measurements.CountsMP):
            return Job(tape).run()
        else:
            raise DeviceException("Measurement process " + type(meas).__name__ + " is not supported by this device.")
        
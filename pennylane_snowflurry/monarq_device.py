from typing import Tuple
from pennylane.devices import Device
from pennylane.transforms.core import TransformProgram
from pennylane.tape import QuantumScript, QuantumTape
from pennylane_snowflurry.execution_config import DefaultExecutionConfig, ExecutionConfig
from pennylane_snowflurry.API.api_adapter import ApiAdapter
from pennylane_snowflurry.transpiler.monarq_transpile import Transpiler
from pennylane_snowflurry.transpiler.transpiler_config import TranspilerConfig, MonarqDefaultConfig
from pennylane_snowflurry.API.api_client import ApiClient
from pennylane_snowflurry.measurements.monarq_device.counts import Counts as MonarqCounts
from pennylane_snowflurry.measurements.monarq_device.sample import Sample as MonarqSample
import pennylane.measurements as measurements

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
    
    _behaviour_config : TranspilerConfig
    
    def __init__(self, 
                 wires = None, 
                 shots = None,  
                 client : ApiClient = None,
                 behaviour_config : TranspilerConfig = None) -> None:

        super().__init__(wires=wires, shots=shots)
        
        if not client:
            raise Exception("The client has not been defined")
        
        if not behaviour_config:
            behaviour_config = MonarqDefaultConfig()
        
        ApiAdapter.initialize(client)
        
        self._behaviour_config = behaviour_config
    

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
        transform_program.add_transform(Transpiler.get_transpiler(self._behaviour_config))
        return transform_program, config

    def execute(self, circuits: QuantumTape | list[QuantumTape], execution_config : ExecutionConfig = DefaultExecutionConfig):
        """
        This function runs provided quantum circuit on MonarQ
        A job is first created, and then ran. Results are returned to the user.
        """
        is_single_circuit : bool = isinstance(circuits, QuantumScript)
        if is_single_circuit:
            circuits = [circuits]
        
        if self.tracker.active:
            for c in circuits:
                self.tracker.update(resources=c.specs["resources"])
            self.tracker.update(batches=1, executions=len(circuits))
            self.tracker.record()

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
        return results if not is_single_circuit else results[0]

    def _measure(self, tape : QuantumTape):
        meas = set([type(mp).__name__ for mp in tape.measurements])
        if len(meas) != 1:
            raise Exception("multiple measurement types not supported yet")
        meas = tape.measurements[0]
        if isinstance(meas, measurements.CountsMP):
            return MonarqCounts().measure(tape)
        elif isinstance(meas, measurements.SampleMP):
            return MonarqSample().measure(tape)
        else:
            raise Exception("Measurement process " + type(meas).__name__ + " is not supported by this device.")
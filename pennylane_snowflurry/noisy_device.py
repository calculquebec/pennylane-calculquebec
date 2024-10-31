import pennylane as qml
from pennylane_snowflurry.execution_config import ExecutionConfig, DefaultExecutionConfig
from typing import Tuple
from pennylane.transforms.core import TransformProgram
from pennylane.tape import QuantumTape, QuantumScript
from pennylane_snowflurry.transpiler.monarq_transpile import Transpiler
from pennylane_snowflurry.API.api_utility import instructions
import pennylane_snowflurry.transpiler.transpiler_enums as enums
from pennylane.devices import Device
from pennylane_snowflurry.utility.debug_utility import add_noise

class NoisyDevice(Device):
    name = "MonarQDevice"
    short_name = "monarq.qubit"
    pennylane_requires = ">=0.30.0"
    author = "CalculQuébec"
    
    realm = "calculqc"
    circuit_name = "test circuit"
    project_id = ""
    machine_name = "yamaska"
    
    operations = {
        key for key in instructions.keys()
    }
    
    observables = {
        "PauliZ"
    }
    
    @property
    def name(self):
        return NoisyDevice.short_name
    
    def __init__(self, wires=None, shots=None, noise = 0.005):
        super().__init__(wires=wires, shots=shots)
        self._noise = noise
        
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
        transform_program.add_transform(qml.transform(add_noise))
        return transform_program, config

    def execute(self, circuits: QuantumTape | list[QuantumTape], execution_config : ExecutionConfig = DefaultExecutionConfig):
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
        
        results = [qml.execute([circuit], qml.device("default.mixed", wires = circuit.wires, shots = circuit.shots.total_shots)) for circuit in circuits]

        return results if not is_single_circuit else results[0]
import pennylane as qml
from pennylane_snowflurry.execution_config import ExecutionConfig, DefaultExecutionConfig
from typing import Tuple
from pennylane.transforms.core import TransformProgram
from pennylane.tape import QuantumTape, QuantumScript
from pennylane_snowflurry.utility.api import instructions
from pennylane.devices import Device
from pennylane_snowflurry.utility.debug import add_noise

class NoisyDevice(Device):
    """
    a device that wraps around default.mixed device, and adds a depolarizing channel for each operation. 
    useful for simulation noise on a circuit
    """
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
        transform_program.add_transform(qml.transform(lambda tape : add_noise(tape, noise = self._noise)))
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
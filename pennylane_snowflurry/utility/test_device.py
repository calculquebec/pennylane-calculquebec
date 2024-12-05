"""
Contains a wrapper around default.qubit which uses MonarQ pre/post processing\n
WARNING : it is not advised to use benchmarkings with this device
"""

from typing import Tuple
import pennylane as qml
from pennylane.devices import Device
from pennylane.transforms.core import TransformProgram
from pennylane.tape import QuantumScript, QuantumTape
from pennylane_snowflurry.execution_config import DefaultExecutionConfig, ExecutionConfig
from pennylane_snowflurry.utility.api import instructions
from pennylane_snowflurry.processing.monarq_postproc import PostProcessor
from pennylane_snowflurry.processing.monarq_preproc import PreProcessor
from pennylane_snowflurry.processing.config import MonarqDefaultConfig

class TestDevice(Device):
    """a device that uses the monarq transpiler but simulates results using default.qubit
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
        return TestDevice.short_name
    
    def __init__(self, 
                 wires = None, 
                 shots = None, 
                 processing_config = None) -> None:
        super().__init__(wires=wires, shots=shots)
        
        if processing_config is None:
            processing_config = MonarqDefaultConfig(use_benchmark=False)
        
        self._processing_config = processing_config
    
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
        transform_program.add_transform(PreProcessor.get_processor(self._processing_config, self.wires))
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
        
        results = [qml.execute([circuit], qml.device("default.qubit", circuit.wires, circuit.shots)) for circuit in circuits]
        post_processed_results = [PostProcessor.get_processor(self._processing_config, self.wires)(circuits[i], res) for i, res in enumerate(results)]

        return post_processed_results if not is_single_circuit else post_processed_results[0]
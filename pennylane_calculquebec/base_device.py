"""
Contains the Device implementation of monarq.default
"""

from typing import Tuple
from pennylane.devices import Device
from pennylane.transforms import transform
from pennylane.transforms.core import TransformProgram
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.devices import ExecutionConfig
from pennylane_calculquebec.API.adapter import ApiAdapter
from pennylane_calculquebec.processing import PreProcessor, PostProcessor
from pennylane_calculquebec.processing.config import (
    ProcessingConfig,
    MonarqDefaultConfig,
)
from pennylane_calculquebec.API.client import ApiClient
from pennylane_calculquebec.API.job import Job
from pennylane_calculquebec.utility.debug import counts_to_probs, compute_expval
import pennylane.measurements as measurements
from pennylane_calculquebec.device_exception import DeviceException


class BaseDevice(Device):
    pennylane_requires = ">=0.36.0"
    author = "CalculQuebec"

    realm = "calculqc"

    observables = {"PauliZ"}
    measurement_methods: dict = {
        "CountsMP": lambda counts: counts,
        "ProbabilityMP": counts_to_probs,
        "ExpectationMP": compute_expval,
    }

    _client: ApiClient
    _processing_config: ProcessingConfig

    @property
    def processing_config(self):
        return self._processing_config

    def __init__(self, wires=None, shots=None, client=None, processing_config=None):
        super().__init__(wires, shots)
        self._circuit_name = None
        self._project_name = None
        self._processing_config = processing_config

        if client is not None:
            self._client = client
            self._client.machine_name = self.machine_name
            ApiAdapter.initialize(self._client)

    def preprocess(
        self,
        execution_config=ExecutionConfig,
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

    def execute(
        self,
        circuits: QuantumTape | list[QuantumTape],
        execution_config=ExecutionConfig,
    ):
        """
        This function runs provided quantum circuit on MonarQ
        A job is first created, and then ran.
        Results are then post-processed and returned to the user.
        """
        is_single_circuit: bool = isinstance(circuits, QuantumScript)
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
        return results if not is_single_circuit else results[0]

    @property
    def machine_name(self):
        raise NotImplementedError()

    def _measure(self, tape: QuantumTape):
        raise NotImplementedError()

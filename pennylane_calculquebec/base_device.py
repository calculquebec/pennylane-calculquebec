"""Abstract base device shared by all MonarQ PennyLane device implementations.

Defines the common preprocessing/execution interface and the supported
measurement types (counts, probabilities, expectation value).  Concrete
subclasses (:class:`~pennylane_calculquebec.MonarqDevice`,
:class:`~pennylane_calculquebec.MonarqSim`,
:class:`~pennylane_calculquebec.monarq_backup.MonarqBackup`) must implement
:attr:`machine_name` and :meth:`_measure`.
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
    """Abstract base class for MonarQ-compatible PennyLane devices.

    Subclasses must implement :attr:`machine_name` and :meth:`_measure`.
    Concrete implementations are :class:`~pennylane_calculquebec.MonarqDevice`,
    :class:`~pennylane_calculquebec.MonarqSim`, and
    :class:`~pennylane_calculquebec.monarq_backup.MonarqBackup`.

    Supported measurements: ``CountsMP``, ``ProbabilityMP``, ``ExpectationMP``.
    """

    pennylane_requires = ">=0.36.0"
    author = "CalculQuebec"

    realm = "calculqc"

    observables = {"PauliZ"}
    measurement_methods: dict = {
        "CountsMP": lambda counts: counts,
        "ProbabilityMP": counts_to_probs,
        "ExpectationMP": compute_expval,
    }
    """dict: Mapping from measurement class name to the post-processing callable
    that converts raw counts into the expected output format."""

    _client: ApiClient
    _processing_config: ProcessingConfig

    @property
    def processing_config(self):
        """Return the active processing configuration.

        Returns:
            ProcessingConfig: the pre- and post-processing pipeline configuration
        """
        return self._processing_config

    def __init__(self, wires=None, shots=None, client=None, processing_config=None):
        """Initialize the base device and optionally connect to the API.

        Args:
            wires (int or Iterable): number of wires or iterable of wire labels.
                Defaults to ``None``.
            shots (int or Sequence[int]): default number of shots per execution.
                Defaults to ``None``.
            client (ApiClient, optional): credentials used to authenticate with
                the Thunderhead API. When ``None``, no API connection is established
                (useful for simulation-only use cases).
            processing_config (ProcessingConfig, optional): custom transpilation
                pipeline. When ``None``, the subclass is expected to provide a
                default configuration.
        """
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
        """Build the PennyLane transform program that preprocesses circuits before execution.

        The transform program applies all pre-processing steps defined in
        :attr:`processing_config` (decomposition, placement, routing, optimisation, …).

        Args:
            execution_config (ExecutionConfig): parameters describing the execution.
                Defaults to ``ExecutionConfig``.

        Returns:
            tuple[TransformProgram, ExecutionConfig]: the transform program and the
            (potentially updated) execution config.
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
        """Execute one or more pre-processed quantum circuits.

        Iterates over the provided tapes and delegates each one to
        :meth:`_measure`.  A single tape is returned as a scalar result;
        a list of tapes is returned as a list.

        Args:
            circuits (QuantumTape or list[QuantumTape]): the circuit(s) to execute
            execution_config (ExecutionConfig): execution parameters.
                Defaults to ``ExecutionConfig``.

        Returns:
            any or list[any]: measurement result(s) in the format determined by the
            measurement type (counts, probabilities, or expectation value)
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
        """Name of the target quantum hardware.

        Returns:
            str: identifier of the machine used for job submission

        Raises:
            NotImplementedError: must be overridden by concrete subclasses
        """
        raise NotImplementedError()

    def _measure(self, tape: QuantumTape):
        """Execute a single tape and return measurement results.

        Args:
            tape (QuantumTape): the circuit to execute

        Returns:
            any: measurement results in the format determined by the measurement type

        Raises:
            NotImplementedError: must be overridden by concrete subclasses
        """
        raise NotImplementedError()

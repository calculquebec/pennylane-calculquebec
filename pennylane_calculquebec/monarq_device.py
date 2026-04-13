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
from pennylane_calculquebec.base_device import BaseDevice
from typing import Callable
from pennylane_calculquebec.logger import logger


class MonarqDevice(BaseDevice):
    """PennyLane device for executing circuits on MonarQ quantum hardware.

    * Extends :class:`~pennylane_calculquebec.base_device.BaseDevice`.
    * Batching is not supported.
    * Shots must be between 1 and 1000 (inclusive).

    Args:
        wires (int or Iterable[Number, str]): number of wires or iterable of
            unique wire labels (e.g. ``[-1, 0, 2]`` or ``['ancilla', 'q1']``).
            Defaults to ``None``.
        shots (int or Sequence[int]): default number of shots per execution.
            Must be in the range ``[1, 1000]``.
        client (ApiClient): credentials for connecting to MonarQ.
            Required — raises :class:`~pennylane_calculquebec.device_exception.DeviceException`
            when ``None``.
        processing_config (ProcessingConfig, optional): transpilation pipeline
            configuration.  Defaults to :func:`~pennylane_calculquebec.processing.config.MonarqDefaultConfig`.

    **Callbacks**

    Three optional callbacks can be attached to monitor job lifecycle events:

    * ``job_started(job_id: int)`` — called when the job is submitted
    * ``job_status_changed(job_id: int, status: str)`` — called each time the
      job status changes while polling
    * ``job_completed(job_id: int)`` — called when the job reaches a terminal state
    """

    name = "MonarqDevice"
    short_name = "monarq.default"

    job_started: Callable[[int], None]
    job_status_changed: Callable[[int, str], None]
    job_completed: Callable[[int], None]

    def __init__(
        self,
        wires=None,
        shots=None,
        client: ApiClient = None,
        processing_config: ProcessingConfig = None,
    ) -> None:
        self.job_started = None
        self.job_status_changed = None
        self.job_completed = None

        if processing_config is None:
            processing_config = MonarqDefaultConfig(self.machine_name)

        super().__init__(wires, shots, client, processing_config)

        if (
            isinstance(shots, int)
            and (shots < 1 or shots > 1000)
            or isinstance(shots, list)
            and (len(shots) < 1 or len(shots) > 1000)
        ):
            raise DeviceException(
                "The number of shots must be contained between 1 and 1000"
            )

        if client is None:
            raise DeviceException(
                "The client has not been defined. Cannot establish connection with MonarQ."
            )

    @property
    def machine_name(self):
        """Name of the primary MonarQ machine targeted by this device.

        Returns:
            str: ``"yamaska"``
        """
        try:
            return "yamaska"
        except Exception as e:
            logger.error(
                "Error %s in machine_name located in MonarqDevice: %s",
                type(e).__name__,
                e,
            )
            return None

    @property
    def name(self):
        """Short identifier of this device as registered with PennyLane.

        Returns:
            str: ``"monarq.default"``
        """
        try:
            return MonarqDevice.short_name
        except Exception as e:
            logger.error(
                "Error %s in name located in MonarqDevice: %s", type(e).__name__, e
            )
            return None

    def _measure(self, tape: QuantumTape):
        """Submit a circuit job to MonarQ and return post-processed results.

        Submits ``tape`` as a job via :class:`~pennylane_calculquebec.API.job.Job`,
        applies the configured post-processing pipeline, then converts the raw
        counts to the measurement type requested by the tape (counts, probabilities,
        or expectation value).

        Args:
            tape (QuantumTape): the pre-processed circuit to execute

        Returns:
            any: result in the format determined by the measurement type
                (``dict`` for counts, ``numpy.ndarray`` for probabilities,
                ``float`` for expectation value)

        Raises:
            DeviceException: if the tape contains more than one measurement
            DeviceException: if the measurement type is not supported
        """
        if len(tape.measurements) != 1:
            raise DeviceException("Multiple measurements not supported")
        meas = type(tape.measurements[0]).__name__

        if not any(
            meas == measurement
            for measurement in MonarqDevice.measurement_methods.keys()
        ):
            raise DeviceException("Measurement not supported")

        job = Job(tape)
        job.started = self.job_started
        job.status_changed = self.job_status_changed
        job.completed = self.job_completed
        results = job.run()

        results = PostProcessor.get_processor(self._processing_config, self.wires)(
            tape, results
        )
        measurement_method = MonarqDevice.measurement_methods[meas]

        return measurement_method(results)

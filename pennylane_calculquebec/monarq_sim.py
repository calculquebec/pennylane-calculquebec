"""
Contains a wrapper around default.mixed which uses MonarQ pre/post processing.
"""

import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane_calculquebec.processing.monarq_postproc import PostProcessor
from pennylane_calculquebec.processing.config import MonarqDefaultConfig
from pennylane.measurements import CountsMP
from pennylane_calculquebec.device_exception import DeviceException
from pennylane_calculquebec.base_device import BaseDevice
from pennylane_calculquebec.processing.steps import (
    GateNoiseSimulation,
    ReadoutNoiseSimulation,
)
from pennylane_calculquebec.logger import logger


class MonarqSim(BaseDevice):
    """PennyLane device that simulates MonarQ execution using ``default.mixed``.

    Applies the full MonarQ transpilation pipeline (decomposition, placement,
    routing, optimisation) and then simulates the resulting native-gate circuit
    with ``default.mixed``, optionally injecting gate noise and readout noise
    derived from the latest hardware benchmark.

    When a ``client`` is provided, noise parameters are fetched from the live
    benchmark data.  Without a client, typical noise values are used instead.

    Args:
        wires (int or Iterable): number of wires or iterable of wire labels.
            Defaults to ``None``.
        shots (int): number of shots used during simulation. Defaults to ``None``.
        client (ApiClient, optional): when provided, enables benchmark-based noise
            simulation.  Defaults to ``None``.
        processing_config (ProcessingConfig, optional): custom transpilation
            pipeline configuration.  Defaults to
            :func:`~pennylane_calculquebec.processing.config.MonarqDefaultConfig`.
    """

    name = "MonarqSim"
    short_name = "monarq.sim"

    @property
    def name(self):
        """Short identifier of this device as registered with PennyLane.

        Returns:
            str: ``"monarq.sim"``
        """
        try:
            return MonarqSim.short_name
        except Exception as e:
            logger.error(
                "Error %s in name located in MonarqSim: %s", type(e).__name__, e
            )
            return None

    def __init__(self, wires=None, shots=None, client=None, processing_config=None):
        try:
            use_benchmark = client is not None

            if processing_config is None:
                processing_config = MonarqDefaultConfig(
                    self.machine_name, use_benchmark
                )

            super().__init__(wires, shots, client, processing_config)
            self.use_benchmark_for_simulation = use_benchmark
        except Exception as e:
            logger.error(
                "Error %s in __init__ located in MonarqSim: %s", type(e).__name__, e
            )

    def _measure(self, tape: QuantumTape):
        """Simulate a circuit on ``default.mixed`` and return post-processed results.

        Execution pipeline:

        1. Rebuild the tape with :class:`~pennylane.measurements.CountsMP` measurements
           and 1000 shots for internal simulation.
        2. Inject gate noise via
           :class:`~pennylane_calculquebec.processing.steps.GateNoiseSimulation`.
        3. Execute on ``default.mixed``.
        4. Apply readout noise via
           :class:`~pennylane_calculquebec.processing.steps.ReadoutNoiseSimulation`.
        5. Apply the post-processing pipeline (e.g. readout error mitigation).
        6. Convert the counts to the measurement type requested by the original tape.

        Args:
            tape (QuantumTape): the pre-processed circuit to simulate

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
            meas == measurement for measurement in MonarqSim.measurement_methods.keys()
        ):
            raise DeviceException("Measurement not supported")

        # simulate counts from given circuit on default mixed
        counts_tape = type(tape)(
            ops=tape.operations,
            measurements=[CountsMP(wires=mp.wires) for mp in tape.measurements],
            shots=1000,
        )

        sim_tape = GateNoiseSimulation(
            self.machine_name, self.use_benchmark_for_simulation
        ).execute(counts_tape)
        results = qml.execute(
            [sim_tape],
            qml.device("default.mixed", wires=sim_tape.wires),
        )[0]

        # apply post processing
        sim_results = ReadoutNoiseSimulation(
            self.machine_name, self.use_benchmark_for_simulation
        ).execute(counts_tape, results)
        results = PostProcessor.get_processor(self._processing_config, self.wires)(
            counts_tape, sim_results
        )

        # return desired measurement method
        measurement_method = MonarqSim.measurement_methods[meas]
        return measurement_method(results)

    @property
    def machine_name(self):
        """Name of the MonarQ machine whose topology is used for simulation.

        Returns:
            str: ``"yamaska"``
        """
        try:
            return "yamaska"
        except Exception as e:
            logger.error(
                "Error %s in machine_name located in MonarqSim: %s", type(e).__name__, e
            )
            return None

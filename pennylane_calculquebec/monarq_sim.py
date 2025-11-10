"""
Contains a wrapper around default.mixed which uses MonarQ pre/post processing\n
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
    """
    a device that uses the monarq transpiler but simulates results using default.mixed
    """

    name = "MonarqSim"
    short_name = "monarq.sim"

    @property
    def name(self):
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
        """
        simulates job to Monarq and returns value, converted to required measurement type

        Args :
            tape (QuantumTape) : the tape from which to get results

        Returns :
            a result, which format can change according to the measurement process
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
        try:
            return "yamaska"
        except Exception as e:
            logger.error(
                "Error %s in machine_name located in MonarqSim: %s", type(e).__name__, e
            )
            return None

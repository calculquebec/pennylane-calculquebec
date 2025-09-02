"""
Contains a pre-processing step for adding noise relative to MonarQ's noise model.
"""

from pennylane_calculquebec.processing.interfaces import PreProcStep
import pennylane_calculquebec.monarq_data as data
from pennylane_calculquebec.utility.noise import (
    TypicalBenchmark,
    amplitude_damping,
    phase_damping,
    depolarizing_noise,
)
import pennylane as qml
from pennylane_calculquebec.utility.api import keys
from pennylane_calculquebec.logger import logger


class GateNoiseSimulation(PreProcStep):
    """
    Adds gate noise to operations from a circuit using MonarQ's noise model
    """

    def __init__(self, machine_name: str, use_benchmark=True):
        self.use_benchmark = use_benchmark
        self.machine_name = machine_name

    @property
    def native_gates(self):
        """
        the set of monarq-native gates

        Returns:
            list[str] : the name of gates that are native to MonarQ
        """
        return data.monarq_native_gates()

    def execute(self, tape):
        # build qubit noise from readout 1 fidelity using typical value if benchmark should not be used
        connectivity = data.get_connectivity(self.machine_name, self.use_benchmark)
        qubit_count = len(set([a for b in connectivity.values() for a in b]))
        coupler_count = len(connectivity)

        qubit_noise = (
            data.get_qubit_noise(self.machine_name)
            if self.use_benchmark
            else [
                depolarizing_noise(TypicalBenchmark.qubit) for _ in range(qubit_count)
            ]
        )

        # build coupler noise from cz fidelity using typical value if benchmark should not be used
        cz_noise = (
            data.get_coupler_noise(self.machine_name)
            if self.use_benchmark
            else {
                tuple(
                    data.get_connectivity(self.machine_name, False)[str(i)]
                ): depolarizing_noise(TypicalBenchmark.cz)
                for i in range(coupler_count)
            }
        )

        # build relaxation using typical t1 if not use_benchmark
        relaxation = (
            data.get_amplitude_damping(self.machine_name)
            if self.use_benchmark
            else [
                amplitude_damping(1e-6, TypicalBenchmark.t1) for _ in range(qubit_count)
            ]
        )

        # build decoherence using typical t2 if not use_benchmark
        decoherence = (
            data.get_phase_damping(self.machine_name)
            if self.use_benchmark
            else [
                phase_damping(1e-6, TypicalBenchmark.t2Ramsey)
                for _ in range(qubit_count)
            ]
        )

        operations = []

        if any(
            operation.name not in self.native_gates for operation in tape.operations
        ):
            raise ValueError(
                "Your circuit should contain only MonarQ native gates. Cannot simulate noise."
            )

        for operation in tape.operations:
            if operation.num_wires != 1:  # can only be a cz gate in this case
                operations.append(operation)
                noises = [
                    noise
                    for coupler, noise in cz_noise.items()
                    if all(wire in coupler for wire in operation.wires)
                ]
                if len(noises) < 1 or all(key is None for key in noises):
                    raise ValueError(
                        "Cannot find CZ gate noise for operation " + str(operation)
                    )

                for wire in operation.wires:
                    operations.append(qml.DepolarizingChannel(noises[0], wires=wire))
                continue

            operations.append(operation)
            for wire in operation.wires:
                operations.append(
                    qml.DepolarizingChannel(qubit_noise[wire], wires=wire)
                )

        return type(tape)(operations, tape.measurements, tape.shots)

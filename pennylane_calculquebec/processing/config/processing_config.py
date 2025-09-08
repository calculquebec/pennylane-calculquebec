"""
contains the base configuration class and presets that can be used to specify monarq.default's processing behaviour
"""

from pennylane_calculquebec.processing.interfaces.base_step import BaseStep
from pennylane_calculquebec.processing.steps import (
    DecomposeReadout,
    CliffordTDecomposition,
    VF2,
    Swaps,
    IterativeCommuteAndMerge,
    MonarqDecomposition,
    GateNoiseSimulation,
    ReadoutNoiseSimulation,
    PrintWires,
    PrintTape,
)
from typing import Callable


class ProcessingConfig:
    """
    a parameter object that can be passed to devices for changing its default behaviour

    Args
        - *args (BaseStep) : all the steps that should be used in this processing config
    """

    _steps: list[BaseStep]

    def __init__(self, *args: BaseStep):
        self._steps = []
        for arg in args:
            self._steps.append(arg)

    @property
    def steps(self):
        """
        all the steps that should be used in this processing config
        """
        return self._steps

    def __eq__(self, other: "ProcessingConfig") -> bool:
        """
        returns true if both configs have the same number of steps, and the steps are the same, in the same order, with the same configuration

        Args:
            - other (ProcessingConfig) : processing config to compare
        Returns:
            (bool) : are both processing config equal?
        """
        if len(self.steps) != len(other.steps):
            return False

        for i, step in enumerate(self.steps):
            other_step = other.steps[i]

            if type(step) != type(other_step) or vars(step) != vars(other_step):
                return False

        return True

    def __getitem__(self, idx: int) -> BaseStep:
        """returns step at index idx

        Args:
            idx (int): the index to return
        Returns:
            (BaseStep) : the step at given index
        """
        return self._steps[idx]

    def __setitem__(self, idx: int, value: BaseStep) -> None:
        """Sets the item at index idx to given value

        Args:
            idx (int): index to modify
            value : value to assign at index
        """
        self._steps[idx] = value


def MonarqDefaultConfig(
    machine_name: str,
    use_benchmark=True,
    q1_acceptance=0.5,
    q2_acceptance=0.5,
    excluded_qubits=[],
    excluded_couplers=[],
) -> ProcessingConfig:
    """
    The default configuration preset for MonarQ

    Args:
        - machine_name (str) : The name of the quantum computer (yukon or yamaska)
        - use_benchmark (bool) : should benchmarks be used for placement, routing, error mitigation, etc?
        - q1_acceptance (float) : what's the treshold at which a qubit is considered usable?
        - q2_acceptance (float) : what's the treshold at which a coupler is considered usable?
        - excluded_qubits (list[int]) : which qubits should not be used?
        - excluded_couplers (list[list[int]]) : which couplers should not be used?

    Returns:
        - (ProcessingConfig) : monarq default config
    """
    return ProcessingConfig(
        DecomposeReadout(),
        CliffordTDecomposition(),
        VF2(
            machine_name,
            use_benchmark,
            q1_acceptance,
            q2_acceptance,
            excluded_qubits,
            excluded_couplers,
        ),
        Swaps(
            machine_name,
            use_benchmark,
            q1_acceptance,
            q2_acceptance,
            excluded_qubits,
            excluded_couplers,
        ),
        IterativeCommuteAndMerge(),
        MonarqDecomposition(),
        IterativeCommuteAndMerge(),
        MonarqDecomposition(),
    )


def MonarqDefaultConfigNoBenchmark(
    machine_name: str, excluded_qubits=[], excluded_couplers=[]
) -> ProcessingConfig:
    """
    The default configuration preset, minus the benchmarking acceptance tests on qubits and couplers in the placement and routing steps.

    Args:
        - machine_name (str) : The name of the quantum computer (yukon or yamaska)
        - excluded_qubits (list[int]) : which qubits should not be used?
        - excluded_couplers (list[list[int]]) : which couplers should not be used?

    Returns:
        - (ProcessingConfig) : Monarq default config without using benchmarks
    """
    return MonarqDefaultConfig(
        machine_name,
        use_benchmark=False,
        excluded_qubits=excluded_qubits,
        excluded_couplers=excluded_couplers,
    )


def EmptyConfig() -> ProcessingConfig:
    """
    A configuration preset that you can use if you want to skip the transpiling step alltogether, and send your job to monarq as is.

    Returns:
        - (ProcessingConfig) : an empty config
    """
    return ProcessingConfig()


def NoPlaceNoRouteConfig() -> ProcessingConfig:
    """
    A configuration preset that omits placement and routing. be sure to use existing qubits and couplers

    Returns:
        - (ProcessingConfig) : monarq default config without placement and routing step
    """
    return ProcessingConfig(
        DecomposeReadout(),
        CliffordTDecomposition(),
        IterativeCommuteAndMerge(),
        MonarqDecomposition(),
        IterativeCommuteAndMerge(),
        MonarqDecomposition(),
    )


def PrintDefaultConfig(
    machine_name: str,
    only_wires=True,
    use_benchmark=True,
    q1_acceptance=0.5,
    q2_acceptance=0.5,
    excluded_qubits=[],
    excluded_couplers=[],
) -> ProcessingConfig:
    """
    The same as the default config, but it prints wires/circuit before and after transpilation

    Args:
        - machine_name (str) : The name of the quantum computer (yukon or yamaska)
        - only_wires (bool) : should we print only the wire mapping, or the whole tape?
        - use_benchmark (bool) : should benchmarks be used for placement, routing, error mitigation, etc?
        - q1_acceptance (float) : what's the treshold at which a qubit is considered usable?
        - q2_acceptance (float) : what's the treshold at which a coupler is considered usable?
        - excluded_qubits (list[int]) : which qubits should not be used?
        - excluded_couplers (list[list[int]]) : which couplers should not be used?

    Returns:
        - (ProcessingConfig) : monarq default config, with a printing preprocessing step
    """
    config = MonarqDefaultConfig(
        machine_name,
        use_benchmark,
        q1_acceptance,
        q2_acceptance,
        excluded_qubits,
        excluded_couplers,
    )
    config.steps.insert(0, PrintWires() if only_wires else PrintTape())
    config.steps.append(PrintWires() if only_wires else PrintTape())

    return config


def PrintNoPlaceNoRouteConfig(only_wires=True) -> ProcessingConfig:
    """
    The same as the NoPlaceNoRoute config, but it prints wires/circuit before and after transpilation

    Returns:
        - (ProcessingConfig) : monarq default config without placement and routing step, with an added printing step
    """
    config = NoPlaceNoRouteConfig()
    config.steps.insert(0, PrintWires() if only_wires else PrintTape())
    config.steps.append(PrintWires() if only_wires else PrintTape())
    return config


def FakeMonarqConfig(machine_name: str, use_benchmark=False) -> ProcessingConfig:
    """
    A configuration preset that does the same thing as the default config, but adds gate and readout noise at the end. This config is deprecated. Use MonarqDefaultConfig instead, or use no config at all.

    Args:
        machine_name (str) : the name of the machine
        use_benchmark (bool) : should benchmarks be used for placement, routing and mitigation?

    Returns:
        ProcessingConfig : MonarqDefaultConfig with gate and readout error simulation
    """
    import warnings

    warnings.warn(
        "FakeMonarqConfig is deprecated. Use MonarqDefaultConfig instead, or use no config at all",
        DeprecationWarning,
        stacklevel=2,
    )
    return ProcessingConfig(
        DecomposeReadout(),
        CliffordTDecomposition(),
        VF2(machine_name, use_benchmark),
        Swaps(machine_name, use_benchmark),
        IterativeCommuteAndMerge(),
        MonarqDecomposition(),
        IterativeCommuteAndMerge(),
        MonarqDecomposition(),
        GateNoiseSimulation(use_benchmark),
        ReadoutNoiseSimulation(use_benchmark),
    )

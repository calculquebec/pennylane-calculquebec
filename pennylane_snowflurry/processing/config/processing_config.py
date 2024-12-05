"""
contains the base configuration class and presets that can be used to specify monarq.default's processing behaviour
"""

from pennylane_snowflurry.processing.interfaces.base_step import BaseStep
from pennylane_snowflurry.processing.steps import DecomposeReadout, CliffordTDecomposition, ASTAR, Swaps, IterativeCommuteAndMerge, MonarqDecomposition
from typing import Callable

class ProcessingConfig:
    """a parameter object that can be passed to devices for changing its default behaviour
    """
    _steps : list[BaseStep]
    
    def __init__(self, *args : BaseStep):
        self._steps = []
        for arg in args:
            self._steps.append(arg)

    @property
    def steps(self): return self._steps


MonarqDefaultConfig : Callable[[bool, float, float, list[int], list[list[int]]], ProcessingConfig] = \
    lambda use_benchmark = True, q1_acceptance = 0.5, q2_acceptance = 0.5, excluded_qubits = [], excluded_couplers = [] : \
        ProcessingConfig(DecomposeReadout(), CliffordTDecomposition(), \
            ASTAR(use_benchmark, q1_acceptance, q2_acceptance, excluded_qubits, excluded_couplers),
            Swaps(use_benchmark, q1_acceptance, q2_acceptance, excluded_qubits, excluded_couplers), 
            IterativeCommuteAndMerge(), MonarqDecomposition(), IterativeCommuteAndMerge(), MonarqDecomposition())
"""The default configuration preset for MonarQ"""


MonarqDefaultConfigNoBenchmark = lambda q1_acceptance = 0.5, q2_acceptance = 0.5, excluded_qubits = [], excluded_couplers = [] : \
    MonarqDefaultConfig(False, q1_acceptance, q2_acceptance, excluded_qubits, excluded_couplers)
"""The default configuration preset, minus the benchmarking acceptance tests on qubits and couplers in the placement and routing steps."""

EmptyConfig = lambda : ProcessingConfig()
"""A configuration preset that you can use if you want to skip the transpiling step alltogether, and send your job to monarq as is."""

NoPlaceNoRouteConfig  = lambda : ProcessingConfig(DecomposeReadout(),
                                        CliffordTDecomposition(),
                                        IterativeCommuteAndMerge(),
                                        MonarqDecomposition())
"""A configuration preset that omits placement and routing. be sure to use existing qubits and couplers """
"""
contains the base configuration class and presets that can be used to specify monarq.default's processing behaviour
"""

from pennylane_calculquebec.processing.interfaces.base_step import BaseStep
from pennylane_calculquebec.processing.steps import DecomposeReadout, CliffordTDecomposition, ISMAGS, Swaps, IterativeCommuteAndMerge, MonarqDecomposition, GateNoiseSimulation, ReadoutNoiseSimulation
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
    
    def __eq__(self, other):
        """
        returns true if both configs have the same number of steps, and the steps are the same, in the same order, with the same configuration
        """
        if len(self.steps) != len(other.steps):
            return False
        
        for i, step in enumerate(self.steps):
            step2 = other.steps[i]
            if type(step) != type(step2) or vars(step) != vars(step):
                return False
        
        return True

    def __getitem__(self, idx):
        """returns step at index idx

        Args:
            idx (int): the index to return
        """
        return self._steps[idx]
    
    def __setitem__(self, idx, value):
        """Sets the item at index idx to given value

        Args:
            idx (int): index to modify
            value : value to assign at index
        """
        self._steps[idx] = value
        
MonarqDefaultConfig : Callable[[bool, float, float, list[int], list[list[int]]], ProcessingConfig] = \
    lambda use_benchmark = True, q1_acceptance = 0.5, q2_acceptance = 0.5, excluded_qubits = [], excluded_couplers = [] : \
        ProcessingConfig(DecomposeReadout(), CliffordTDecomposition(), \
            ISMAGS(use_benchmark, q1_acceptance, q2_acceptance, excluded_qubits, excluded_couplers),
            Swaps(use_benchmark, q1_acceptance, q2_acceptance, excluded_qubits, excluded_couplers), 
            IterativeCommuteAndMerge(), MonarqDecomposition(), IterativeCommuteAndMerge(), MonarqDecomposition())
"""The default configuration preset for MonarQ"""


MonarqDefaultConfigNoBenchmark : Callable[[list[int], list[list[int]]], ProcessingConfig]= lambda excluded_qubits = [], excluded_couplers = [] : \
    MonarqDefaultConfig(use_benchmark = False, excluded_qubits = excluded_qubits, excluded_couplers = excluded_couplers)
"""The default configuration preset, minus the benchmarking acceptance tests on qubits and couplers in the placement and routing steps."""

EmptyConfig = lambda : ProcessingConfig()
"""A configuration preset that you can use if you want to skip the transpiling step alltogether, and send your job to monarq as is."""

NoPlaceNoRouteConfig  = lambda : ProcessingConfig(DecomposeReadout(),
                                        CliffordTDecomposition(),
                                        IterativeCommuteAndMerge(),
                                        MonarqDecomposition(), 
                                        IterativeCommuteAndMerge(),
                                        MonarqDecomposition())
"""A configuration preset that omits placement and routing. be sure to use existing qubits and couplers """

FakeMonarqConfig = lambda use_benchmark = False: ProcessingConfig(DecomposeReadout(),
                                             CliffordTDecomposition(),
                                             ISMAGS(use_benchmark),
                                             Swaps(use_benchmark),
                                             IterativeCommuteAndMerge(),
                                             MonarqDecomposition(),
                                             IterativeCommuteAndMerge(),
                                             MonarqDecomposition(),
                                             GateNoiseSimulation(use_benchmark),
                                             ReadoutNoiseSimulation(use_benchmark))
"""
A configuration preset that does the same thing as the default config, but adds gate and readout noise at the end
"""
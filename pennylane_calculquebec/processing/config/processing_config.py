"""
contains the base configuration class and presets that can be used to specify monarq.default's processing behaviour
"""

from pennylane_calculquebec.processing.interfaces.base_step import BaseStep
from pennylane_calculquebec.processing.steps import DecomposeReadout, CliffordTDecomposition, VF2, Swaps, IterativeCommuteAndMerge, MonarqDecomposition, GateNoiseSimulation, ReadoutNoiseSimulation, PrintWires, PrintTape
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
            other_step = other.steps[i]
            
            if type(step) != type(other_step) or vars(step) != vars(other_step):
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
        
def MonarqDefaultConfig(machine_name : str, use_benchmark = True, q1_acceptance = 0.5, q2_acceptance = 0.5, excluded_qubits = [], excluded_couplers = []):
    """The default configuration preset for MonarQ"""
    return ProcessingConfig(DecomposeReadout(), CliffordTDecomposition(), \
            VF2(machine_name, use_benchmark, q1_acceptance, q2_acceptance, excluded_qubits, excluded_couplers),
            Swaps(machine_name, use_benchmark, q1_acceptance, q2_acceptance, excluded_qubits, excluded_couplers), 
            IterativeCommuteAndMerge(), MonarqDecomposition(), IterativeCommuteAndMerge(), MonarqDecomposition())



def MonarqDefaultConfigNoBenchmark(machine_name : str, excluded_qubits = [], excluded_couplers = []):
    """The default configuration preset, minus the benchmarking acceptance tests on qubits and couplers in the placement and routing steps."""
    return MonarqDefaultConfig(machine_name, use_benchmark = False, excluded_qubits = excluded_qubits, excluded_couplers = excluded_couplers)


def EmptyConfig(): 
    """A configuration preset that you can use if you want to skip the transpiling step alltogether, and send your job to monarq as is."""
    return ProcessingConfig()


def NoPlaceNoRouteConfig(): 
    """A configuration preset that omits placement and routing. be sure to use existing qubits and couplers """
    return ProcessingConfig(DecomposeReadout(),
                            CliffordTDecomposition(),
                            IterativeCommuteAndMerge(),
                            MonarqDecomposition(), 
                            IterativeCommuteAndMerge(),
                            MonarqDecomposition())



def PrintDefaultConfig(machine_name : str, only_wires = True, use_benchmark = True, q1_acceptance = 0.5, q2_acceptance = 0.5, excluded_qubits = [], excluded_couplers = []):
    """The same as the default config, but it prints wires/circuit before and after transpilation"""
    config = MonarqDefaultConfig(machine_name, use_benchmark, q1_acceptance, q2_acceptance, excluded_qubits, excluded_couplers)
    config.steps.insert(0, PrintWires() if only_wires else PrintTape())
    config.steps.append(PrintWires() if only_wires else PrintTape())

    return config

def PrintNoPlaceNoRouteConfig(only_wires = True):
    """The same as the NoPlaceNoRoute config, but it prints wires/circuit before and after transpilation"""
    config = NoPlaceNoRouteConfig()
    config.steps.insert(0, PrintWires() if only_wires else PrintTape())
    config.steps.append(PrintWires() if only_wires else PrintTape())
    return config

def FakeMonarqConfig(machine_name : str, use_benchmark = False): 
    """
    A configuration preset that does the same thing as the default config, but adds gate and readout noise at the end
    """
    return ProcessingConfig(DecomposeReadout(),
                            CliffordTDecomposition(),
                            VF2(machine_name, use_benchmark),
                            Swaps(machine_name, use_benchmark),
                            IterativeCommuteAndMerge(),
                            MonarqDecomposition(),
                            IterativeCommuteAndMerge(),
                            MonarqDecomposition(),
                            GateNoiseSimulation(use_benchmark),
                            ReadoutNoiseSimulation(use_benchmark))

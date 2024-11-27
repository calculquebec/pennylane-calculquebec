from pennylane_snowflurry.processing.interfaces.base_step import BaseStep
from pennylane_snowflurry.processing.steps import CliffordTDecomposition, ASTAR, Swaps, IterativeCommuteAndMerge, MonarqDecomposition

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
    
    
class MonarqDefaultConfig(ProcessingConfig):
    def __init__(self, 
                 use_benchmark = True, 
                 q1_acceptance = 0.5, 
                 q2_acceptance = 0.5, 
                 excluded_qubits : list[int] = [], 
                 excluded_couplers : list[list[int]] = []):
        
        super().__init__(CliffordTDecomposition(), 
                         ASTAR(use_benchmark = use_benchmark, 
                               q1_acceptance = q1_acceptance, 
                               q2_acceptance = q2_acceptance, 
                               excluded_qubits=excluded_qubits, 
                               excluded_couplers=excluded_couplers),
                         Swaps(use_benchmark=use_benchmark, 
                               q1_acceptance=q1_acceptance, 
                               q2_acceptance=q2_acceptance, 
                               excluded_qubits=excluded_qubits, 
                               excluded_couplers=excluded_couplers),
                         IterativeCommuteAndMerge(),
                         MonarqDecomposition())
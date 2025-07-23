"""
this module contains every concrete pre-processing / post-processing steps currently implemented.
"""

from .base_decomposition import CliffordTDecomposition
from .placement import ASTAR, ISMAGS, VF2
from .routing import Swaps
from .optimization import IterativeCommuteAndMerge
from .native_decomposition import MonarqDecomposition
from .readout_error_mitigation import MatrixReadoutMitigation, IBUReadoutMitigation
from .decompose_readout import DecomposeReadout
from .gate_noise_simulation import GateNoiseSimulation
from .readout_noise_simulation import ReadoutNoiseSimulation
from .print_steps import PrintResults, PrintTape, PrintWires

from pennylane_calculquebec.processing import ProcessingError
class StepsError(ProcessingError):
    """Error related to steps."""
    def __init__(self):
        predefined = "Steps Error/"
        error_message = self.__class__.__name__
        full_message = f"{predefined} {error_message}"
        super().__init__(full_message)
        self.message = full_message

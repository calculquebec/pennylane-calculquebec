"""Concrete pre-processing and post-processing steps for the MonarQ transpilation pipeline.

Pre-processing steps (applied before circuit execution):

* :class:`~pennylane_calculquebec.processing.steps.base_decomposition.CliffordTDecomposition` — decomposes gates into the Clifford+T gate set
* :class:`~pennylane_calculquebec.processing.steps.native_decomposition.MonarqDecomposition` — further decomposes into MonarQ native gates
* :class:`~pennylane_calculquebec.processing.steps.decompose_readout.DecomposeReadout` — decomposes non-computational-basis measurements
* :class:`~pennylane_calculquebec.processing.steps.placement.ASTAR` / :class:`~pennylane_calculquebec.processing.steps.placement.VF2` / :class:`~pennylane_calculquebec.processing.steps.placement.ISMAGS` — qubit placement algorithms
* :class:`~pennylane_calculquebec.processing.steps.routing.Swaps` — SWAP-based qubit routing
* :class:`~pennylane_calculquebec.processing.steps.optimization.IterativeCommuteAndMerge` — gate commutation and merging optimisation
* :class:`~pennylane_calculquebec.processing.steps.gate_noise_simulation.GateNoiseSimulation` — inserts gate-noise channels for simulation

Post-processing steps (applied to measurement results):

* :class:`~pennylane_calculquebec.processing.steps.readout_noise_simulation.ReadoutNoiseSimulation` — applies readout noise for simulation
* :class:`~pennylane_calculquebec.processing.steps.readout_error_mitigation.MatrixReadoutMitigation` — matrix-inversion readout error mitigation
* :class:`~pennylane_calculquebec.processing.steps.readout_error_mitigation.IBUReadoutMitigation` — iterative Bayesian unfolding readout error mitigation

Debug steps:

* :class:`~pennylane_calculquebec.processing.steps.print_steps.PrintTape` / :class:`~pennylane_calculquebec.processing.steps.print_steps.PrintResults` / :class:`~pennylane_calculquebec.processing.steps.print_steps.PrintWires` — print intermediate pipeline state
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
from pennylane_calculquebec.exceptions import StepsError

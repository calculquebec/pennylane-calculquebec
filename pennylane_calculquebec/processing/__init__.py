"""Circuit processing pipeline for MonarQ devices.

Contains pre-processing (transpilation) and post-processing steps organised
into submodules:

* :mod:`~pennylane_calculquebec.processing.steps` — concrete pre/post-processing steps
  (decomposition, placement, routing, optimisation, noise simulation, error mitigation)
* :mod:`~pennylane_calculquebec.processing.config` — pipeline configuration classes and presets
* :mod:`~pennylane_calculquebec.processing.interfaces` — abstract base classes for custom steps
* :mod:`~pennylane_calculquebec.processing.custom_gates` — MonarQ native gate definitions
"""

from .monarq_postproc import PostProcessor
from .monarq_preproc import PreProcessor
from pennylane_calculquebec.exceptions import ProcessingError

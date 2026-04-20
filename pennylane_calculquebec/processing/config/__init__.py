"""Processing pipeline configuration classes and built-in presets.

Exposes :class:`~pennylane_calculquebec.processing.config.processing_config.ProcessingConfig`
and the following ready-to-use preset factories:

* :func:`~pennylane_calculquebec.processing.config.processing_config.MonarqDefaultConfig` — full transpilation pipeline with benchmark data
* :func:`~pennylane_calculquebec.processing.config.processing_config.MonarqDefaultConfigNoBenchmark` — full pipeline without benchmark
* :func:`~pennylane_calculquebec.processing.config.processing_config.NoPlaceNoRouteConfig` — decomposition only
* :func:`~pennylane_calculquebec.processing.config.processing_config.EmptyConfig` — no processing steps
* :func:`~pennylane_calculquebec.processing.config.processing_config.PrintDefaultConfig` — default pipeline with debug printing
"""

from .processing_config import (
    ProcessingConfig,
    MonarqDefaultConfig,
    NoPlaceNoRouteConfig,
    MonarqDefaultConfigNoBenchmark,
    EmptyConfig,
    FakeMonarqConfig,
    PrintDefaultConfig,
    PrintNoPlaceNoRouteConfig,
)

from pennylane_calculquebec.exceptions import ConfigError

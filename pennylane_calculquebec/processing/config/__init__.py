"""This module contains configuration classes and presets."""

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

from pennylane_calculquebec.exception import ConfigError
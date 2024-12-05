"""this is the top level module for the Pennylane Snowflurry plugin. It is used for communicating with MonarQ.
"""

from .julia_setup import JuliaEnv

JuliaEnv().update()

from .pennylane_converter import PennylaneConverter
from .snowflurry_device import SnowflurryQubitDevice
from .monarq_device import MonarqDevice

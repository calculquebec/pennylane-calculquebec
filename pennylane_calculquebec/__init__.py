"""this is the top level module for the Pennylane Snowflurry plugin. It is used for communicating with MonarQ.
"""
import importlib.util

if importlib.util.find_spec("juliacall") is not None:
    from .snowflurry_device import SnowflurryQubitDevice

from .monarq_device import MonarqDevice

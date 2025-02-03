"""this is the top level module for the Pennylane Snowflurry plugin. It is used for communicating with MonarQ.
"""
import importlib.util

if importlib.util.find_spec("juliacall") is not None:
    from .snowflurry_device import SnowflurryQubitDevice
else:
    print("won't import snowflurry.qubit")

from .monarq_device import MonarqDevice

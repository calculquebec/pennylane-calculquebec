"""
This module contains everything service/network related for communicating with MonarQ.
"""

from pennylane_calculquebec import PennylaneCQError

class ApiError(PennylaneCQError):
    """Error related to API."""
    def __init__(self, message: str):
        predefined = "API Error/ "
        full_message = f"{predefined}{message}"
        super().__init__(full_message)
        self.message = full_message
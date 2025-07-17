from .pennylane_cq_error import PennylaneCQError

class DeviceError(PennylaneCQError):
    """Error related to device."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

from pennylane_calculquebec.pennylane_cq_error import PennylaneCQError

class DeviceError(PennylaneCQError):
    """Error related to device."""
    def __init__(self, message: str):
        predefined = "Device Error/"
        full_message = f"{predefined}{message}"
        super().__init__(full_message)
        self.message = full_message
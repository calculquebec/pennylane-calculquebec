from pennylane_calculquebec.pennylane_cq_error import PennylaneCQError

class ApiError(PennylaneCQError):
    """Error related to API."""
    def __init__(self, message: str):
        predefined = "API Error/ "
        full_message = f"{predefined}{message}"
        super().__init__(full_message)
        self.message = full_message
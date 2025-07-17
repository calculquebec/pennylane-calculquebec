from .pennylane_cq_error import PennylaneCQError

class ApiError(PennylaneCQError):
    """Error related to API."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

from .pennylane_cq_error import PennylaneCQError

class ProcessingError(PennylaneCQError):
    """Error related to processing."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

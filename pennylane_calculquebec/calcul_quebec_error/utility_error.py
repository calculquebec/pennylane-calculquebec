from .pennylane_cq_error import PennylaneCQError

class UtilityError(PennylaneCQError):
    """Error related to utility."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

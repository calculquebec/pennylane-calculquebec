from .pennylane_cq_error import PennylaneCQError

class UtilityError(PennylaneCQError):
    """Error related to utility."""
    def __init__(self, message: str):
        predefined = "Utility Error/"
        full_message = f"{predefined}{message}"
        super().__init__(full_message)
        self.message = full_message

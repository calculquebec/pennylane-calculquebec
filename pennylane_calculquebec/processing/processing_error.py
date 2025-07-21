from pennylane_calculquebec.pennylane_cq_error import PennylaneCQError
class ProcessingError(PennylaneCQError):
    """Error related to processing."""
    def __init__(self, message: str):
        predefined = "Processing Error/"
        full_message = f"{predefined}{message}"
        super().__init__(full_message)
        self.message = full_message
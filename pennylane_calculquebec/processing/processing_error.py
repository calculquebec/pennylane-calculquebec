from pennylane_calculquebec.exception import PennylaneCQError as CalculQuebecError
class ProcessingError(CalculQuebecError):
    """Error related to processing."""
    def __init__(self, message: str):
        predefined = "Processing Error/"
        full_message = f"{predefined}{message}"
        super().__init__(full_message)
        self.message = full_message
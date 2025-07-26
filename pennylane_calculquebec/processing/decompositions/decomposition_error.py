from .. import ProcessingError
class DecompositionError(ProcessingError):
    """Error related to decomposition."""
    def __init__(self, message: str):
        predefined = "Decomposition Error/"
        full_message = f"{predefined}{message}"
        super().__init__(full_message)
        self.message = full_message
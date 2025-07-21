from pennylane_calculquebec.processing.processing_error import ProcessingError

class DecompositionsError(ProcessingError):
    """Error related to decompositions."""
    def __init__(self, message: str):
        predefined = "Decompositions Error/"
        full_message = f"{predefined}{message}"
        super().__init__(full_message)
        self.message = full_message
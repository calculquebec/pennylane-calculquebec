from pennylane_calculquebec.processing.processing_error import ProcessingError
class OptimizationError(ProcessingError):
    """Error related to optimization."""
    def __init__(self, message: str):
        predefined = "Optimization Error/"
        full_message = f"{predefined}{message}"
        super().__init__(full_message)
        self.message = full_message
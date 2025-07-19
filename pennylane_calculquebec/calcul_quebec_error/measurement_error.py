from .processing_error import ProcessingError

class MeasurementError(ProcessingError):
    """Error related to measurement."""
    def __init__(self, message: str):
        predefined = "Measurement Error/"
        full_message = f"{predefined}{message}"
        super().__init__(full_message)
        self.message = full_message

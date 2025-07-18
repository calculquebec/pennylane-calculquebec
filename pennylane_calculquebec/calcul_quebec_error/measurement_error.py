from .processing_error import ProcessingError

class MeasurementError(ProcessingError):
    """Error related to measurement."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

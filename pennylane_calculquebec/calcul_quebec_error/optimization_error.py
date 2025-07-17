from .processing_error import ProcessingError

class OptimizationError(ProcessingError):
    """Error related to optimization."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

from .processing_error import ProcessingError

class StepsError(ProcessingError):
    """Error related to steps."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

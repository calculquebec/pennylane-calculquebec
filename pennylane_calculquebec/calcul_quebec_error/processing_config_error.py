from .processing_error import ProcessingError

class ProcessingConfigError(ProcessingError):
    """Error related to processing config."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

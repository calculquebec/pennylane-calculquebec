from .processing_error import ProcessingError

class DecompositionsError(ProcessingError):
    """Error related to decompositions."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

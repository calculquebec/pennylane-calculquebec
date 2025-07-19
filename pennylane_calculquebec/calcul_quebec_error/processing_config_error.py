from .processing_error import ProcessingError

class ProcessingConfigError(ProcessingError):
    """Error related to processing config."""
    def __init__(self, message: str):
        predefined = "Processing Config Error/"
        full_message = f"{predefined}{message}"
        super().__init__(full_message)
        self.message = full_message

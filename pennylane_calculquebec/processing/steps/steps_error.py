from pennylane_calculquebec.processing.processing_error import ProcessingError
class StepsError(ProcessingError):
    """Error related to steps."""
    def __init__(self, message: str):
        predefined = "Steps Error/"
        full_message = f"{predefined}{message}"
        super().__init__(full_message)
        self.message = full_message
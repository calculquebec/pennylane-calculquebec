from pennylane_calculquebec.processing.processing_error import ProcessingError
class StepsError(ProcessingError):
    """Error related to steps."""
    def __init__(self):
        predefined = "Steps Error/"
        error_message = self.__class__.__name__
        full_message = f"{predefined} {error_message}"
        super().__init__(full_message)
        self.message = full_message

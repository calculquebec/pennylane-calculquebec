from .error import Error

class PennylaneCQError(Error):
    """Pennylane Calcul Quebec base error."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
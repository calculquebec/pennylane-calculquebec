

class PennylaneCQError(Exception):
    """Pennylane Calcul Quebec base error."""
    def __init__(self, message: str):
        predefined = "Error coming from Pennylane Calcul Quebec/"
        full_message = f"{predefined}{message}"
        super().__init__(full_message)
        self.message = full_message
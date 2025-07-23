from pennylane_calculquebec import PennylaneCQError

class UtilityError(PennylaneCQError):
    """Error related to utility."""
    def __init__(self,):
        predefined = "Utility Error/"
        error_type=self.__class__.__name__
        full_message = f"{predefined} of type {error_type}"
        super().__init__(full_message)
        self.message = full_message
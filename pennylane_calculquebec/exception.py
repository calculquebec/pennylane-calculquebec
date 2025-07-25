from pennylane_calculquebec.logger import logger
class PennylaneCQError(Exception):
    """Pennylane Calcul Quebec base error."""
    def __init__(self, message: str):
        predefined = "Error coming from Pennylane Calcul Quebec/"
        full_message = f"{predefined}{message}"
        super().__init__(full_message)
        self.message = full_message
        logger.error(full_message)

class DeviceError(PennylaneCQError):
    """Error related to device."""
    def __init__(self, name_of_device:str):
        predefined = "Device Error/"
        error_message= self.__class__.__name__
        full_message = f"{predefined} of type {error_message} for device {name_of_device}"
        super().__init__(full_message)
        self.name_of_device = name_of_device
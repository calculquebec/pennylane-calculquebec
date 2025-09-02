class PennylaneCQError(Exception):
    """Pennylane Calcul Quebec base error."""

    def __init__(self, message: str):
        class_names = []
        cls = self.__class__
        while cls != Exception:
            class_names.append(cls.__name__)
            if not cls.__bases__:
                break
            cls = cls.__bases__[0]
        class_path = " / ".join(reversed(class_names))
        predefined = f"{class_path} "
        full_message = f"{predefined}{message}"
        super().__init__(full_message)
        self.message = full_message
        from pennylane_calculquebec.logger import logger

        logger.error(full_message)


class DeviceError(PennylaneCQError):
    """Error related to device."""


class ProcessingError(PennylaneCQError):
    """Error related to processing."""


class UtilityError(PennylaneCQError):
    """Error related to utility."""


class ApiError(PennylaneCQError):
    """Error related to API."""


class ConfigError(ProcessingError):
    """Error related to processing configuration."""


class StepsError(ProcessingError):
    """Error related to steps."""

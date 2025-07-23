"""This module and its submodule contains everything one may need for using monarq.default processing capabilities."""

from .monarq_postproc import PostProcessor
from .monarq_preproc import PreProcessor

from pennylane_calculquebec import PennylaneCQError as CalculQuebecError
class ProcessingError(CalculQuebecError):
    """Error related to processing."""
    def __init__(self, message: str):
        predefined = "Processing Error/"
        full_message = f"{predefined}{message}"
        super().__init__(full_message)
        self.message = full_message
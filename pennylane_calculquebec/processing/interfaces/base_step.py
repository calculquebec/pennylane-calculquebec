"""
This is a base class for every processing steps. \n
This should probably not be used. Try using PreProcStep or PostProcStep.
"""

class BaseStep:
    """
    a step that can be inherited in order to be appended to a TranspilerConfig. 
    Adding a step to a TranspilerConfig means it will be applied in the preprocess step of MonarqDevice.
    """
    pass
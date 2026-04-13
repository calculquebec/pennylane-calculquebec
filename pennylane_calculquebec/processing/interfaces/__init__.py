"""Abstract base classes for custom pre-processing and post-processing steps.

Provides :class:`~pennylane_calculquebec.processing.interfaces.pre_proc_step.PreProcStep`
and :class:`~pennylane_calculquebec.processing.interfaces.post_proc_step.PostProcStep`
as the extension points for adding new transpilation or result-processing steps
to the MonarQ pipeline.
"""

from .post_proc_step import PostProcStep
from .pre_proc_step import PreProcStep

"""Legacy alias module for :class:`~pennylane_calculquebec.exceptions.DeviceError`.

Kept for backwards compatibility.  New code should import
:class:`~pennylane_calculquebec.exceptions.DeviceError` directly.
"""

from pennylane_calculquebec.exceptions import DeviceError


class DeviceException(DeviceError):
    """Exception for device-related errors (legacy alias)."""

    pass

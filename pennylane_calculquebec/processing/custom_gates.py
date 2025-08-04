"""
Contains custom gates for completing MonarQ's native gate set
"""

from pennylane.operation import Operation
from functools import lru_cache
import numpy as np
import pennylane as qml
from copy import copy
from pennylane_calculquebec.logger import logger


class TDagger(Operation):
    r"""ajoint(T)(pi/2)(wires)
    The single-qubit ajoint of T operation

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """

    num_wires = 1
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    basis = "Z"

    batch_size = None

    @staticmethod
    @lru_cache()
    def compute_matrix():
        try:
            return qml.PhaseShift.compute_matrix(-np.pi / 4)
        except Exception as e:
            logger.error(
                "Error %s in compute_matrix located in TDagger: %s", type(e).__name__, e
            )
            return None

    @staticmethod
    def compute_eigvals():
        try:
            return np.linalg.eigvals(TDagger.compute_matrix())
        except Exception as e:
            logger.error(
                "Error %s in compute_eigvals located in TDagger: %s",
                type(e).__name__,
                e,
            )
            return None

    @staticmethod
    def compute_decomposition(wires):
        try:
            return [qml.adjoint(qml.T(wires))]
        except Exception as e:
            logger.error(
                "Error %s in compute_decomposition located in TDagger: %s",
                type(e).__name__,
                e,
            )
            return []

    def pow(self, z):
        z = z % 8
        pow_map = {
            0: [],
            1: [copy(self)],
            2: [qml.adjoint(qml.S)(wires=self.wires)],
            3: [qml.PauliZ(wires=self.wires), qml.T(wires=self.wires)],
            4: [qml.PauliZ(wires=self.wires)],
            5: [qml.S(wires=self.wires), qml.T(wires=self.wires)],
            6: [qml.S(wires=self.wires)],
            7: [qml.T(wires=self.wires)],
        }
        return pow_map[z]

    def adjoint(self):
        return qml.T(self.wires)

    def single_qubit_rot_angles(self):
        return [-np.pi / 4, 0, 0]


class X90(Operation):
    r"""RX(pi/2)(wires)
    The single-qubit rotation of 90 degrees around the X axis

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """

    num_wires = 1
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    basis = "X"

    batch_size = None

    @staticmethod
    @lru_cache()
    def compute_matrix():
        try:
            return qml.RX.compute_matrix(np.pi / 2)
        except Exception as e:
            logger.error(
                "Error %s in compute_matrix located in X90: %s", type(e).__name__, e
            )
            return None

    @staticmethod
    def compute_eigvals():
        try:
            return np.linalg.eigvals(X90.compute_matrix())
        except Exception as e:
            logger.error(
                "Error %s in compute_eigvals located in X90: %s", type(e).__name__, e
            )
            return None

    @staticmethod
    def compute_decomposition(wires):
        try:
            return [qml.RX(np.pi / 2, wires)]
        except Exception as e:
            logger.error(
                "Error %s in compute_decomposition located in X90: %s",
                type(e).__name__,
                e,
            )
            return []

    def pow(self, z):
        z = z % 8
        angle = z * np.pi / 2
        return [qml.RX(angle, self.wires)]

    def adjoint(self):
        return XM90(self.wires)

    def single_qubit_rot_angles(self):
        return [np.pi / 2, np.pi / 2, -np.pi / 2]


class XM90(Operation):
    r"""RX(-pi/2)(wires)
    The single-qubit rotation of -90 degrees around the X axis

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """

    num_wires = 1
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    basis = "X"

    batch_size = None

    @staticmethod
    @lru_cache()
    def compute_matrix():
        try:
            return qml.RX.compute_matrix(-np.pi / 2)
        except Exception as e:
            logger.error(
                "Error %s in compute_matrix located in XM90: %s", type(e).__name__, e
            )
            return None

    @staticmethod
    def compute_eigvals():
        try:
            return np.linalg.eigvals(XM90.compute_matrix())
        except Exception as e:
            logger.error(
                "Error %s in compute_eigvals located in XM90: %s", type(e).__name__, e
            )
            return None

    @staticmethod
    def compute_decomposition(wires):
        try:
            return [qml.RX(-np.pi / 2, wires)]
        except Exception as e:
            logger.error(
                "Error %s in compute_decomposition located in XM90: %s",
                type(e).__name__,
                e,
            )
            return []

    def pow(self, z):
        z = z % 8
        angle = -z * np.pi / 2
        return [qml.RX(angle, self.wires)]

    def adjoint(self):
        return X90(self.wires)

    def single_qubit_rot_angles(self):
        return [np.pi / 2, -np.pi / 2, -np.pi / 2]


class Y90(Operation):
    r"""RY(pi/2)(wires)
    The single-qubit rotation of 90 degrees around the Y axis

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """

    num_wires = 1
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    basis = "Y"

    batch_size = None

    @staticmethod
    @lru_cache()
    def compute_matrix():
        try:
            return qml.RY.compute_matrix(np.pi / 2)
        except Exception as e:
            logger.error(
                "Error %s in compute_matrix located in Y90: %s", type(e).__name__, e
            )
            return None

    @staticmethod
    def compute_eigvals():
        try:
            return np.linalg.eigvals(Y90.compute_matrix())
        except Exception as e:
            logger.error(
                "Error %s in compute_eigvals located in Y90: %s", type(e).__name__, e
            )
            return None

    @staticmethod
    def compute_decomposition(wires):
        try:
            return [qml.RY(np.pi / 2, wires)]
        except Exception as e:
            logger.error(
                "Error %s in compute_decomposition located in Y90: %s",
                type(e).__name__,
                e,
            )
            return []

    def pow(self, z):
        z = z % 8
        angle = z * np.pi / 2
        return [qml.RY(angle, self.wires)]

    def adjoint(self):
        return YM90(self.wires)

    def single_qubit_rot_angles(self):
        return [0, np.pi / 2, 0]


class YM90(Operation):
    r"""RY(-pi/2)(wires)
    The single-qubit rotation of -90 degrees around the Y axis

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """

    num_wires = 1
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    basis = "Y"

    batch_size = None

    @staticmethod
    @lru_cache()
    def compute_matrix():
        try:
            return qml.RY.compute_matrix(-np.pi / 2)
        except Exception as e:
            logger.error(
                "Error %s in compute_matrix located in YM90: %s", type(e).__name__, e
            )
            return None

    @staticmethod
    def compute_eigvals():
        try:
            return np.linalg.eigvals(YM90.compute_matrix())
        except Exception as e:
            logger.error(
                "Error %s in compute_eigvals located in YM90: %s", type(e).__name__, e
            )
            return None

    @staticmethod
    def compute_decomposition(wires):
        try:
            return [qml.RY(-np.pi / 2, wires)]
        except Exception as e:
            logger.error(
                "Error %s in compute_decomposition located in YM90: %s",
                type(e).__name__,
                e,
            )
            return []

    def pow(self, z):
        z = z % 8
        angle = -z * np.pi / 2
        return [qml.RY(angle, self.wires)]

    def adjoint(self):
        return Y90(self.wires)

    def single_qubit_rot_angles(self):
        return [0, -np.pi / 2, 0]


class Z90(Operation):
    r"""RZ(pi/2)(wires)
    The single-qubit rotation of 90 degrees around the Z axis

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """

    num_wires = 1
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    basis = "Z"

    batch_size = None

    @staticmethod
    @lru_cache()
    def compute_matrix():
        try:
            return qml.RZ.compute_matrix(np.pi / 2)
        except Exception as e:
            logger.error(
                "Error %s in compute_matrix located in Z90: %s", type(e).__name__, e
            )
            return None

    @staticmethod
    def compute_eigvals():
        try:
            return np.linalg.eigvals(Z90.compute_matrix())
        except Exception as e:
            logger.error(
                "Error %s in compute_eigvals located in Z90: %s", type(e).__name__, e
            )
            return None

    @staticmethod
    def compute_decomposition(wires):
        try:
            return [qml.RZ(np.pi / 2, wires)]
        except Exception as e:
            logger.error(
                "Error %s in compute_decomposition located in Z90: %s",
                type(e).__name__,
                e,
            )
            return []

    def pow(self, z):
        z = z % 8
        angle = z * np.pi / 2
        return [qml.RZ(angle, self.wires)]

    def adjoint(self):
        return ZM90(self.wires)

    def single_qubit_rot_angles(self):
        return [np.pi / 2, 0, 0]


class ZM90(Operation):
    r"""RZ(-pi/2)(wires)
    The single-qubit rotation of -90 degrees around the Z axis

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """

    num_wires = 1
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    basis = "Z"

    batch_size = None

    @staticmethod
    @lru_cache()
    def compute_matrix():
        try:
            return qml.RZ.compute_matrix(-np.pi / 2)
        except Exception as e:
            logger.error(
                "Error %s in compute_matrix located in ZM90: %s", type(e).__name__, e
            )
            return None

    @staticmethod
    def compute_eigvals():
        try:
            return np.linalg.eigvals(ZM90.compute_matrix())
        except Exception as e:
            logger.error(
                "Error %s in compute_eigvals located in ZM90: %s", type(e).__name__, e
            )
            return None

    @staticmethod
    def compute_decomposition(wires):
        try:
            return [qml.RZ(-np.pi / 2, wires)]
        except Exception as e:
            logger.error(
                "Error %s in compute_decomposition located in ZM90: %s",
                type(e).__name__,
                e,
            )
            return []

    def pow(self, z):
        z = z % 8
        angle = -z * np.pi / 2
        return [qml.RZ(angle, self.wires)]

    def adjoint(self):
        return Z90(self.wires)

    def single_qubit_rot_angles(self):
        return [-np.pi / 2, 0, 0]

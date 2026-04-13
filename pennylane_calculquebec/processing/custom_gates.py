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
    r"""The single-qubit adjoint of the T gate, equivalent to :math:`T^\dagger = \text{PhaseShift}(-\pi/4)`.

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
        """Compute the canonical matrix representation of TDagger.

        Returns:
            numpy.ndarray: 2x2 unitary matrix for the :math:`T^\dagger` gate.
        """
        try:
            return qml.PhaseShift.compute_matrix(-np.pi / 4)
        except Exception as e:
            logger.error(
                "Error %s in compute_matrix located in TDagger: %s", type(e).__name__, e
            )
            return None

    @staticmethod
    def compute_eigvals():
        """Compute the eigenvalues of TDagger.

        Returns:
            numpy.ndarray: eigenvalues of the :math:`T^\dagger` gate.
        """
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
        """Decompose TDagger into primitive PennyLane operations.

        Args:
            wires (Sequence[int] or int): the wire the operation acts on

        Returns:
            list[Operation]: ``[adjoint(T(wires))]``
        """
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
        """Raise TDagger to an integer power.

        Reduces ``z`` modulo 8 (the order of :math:`T^\dagger` in :math:`U(1)`) and returns
        the corresponding sequence of operations using the minimal gate set.

        Args:
            z (int): the exponent

        Returns:
            list[Operation]: equivalent gate sequence for ``TDagger ** z``
        """
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
        """Return the adjoint of TDagger, which is the T gate.

        Returns:
            Operation: :class:`~pennylane.T` acting on the same wire
        """
        return qml.T(self.wires)

    def single_qubit_rot_angles(self):
        """Euler rotation angles (ZYZ convention) that reproduce TDagger.

        Returns:
            list[float]: ``[phi, theta, omega]`` such that
            ``RZ(phi) @ RY(theta) @ RZ(omega)`` equals :math:`T^\dagger`.
        """
        return [-np.pi / 4, 0, 0]


class X90(Operation):
    r"""The single-qubit rotation of 90 degrees around the X axis, equivalent to :math:`RX(\pi/2)`.

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
        """Compute the canonical matrix representation of X90.

        Returns:
            numpy.ndarray: 2x2 unitary matrix for :math:`RX(\pi/2)`.
        """
        try:
            return qml.RX.compute_matrix(np.pi / 2)
        except Exception as e:
            logger.error(
                "Error %s in compute_matrix located in X90: %s", type(e).__name__, e
            )
            return None

    @staticmethod
    def compute_eigvals():
        """Compute the eigenvalues of X90.

        Returns:
            numpy.ndarray: eigenvalues of :math:`RX(\pi/2)`.
        """
        try:
            return np.linalg.eigvals(X90.compute_matrix())
        except Exception as e:
            logger.error(
                "Error %s in compute_eigvals located in X90: %s", type(e).__name__, e
            )
            return None

    @staticmethod
    def compute_decomposition(wires):
        """Decompose X90 into primitive PennyLane operations.

        Args:
            wires (Sequence[int] or int): the wire the operation acts on

        Returns:
            list[Operation]: ``[RX(pi/2, wires)]``
        """
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
        """Raise X90 to an integer power.

        Args:
            z (int): the exponent

        Returns:
            list[Operation]: ``[RX(z * pi/2, wires)]``
        """
        z = z % 8
        angle = z * np.pi / 2
        return [qml.RX(angle, self.wires)]

    def adjoint(self):
        """Return the adjoint of X90, which is XM90.

        Returns:
            Operation: :class:`XM90` acting on the same wire
        """
        return XM90(self.wires)

    def single_qubit_rot_angles(self):
        """Euler rotation angles (ZYZ convention) that reproduce X90.

        Returns:
            list[float]: ``[phi, theta, omega]`` such that
            ``RZ(phi) @ RY(theta) @ RZ(omega)`` equals :math:`RX(\pi/2)`.
        """
        return [np.pi / 2, np.pi / 2, -np.pi / 2]


class XM90(Operation):
    r"""The single-qubit rotation of -90 degrees around the X axis, equivalent to :math:`RX(-\pi/2)`.

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
        """Compute the canonical matrix representation of XM90.

        Returns:
            numpy.ndarray: 2x2 unitary matrix for :math:`RX(-\pi/2)`.
        """
        try:
            return qml.RX.compute_matrix(-np.pi / 2)
        except Exception as e:
            logger.error(
                "Error %s in compute_matrix located in XM90: %s", type(e).__name__, e
            )
            return None

    @staticmethod
    def compute_eigvals():
        """Compute the eigenvalues of XM90.

        Returns:
            numpy.ndarray: eigenvalues of :math:`RX(-\pi/2)`.
        """
        try:
            return np.linalg.eigvals(XM90.compute_matrix())
        except Exception as e:
            logger.error(
                "Error %s in compute_eigvals located in XM90: %s", type(e).__name__, e
            )
            return None

    @staticmethod
    def compute_decomposition(wires):
        """Decompose XM90 into primitive PennyLane operations.

        Args:
            wires (Sequence[int] or int): the wire the operation acts on

        Returns:
            list[Operation]: ``[RX(-pi/2, wires)]``
        """
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
        """Raise XM90 to an integer power.

        Args:
            z (int): the exponent

        Returns:
            list[Operation]: ``[RX(-z * pi/2, wires)]``
        """
        z = z % 8
        angle = -z * np.pi / 2
        return [qml.RX(angle, self.wires)]

    def adjoint(self):
        """Return the adjoint of XM90, which is X90.

        Returns:
            Operation: :class:`X90` acting on the same wire
        """
        return X90(self.wires)

    def single_qubit_rot_angles(self):
        """Euler rotation angles (ZYZ convention) that reproduce XM90.

        Returns:
            list[float]: ``[phi, theta, omega]`` such that
            ``RZ(phi) @ RY(theta) @ RZ(omega)`` equals :math:`RX(-\pi/2)`.
        """
        return [np.pi / 2, -np.pi / 2, -np.pi / 2]


class Y90(Operation):
    r"""The single-qubit rotation of 90 degrees around the Y axis, equivalent to :math:`RY(\pi/2)`.

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
        """Compute the canonical matrix representation of Y90.

        Returns:
            numpy.ndarray: 2x2 unitary matrix for :math:`RY(\pi/2)`.
        """
        try:
            return qml.RY.compute_matrix(np.pi / 2)
        except Exception as e:
            logger.error(
                "Error %s in compute_matrix located in Y90: %s", type(e).__name__, e
            )
            return None

    @staticmethod
    def compute_eigvals():
        """Compute the eigenvalues of Y90.

        Returns:
            numpy.ndarray: eigenvalues of :math:`RY(\pi/2)`.
        """
        try:
            return np.linalg.eigvals(Y90.compute_matrix())
        except Exception as e:
            logger.error(
                "Error %s in compute_eigvals located in Y90: %s", type(e).__name__, e
            )
            return None

    @staticmethod
    def compute_decomposition(wires):
        """Decompose Y90 into primitive PennyLane operations.

        Args:
            wires (Sequence[int] or int): the wire the operation acts on

        Returns:
            list[Operation]: ``[RY(pi/2, wires)]``
        """
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
        """Raise Y90 to an integer power.

        Args:
            z (int): the exponent

        Returns:
            list[Operation]: ``[RY(z * pi/2, wires)]``
        """
        z = z % 8
        angle = z * np.pi / 2
        return [qml.RY(angle, self.wires)]

    def adjoint(self):
        """Return the adjoint of Y90, which is YM90.

        Returns:
            Operation: :class:`YM90` acting on the same wire
        """
        return YM90(self.wires)

    def single_qubit_rot_angles(self):
        """Euler rotation angles (ZYZ convention) that reproduce Y90.

        Returns:
            list[float]: ``[phi, theta, omega]`` such that
            ``RZ(phi) @ RY(theta) @ RZ(omega)`` equals :math:`RY(\pi/2)`.
        """
        return [0, np.pi / 2, 0]


class YM90(Operation):
    r"""The single-qubit rotation of -90 degrees around the Y axis, equivalent to :math:`RY(-\pi/2)`.

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
        """Compute the canonical matrix representation of YM90.

        Returns:
            numpy.ndarray: 2x2 unitary matrix for :math:`RY(-\pi/2)`.
        """
        try:
            return qml.RY.compute_matrix(-np.pi / 2)
        except Exception as e:
            logger.error(
                "Error %s in compute_matrix located in YM90: %s", type(e).__name__, e
            )
            return None

    @staticmethod
    def compute_eigvals():
        """Compute the eigenvalues of YM90.

        Returns:
            numpy.ndarray: eigenvalues of :math:`RY(-\pi/2)`.
        """
        try:
            return np.linalg.eigvals(YM90.compute_matrix())
        except Exception as e:
            logger.error(
                "Error %s in compute_eigvals located in YM90: %s", type(e).__name__, e
            )
            return None

    @staticmethod
    def compute_decomposition(wires):
        """Decompose YM90 into primitive PennyLane operations.

        Args:
            wires (Sequence[int] or int): the wire the operation acts on

        Returns:
            list[Operation]: ``[RY(-pi/2, wires)]``
        """
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
        """Raise YM90 to an integer power.

        Args:
            z (int): the exponent

        Returns:
            list[Operation]: ``[RY(-z * pi/2, wires)]``
        """
        z = z % 8
        angle = -z * np.pi / 2
        return [qml.RY(angle, self.wires)]

    def adjoint(self):
        """Return the adjoint of YM90, which is Y90.

        Returns:
            Operation: :class:`Y90` acting on the same wire
        """
        return Y90(self.wires)

    def single_qubit_rot_angles(self):
        """Euler rotation angles (ZYZ convention) that reproduce YM90.

        Returns:
            list[float]: ``[phi, theta, omega]`` such that
            ``RZ(phi) @ RY(theta) @ RZ(omega)`` equals :math:`RY(-\pi/2)`.
        """
        return [0, -np.pi / 2, 0]


class Z90(Operation):
    r"""The single-qubit rotation of 90 degrees around the Z axis, equivalent to :math:`RZ(\pi/2)`.

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
        """Compute the canonical matrix representation of Z90.

        Returns:
            numpy.ndarray: 2x2 unitary matrix for :math:`RZ(\pi/2)`.
        """
        try:
            return qml.RZ.compute_matrix(np.pi / 2)
        except Exception as e:
            logger.error(
                "Error %s in compute_matrix located in Z90: %s", type(e).__name__, e
            )
            return None

    @staticmethod
    def compute_eigvals():
        """Compute the eigenvalues of Z90.

        Returns:
            numpy.ndarray: eigenvalues of :math:`RZ(\pi/2)`.
        """
        try:
            return np.linalg.eigvals(Z90.compute_matrix())
        except Exception as e:
            logger.error(
                "Error %s in compute_eigvals located in Z90: %s", type(e).__name__, e
            )
            return None

    @staticmethod
    def compute_decomposition(wires):
        """Decompose Z90 into primitive PennyLane operations.

        Args:
            wires (Sequence[int] or int): the wire the operation acts on

        Returns:
            list[Operation]: ``[RZ(pi/2, wires)]``
        """
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
        """Raise Z90 to an integer power.

        Args:
            z (int): the exponent

        Returns:
            list[Operation]: ``[RZ(z * pi/2, wires)]``
        """
        z = z % 8
        angle = z * np.pi / 2
        return [qml.RZ(angle, self.wires)]

    def adjoint(self):
        """Return the adjoint of Z90, which is ZM90.

        Returns:
            Operation: :class:`ZM90` acting on the same wire
        """
        return ZM90(self.wires)

    def single_qubit_rot_angles(self):
        """Euler rotation angles (ZYZ convention) that reproduce Z90.

        Returns:
            list[float]: ``[phi, theta, omega]`` such that
            ``RZ(phi) @ RY(theta) @ RZ(omega)`` equals :math:`RZ(\pi/2)`.
        """
        return [np.pi / 2, 0, 0]


class ZM90(Operation):
    r"""The single-qubit rotation of -90 degrees around the Z axis, equivalent to :math:`RZ(-\pi/2)`.

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
        """Compute the canonical matrix representation of ZM90.

        Returns:
            numpy.ndarray: 2x2 unitary matrix for :math:`RZ(-\pi/2)`.
        """
        try:
            return qml.RZ.compute_matrix(-np.pi / 2)
        except Exception as e:
            logger.error(
                "Error %s in compute_matrix located in ZM90: %s", type(e).__name__, e
            )
            return None

    @staticmethod
    def compute_eigvals():
        """Compute the eigenvalues of ZM90.

        Returns:
            numpy.ndarray: eigenvalues of :math:`RZ(-\pi/2)`.
        """
        try:
            return np.linalg.eigvals(ZM90.compute_matrix())
        except Exception as e:
            logger.error(
                "Error %s in compute_eigvals located in ZM90: %s", type(e).__name__, e
            )
            return None

    @staticmethod
    def compute_decomposition(wires):
        """Decompose ZM90 into primitive PennyLane operations.

        Args:
            wires (Sequence[int] or int): the wire the operation acts on

        Returns:
            list[Operation]: ``[RZ(-pi/2, wires)]``
        """
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
        """Raise ZM90 to an integer power.

        Args:
            z (int): the exponent

        Returns:
            list[Operation]: ``[RZ(-z * pi/2, wires)]``
        """
        z = z % 8
        angle = -z * np.pi / 2
        return [qml.RZ(angle, self.wires)]

    def adjoint(self):
        """Return the adjoint of ZM90, which is Z90.

        Returns:
            Operation: :class:`Z90` acting on the same wire
        """
        return Z90(self.wires)

    def single_qubit_rot_angles(self):
        """Euler rotation angles (ZYZ convention) that reproduce ZM90.

        Returns:
            list[float]: ``[phi, theta, omega]`` such that
            ``RZ(phi) @ RY(theta) @ RZ(omega)`` equals :math:`RZ(-\pi/2)`.
        """
        return [-np.pi / 2, 0, 0]

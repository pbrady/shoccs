"""
Core field data structures for SHOCCS.

This module provides ScalarField and VectorField classes for representing
scalar and vector fields on computational domains with boundary regions.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ScalarField:
    """
    Represents a scalar field on a computational domain.

    A scalar field consists of values on the domain interior (D) and
    optional boundary regions (Rx, Ry, Rz).

    Attributes:
        D: Domain interior values (NumPy array)
        Rx: Boundary region in x-direction (NumPy array or None)
        Ry: Boundary region in y-direction (NumPy array or None)
        Rz: Boundary region in z-direction (NumPy array or None)
    """
    D: np.ndarray
    Rx: Optional[np.ndarray] = None
    Ry: Optional[np.ndarray] = None
    Rz: Optional[np.ndarray] = None

    @classmethod
    def zeros(cls, shape, Rx_shape=None, Ry_shape=None, Rz_shape=None):
        """
        Create a ScalarField initialized with zeros.

        Args:
            shape: Shape of the domain array
            Rx_shape: Shape of the Rx boundary region (optional)
            Ry_shape: Shape of the Ry boundary region (optional)
            Rz_shape: Shape of the Rz boundary region (optional)

        Returns:
            ScalarField with zero-initialized arrays
        """
        D = np.zeros(shape)
        Rx = np.zeros(Rx_shape) if Rx_shape is not None else None
        Ry = np.zeros(Ry_shape) if Ry_shape is not None else None
        Rz = np.zeros(Rz_shape) if Rz_shape is not None else None
        return cls(D=D, Rx=Rx, Ry=Ry, Rz=Rz)

    def zeros_like(self):
        """
        Create a new ScalarField with the same structure, initialized with zeros.

        Returns:
            ScalarField with zero-initialized arrays matching this field's structure
        """
        D = np.zeros_like(self.D)
        Rx = np.zeros_like(self.Rx) if self.Rx is not None else None
        Ry = np.zeros_like(self.Ry) if self.Ry is not None else None
        Rz = np.zeros_like(self.Rz) if self.Rz is not None else None
        return ScalarField(D=D, Rx=Rx, Ry=Ry, Rz=Rz)

    def copy(self):
        """
        Create a deep copy of this ScalarField.

        Returns:
            ScalarField with copied arrays
        """
        D = self.D.copy()
        Rx = self.Rx.copy() if self.Rx is not None else None
        Ry = self.Ry.copy() if self.Ry is not None else None
        Rz = self.Rz.copy() if self.Rz is not None else None
        return ScalarField(D=D, Rx=Rx, Ry=Ry, Rz=Rz)

    def __add__(self, other):
        """
        Add two ScalarFields or add a scalar to a ScalarField.

        Args:
            other: ScalarField or scalar value

        Returns:
            ScalarField with element-wise sum
        """
        if isinstance(other, ScalarField):
            D = self.D + other.D
            Rx = self.Rx + other.Rx if self.Rx is not None and other.Rx is not None else None
            Ry = self.Ry + other.Ry if self.Ry is not None and other.Ry is not None else None
            Rz = self.Rz + other.Rz if self.Rz is not None and other.Rz is not None else None
            return ScalarField(D=D, Rx=Rx, Ry=Ry, Rz=Rz)
        else:
            # Scalar addition
            D = self.D + other
            Rx = self.Rx + other if self.Rx is not None else None
            Ry = self.Ry + other if self.Ry is not None else None
            Rz = self.Rz + other if self.Rz is not None else None
            return ScalarField(D=D, Rx=Rx, Ry=Ry, Rz=Rz)

    def __radd__(self, other):
        """Right addition (scalar + ScalarField)."""
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtract two ScalarFields or subtract a scalar from a ScalarField.

        Args:
            other: ScalarField or scalar value

        Returns:
            ScalarField with element-wise difference
        """
        if isinstance(other, ScalarField):
            D = self.D - other.D
            Rx = self.Rx - other.Rx if self.Rx is not None and other.Rx is not None else None
            Ry = self.Ry - other.Ry if self.Ry is not None and other.Ry is not None else None
            Rz = self.Rz - other.Rz if self.Rz is not None and other.Rz is not None else None
            return ScalarField(D=D, Rx=Rx, Ry=Ry, Rz=Rz)
        else:
            # Scalar subtraction
            D = self.D - other
            Rx = self.Rx - other if self.Rx is not None else None
            Ry = self.Ry - other if self.Ry is not None else None
            Rz = self.Rz - other if self.Rz is not None else None
            return ScalarField(D=D, Rx=Rx, Ry=Ry, Rz=Rz)

    def __rsub__(self, other):
        """Right subtraction (scalar - ScalarField)."""
        return (-self).__add__(other)

    def __mul__(self, other):
        """
        Multiply two ScalarFields or multiply a ScalarField by a scalar.

        Args:
            other: ScalarField or scalar value

        Returns:
            ScalarField with element-wise product
        """
        if isinstance(other, ScalarField):
            D = self.D * other.D
            Rx = self.Rx * other.Rx if self.Rx is not None and other.Rx is not None else None
            Ry = self.Ry * other.Ry if self.Ry is not None and other.Ry is not None else None
            Rz = self.Rz * other.Rz if self.Rz is not None and other.Rz is not None else None
            return ScalarField(D=D, Rx=Rx, Ry=Ry, Rz=Rz)
        else:
            # Scalar multiplication
            D = self.D * other
            Rx = self.Rx * other if self.Rx is not None else None
            Ry = self.Ry * other if self.Ry is not None else None
            Rz = self.Rz * other if self.Rz is not None else None
            return ScalarField(D=D, Rx=Rx, Ry=Ry, Rz=Rz)

    def __rmul__(self, other):
        """Right multiplication (scalar * ScalarField)."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Divide a ScalarField by another ScalarField or by a scalar.

        Args:
            other: ScalarField or scalar value

        Returns:
            ScalarField with element-wise quotient
        """
        if isinstance(other, ScalarField):
            D = self.D / other.D
            Rx = self.Rx / other.Rx if self.Rx is not None and other.Rx is not None else None
            Ry = self.Ry / other.Ry if self.Ry is not None and other.Ry is not None else None
            Rz = self.Rz / other.Rz if self.Rz is not None and other.Rz is not None else None
            return ScalarField(D=D, Rx=Rx, Ry=Ry, Rz=Rz)
        else:
            # Scalar division
            D = self.D / other
            Rx = self.Rx / other if self.Rx is not None else None
            Ry = self.Ry / other if self.Ry is not None else None
            Rz = self.Rz / other if self.Rz is not None else None
            return ScalarField(D=D, Rx=Rx, Ry=Ry, Rz=Rz)

    def __neg__(self):
        """
        Negate a ScalarField.

        Returns:
            ScalarField with negated values
        """
        D = -self.D
        Rx = -self.Rx if self.Rx is not None else None
        Ry = -self.Ry if self.Ry is not None else None
        Rz = -self.Rz if self.Rz is not None else None
        return ScalarField(D=D, Rx=Rx, Ry=Ry, Rz=Rz)

    def norm(self, order=2):
        """
        Compute the norm of the ScalarField.

        Args:
            order: Order of the norm (default: 2 for L2 norm)
                  Can be any value supported by numpy.linalg.norm

        Returns:
            Scalar norm value
        """
        # Compute norm of domain
        norm_val = np.linalg.norm(self.D.flatten(), ord=order)

        # Add contributions from boundary regions if they exist
        if self.Rx is not None:
            norm_val = np.power(
                np.power(norm_val, order) + np.power(np.linalg.norm(self.Rx.flatten(), ord=order), order),
                1.0 / order
            )
        if self.Ry is not None:
            norm_val = np.power(
                np.power(norm_val, order) + np.power(np.linalg.norm(self.Ry.flatten(), ord=order), order),
                1.0 / order
            )
        if self.Rz is not None:
            norm_val = np.power(
                np.power(norm_val, order) + np.power(np.linalg.norm(self.Rz.flatten(), ord=order), order),
                1.0 / order
            )

        return norm_val


@dataclass
class VectorField:
    """
    Represents a vector field on a computational domain.

    A vector field consists of three scalar field components (x, y, z).

    Attributes:
        x: ScalarField representing x-component
        y: ScalarField representing y-component
        z: ScalarField representing z-component
    """
    x: ScalarField
    y: ScalarField
    z: ScalarField

    def zeros_like(self):
        """
        Create a new VectorField with the same structure, initialized with zeros.

        Returns:
            VectorField with zero-initialized components
        """
        return VectorField(
            x=self.x.zeros_like(),
            y=self.y.zeros_like(),
            z=self.z.zeros_like()
        )

    def __add__(self, other):
        """
        Add two VectorFields.

        Args:
            other: VectorField

        Returns:
            VectorField with component-wise sum
        """
        if not isinstance(other, VectorField):
            raise TypeError("Can only add VectorField to VectorField")

        return VectorField(
            x=self.x + other.x,
            y=self.y + other.y,
            z=self.z + other.z
        )

    def __mul__(self, scalar):
        """
        Multiply a VectorField by a scalar.

        Args:
            scalar: Scalar value or ScalarField

        Returns:
            VectorField with scaled components
        """
        return VectorField(
            x=self.x * scalar,
            y=self.y * scalar,
            z=self.z * scalar
        )

    def __rmul__(self, scalar):
        """Right multiplication (scalar * VectorField)."""
        return self.__mul__(scalar)

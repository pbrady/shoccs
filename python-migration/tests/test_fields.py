"""
Unit tests for ScalarField and VectorField classes.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from shoccs.fields import ScalarField, VectorField


class TestScalarField:
    """Tests for ScalarField class."""

    def test_zeros_creation(self):
        """Test creating a ScalarField with zeros."""
        shape = (10, 10)
        field = ScalarField.zeros(shape)

        assert field.D.shape == shape
        assert np.all(field.D == 0)
        assert field.Rx is None
        assert field.Ry is None
        assert field.Rz is None

    def test_zeros_creation_with_boundary_regions(self):
        """Test creating a ScalarField with boundary regions."""
        shape = (10, 10)
        Rx_shape = (5, 10)
        Ry_shape = (10, 5)
        Rz_shape = (10, 10)

        field = ScalarField.zeros(shape, Rx_shape=Rx_shape, Ry_shape=Ry_shape, Rz_shape=Rz_shape)

        assert field.D.shape == shape
        assert field.Rx.shape == Rx_shape
        assert field.Ry.shape == Ry_shape
        assert field.Rz.shape == Rz_shape
        assert np.all(field.D == 0)
        assert np.all(field.Rx == 0)
        assert np.all(field.Ry == 0)
        assert np.all(field.Rz == 0)

    def test_zeros_like(self):
        """Test creating a zero field with same structure."""
        D = np.ones((10, 10)) * 5
        Rx = np.ones((5, 10)) * 3
        field = ScalarField(D=D, Rx=Rx)

        zero_field = field.zeros_like()

        assert zero_field.D.shape == field.D.shape
        assert zero_field.Rx.shape == field.Rx.shape
        assert np.all(zero_field.D == 0)
        assert np.all(zero_field.Rx == 0)
        assert zero_field.Ry is None
        assert zero_field.Rz is None

    def test_copy(self):
        """Test deep copying a ScalarField."""
        D = np.array([[1, 2], [3, 4]])
        Rx = np.array([[5, 6]])
        field = ScalarField(D=D, Rx=Rx)

        field_copy = field.copy()

        # Check values are equal
        assert np.all(field_copy.D == field.D)
        assert np.all(field_copy.Rx == field.Rx)

        # Check that it's a deep copy
        field_copy.D[0, 0] = 999
        assert field.D[0, 0] == 1  # Original unchanged

    def test_addition_scalar_fields(self):
        """Test adding two ScalarFields."""
        D1 = np.array([[1, 2], [3, 4]])
        D2 = np.array([[5, 6], [7, 8]])
        field1 = ScalarField(D=D1)
        field2 = ScalarField(D=D2)

        result = field1 + field2

        expected = np.array([[6, 8], [10, 12]])
        assert np.all(result.D == expected)

    def test_addition_with_scalar(self):
        """Test adding a scalar to a ScalarField."""
        D = np.array([[1, 2], [3, 4]])
        field = ScalarField(D=D)

        result = field + 10

        expected = np.array([[11, 12], [13, 14]])
        assert np.all(result.D == expected)

    def test_addition_scalar_right(self):
        """Test right addition (scalar + field)."""
        D = np.array([[1, 2], [3, 4]])
        field = ScalarField(D=D)

        result = 10 + field

        expected = np.array([[11, 12], [13, 14]])
        assert np.all(result.D == expected)

    def test_subtraction_scalar_fields(self):
        """Test subtracting two ScalarFields."""
        D1 = np.array([[10, 20], [30, 40]])
        D2 = np.array([[1, 2], [3, 4]])
        field1 = ScalarField(D=D1)
        field2 = ScalarField(D=D2)

        result = field1 - field2

        expected = np.array([[9, 18], [27, 36]])
        assert np.all(result.D == expected)

    def test_subtraction_with_scalar(self):
        """Test subtracting a scalar from a ScalarField."""
        D = np.array([[10, 20], [30, 40]])
        field = ScalarField(D=D)

        result = field - 5

        expected = np.array([[5, 15], [25, 35]])
        assert np.all(result.D == expected)

    def test_subtraction_scalar_right(self):
        """Test right subtraction (scalar - field)."""
        D = np.array([[1, 2], [3, 4]])
        field = ScalarField(D=D)

        result = 10 - field

        expected = np.array([[9, 8], [7, 6]])
        assert np.all(result.D == expected)

    def test_multiplication_scalar_fields(self):
        """Test multiplying two ScalarFields."""
        D1 = np.array([[2, 3], [4, 5]])
        D2 = np.array([[10, 10], [10, 10]])
        field1 = ScalarField(D=D1)
        field2 = ScalarField(D=D2)

        result = field1 * field2

        expected = np.array([[20, 30], [40, 50]])
        assert np.all(result.D == expected)

    def test_multiplication_with_scalar(self):
        """Test multiplying a ScalarField by a scalar."""
        D = np.array([[1, 2], [3, 4]])
        field = ScalarField(D=D)

        result = field * 3

        expected = np.array([[3, 6], [9, 12]])
        assert np.all(result.D == expected)

    def test_multiplication_scalar_right(self):
        """Test right multiplication (scalar * field)."""
        D = np.array([[1, 2], [3, 4]])
        field = ScalarField(D=D)

        result = 3 * field

        expected = np.array([[3, 6], [9, 12]])
        assert np.all(result.D == expected)

    def test_division_scalar_fields(self):
        """Test dividing two ScalarFields."""
        D1 = np.array([[20, 30], [40, 50]])
        D2 = np.array([[2, 3], [4, 5]])
        field1 = ScalarField(D=D1)
        field2 = ScalarField(D=D2)

        result = field1 / field2

        expected = np.array([[10, 10], [10, 10]])
        assert np.all(result.D == expected)

    def test_division_with_scalar(self):
        """Test dividing a ScalarField by a scalar."""
        D = np.array([[10, 20], [30, 40]])
        field = ScalarField(D=D)

        result = field / 2

        expected = np.array([[5, 10], [15, 20]])
        assert np.all(result.D == expected)

    def test_negation(self):
        """Test negating a ScalarField."""
        D = np.array([[1, -2], [3, -4]])
        field = ScalarField(D=D)

        result = -field

        expected = np.array([[-1, 2], [-3, 4]])
        assert np.all(result.D == expected)

    def test_norm_l2(self):
        """Test L2 norm of a ScalarField."""
        D = np.array([[3, 4]])
        field = ScalarField(D=D)

        norm = field.norm(order=2)

        expected = np.sqrt(3**2 + 4**2)
        assert np.isclose(norm, expected)

    def test_norm_l1(self):
        """Test L1 norm of a ScalarField."""
        D = np.array([[3, 4]])
        field = ScalarField(D=D)

        norm = field.norm(order=1)

        expected = 3 + 4
        assert np.isclose(norm, expected)

    def test_norm_linf(self):
        """Test L-infinity norm of a ScalarField."""
        D = np.array([[3, 4, 2]])
        field = ScalarField(D=D)

        norm = field.norm(order=np.inf)

        expected = 4
        assert np.isclose(norm, expected)

    def test_norm_with_boundary_regions(self):
        """Test norm with boundary regions."""
        D = np.array([[3, 4]])
        Rx = np.array([[0, 0]])
        field = ScalarField(D=D, Rx=Rx)

        norm = field.norm(order=2)

        # Should be same as without boundary since Rx is zeros
        expected = np.sqrt(3**2 + 4**2)
        assert np.isclose(norm, expected)

    def test_operations_preserve_boundary_regions(self):
        """Test that operations preserve boundary region structure."""
        D1 = np.array([[1, 2]])
        Rx1 = np.array([[3, 4]])
        D2 = np.array([[5, 6]])
        Rx2 = np.array([[7, 8]])

        field1 = ScalarField(D=D1, Rx=Rx1)
        field2 = ScalarField(D=D2, Rx=Rx2)

        result = field1 + field2

        assert result.Rx is not None
        assert np.all(result.Rx == np.array([[10, 12]]))


class TestVectorField:
    """Tests for VectorField class."""

    def test_creation(self):
        """Test creating a VectorField."""
        x = ScalarField(D=np.ones((5, 5)))
        y = ScalarField(D=np.ones((5, 5)) * 2)
        z = ScalarField(D=np.ones((5, 5)) * 3)

        vfield = VectorField(x=x, y=y, z=z)

        assert np.all(vfield.x.D == 1)
        assert np.all(vfield.y.D == 2)
        assert np.all(vfield.z.D == 3)

    def test_zeros_like(self):
        """Test creating a zero VectorField with same structure."""
        x = ScalarField(D=np.ones((5, 5)) * 10)
        y = ScalarField(D=np.ones((5, 5)) * 20)
        z = ScalarField(D=np.ones((5, 5)) * 30)

        vfield = VectorField(x=x, y=y, z=z)
        zero_vfield = vfield.zeros_like()

        assert zero_vfield.x.D.shape == vfield.x.D.shape
        assert zero_vfield.y.D.shape == vfield.y.D.shape
        assert zero_vfield.z.D.shape == vfield.z.D.shape
        assert np.all(zero_vfield.x.D == 0)
        assert np.all(zero_vfield.y.D == 0)
        assert np.all(zero_vfield.z.D == 0)

    def test_addition(self):
        """Test adding two VectorFields."""
        x1 = ScalarField(D=np.array([[1, 2]]))
        y1 = ScalarField(D=np.array([[3, 4]]))
        z1 = ScalarField(D=np.array([[5, 6]]))
        vfield1 = VectorField(x=x1, y=y1, z=z1)

        x2 = ScalarField(D=np.array([[10, 20]]))
        y2 = ScalarField(D=np.array([[30, 40]]))
        z2 = ScalarField(D=np.array([[50, 60]]))
        vfield2 = VectorField(x=x2, y=y2, z=z2)

        result = vfield1 + vfield2

        assert np.all(result.x.D == np.array([[11, 22]]))
        assert np.all(result.y.D == np.array([[33, 44]]))
        assert np.all(result.z.D == np.array([[55, 66]]))

    def test_addition_type_error(self):
        """Test that adding non-VectorField raises TypeError."""
        x = ScalarField(D=np.array([[1, 2]]))
        y = ScalarField(D=np.array([[3, 4]]))
        z = ScalarField(D=np.array([[5, 6]]))
        vfield = VectorField(x=x, y=y, z=z)

        with pytest.raises(TypeError):
            result = vfield + 5

    def test_scalar_multiplication(self):
        """Test multiplying a VectorField by a scalar."""
        x = ScalarField(D=np.array([[1, 2]]))
        y = ScalarField(D=np.array([[3, 4]]))
        z = ScalarField(D=np.array([[5, 6]]))
        vfield = VectorField(x=x, y=y, z=z)

        result = vfield * 3

        assert np.all(result.x.D == np.array([[3, 6]]))
        assert np.all(result.y.D == np.array([[9, 12]]))
        assert np.all(result.z.D == np.array([[15, 18]]))

    def test_scalar_multiplication_right(self):
        """Test right multiplication (scalar * VectorField)."""
        x = ScalarField(D=np.array([[1, 2]]))
        y = ScalarField(D=np.array([[3, 4]]))
        z = ScalarField(D=np.array([[5, 6]]))
        vfield = VectorField(x=x, y=y, z=z)

        result = 3 * vfield

        assert np.all(result.x.D == np.array([[3, 6]]))
        assert np.all(result.y.D == np.array([[9, 12]]))
        assert np.all(result.z.D == np.array([[15, 18]]))

    def test_scalar_field_multiplication(self):
        """Test multiplying a VectorField by a ScalarField."""
        x = ScalarField(D=np.array([[1, 2]]))
        y = ScalarField(D=np.array([[3, 4]]))
        z = ScalarField(D=np.array([[5, 6]]))
        vfield = VectorField(x=x, y=y, z=z)

        scalar_field = ScalarField(D=np.array([[2, 3]]))
        result = vfield * scalar_field

        assert np.all(result.x.D == np.array([[2, 6]]))
        assert np.all(result.y.D == np.array([[6, 12]]))
        assert np.all(result.z.D == np.array([[10, 18]]))


class TestIntegration:
    """Integration tests combining multiple operations."""

    def test_complex_scalar_expression(self):
        """Test a complex expression with multiple operations."""
        D1 = np.array([[1, 2], [3, 4]])
        D2 = np.array([[5, 6], [7, 8]])
        D3 = np.array([[2, 2], [2, 2]])

        field1 = ScalarField(D=D1)
        field2 = ScalarField(D=D2)
        field3 = ScalarField(D=D3)

        # (field1 + field2) * field3 - 1
        result = (field1 + field2) * field3 - 1

        expected = np.array([[11, 15], [19, 23]])
        assert np.all(result.D == expected)

    def test_vector_field_linear_combination(self):
        """Test linear combination of vector fields."""
        x1 = ScalarField(D=np.array([[1, 0]]))
        y1 = ScalarField(D=np.array([[0, 1]]))
        z1 = ScalarField(D=np.array([[0, 0]]))
        vfield1 = VectorField(x=x1, y=y1, z=z1)

        x2 = ScalarField(D=np.array([[0, 1]]))
        y2 = ScalarField(D=np.array([[1, 0]]))
        z2 = ScalarField(D=np.array([[1, 1]]))
        vfield2 = VectorField(x=x2, y=y2, z=z2)

        # 2 * vfield1 + 3 * vfield2
        result = 2 * vfield1 + 3 * vfield2

        assert np.all(result.x.D == np.array([[2, 3]]))
        assert np.all(result.y.D == np.array([[3, 2]]))
        assert np.all(result.z.D == np.array([[3, 3]]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

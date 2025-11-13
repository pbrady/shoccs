#!/usr/bin/env python3
"""
Verification script for mesh implementation.

This script performs quick checks to ensure the mesh implementation
is working correctly.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from shoccs.geometry import CartesianMesh, BoundaryPoint


def verify_cartesian_mesh():
    """Verify CartesianMesh implementation."""
    print("Verifying CartesianMesh...")

    # Create a test mesh
    mesh = CartesianMesh(
        nx=11, ny=11, nz=11,
        xmin=0.0, xmax=1.0,
        ymin=0.0, ymax=1.0,
        zmin=0.0, zmax=1.0
    )

    # Verify attributes
    assert mesh.nx == 11, "nx mismatch"
    assert mesh.ny == 11, "ny mismatch"
    assert mesh.nz == 11, "nz mismatch"

    # Verify properties
    assert np.isclose(mesh.dx, 0.1), "dx calculation incorrect"
    assert np.isclose(mesh.dy, 0.1), "dy calculation incorrect"
    assert np.isclose(mesh.dz, 0.1), "dz calculation incorrect"
    assert mesh.shape == (11, 11, 11), "shape mismatch"

    # Verify methods
    assert mesh.size() == 11**3, "size calculation incorrect"
    assert np.isclose(mesh.volume(), 1.0), "volume calculation incorrect"

    # Verify coordinates
    x, y, z = mesh.coordinates()
    assert len(x) == 11, "x coordinate length mismatch"
    assert np.isclose(x[0], 0.0), "x start mismatch"
    assert np.isclose(x[-1], 1.0), "x end mismatch"

    print("  ✓ All CartesianMesh checks passed")
    return True


def verify_boundary_point():
    """Verify BoundaryPoint implementation."""
    print("Verifying BoundaryPoint...")

    # Create a test boundary point
    bp = BoundaryPoint(
        position=(0.5, 0.5, 0.5),
        psi=0.25,
        solid_coord=(5, 5, 5),
        shape_id=0,
        ray_outside=True
    )

    # Verify attributes
    assert bp.position == (0.5, 0.5, 0.5), "position mismatch"
    assert bp.psi == 0.25, "psi mismatch"
    assert bp.solid_coord == (5, 5, 5), "solid_coord mismatch"
    assert bp.shape_id == 0, "shape_id mismatch"
    assert bp.ray_outside is True, "ray_outside mismatch"

    print("  ✓ All BoundaryPoint checks passed")
    return True


def verify_integration():
    """Verify mesh and boundary point work together."""
    print("Verifying integration...")

    mesh = CartesianMesh(
        nx=10, ny=10, nz=10,
        xmin=0.0, xmax=1.0,
        ymin=0.0, ymax=1.0,
        zmin=0.0, zmax=1.0
    )

    # Create boundary points within mesh
    bp1 = BoundaryPoint(
        position=(0.5, 0.5, 0.5),
        psi=0.3,
        solid_coord=(5, 5, 5),
        shape_id=0,
        ray_outside=True
    )

    bp2 = BoundaryPoint(
        position=(0.25, 0.75, 0.5),
        psi=0.6,
        solid_coord=(2, 7, 5),
        shape_id=0,
        ray_outside=False
    )

    # Verify boundary points are within mesh bounds
    for bp in [bp1, bp2]:
        x, y, z = bp.position
        assert mesh.xmin <= x <= mesh.xmax, "boundary point x out of bounds"
        assert mesh.ymin <= y <= mesh.ymax, "boundary point y out of bounds"
        assert mesh.zmin <= z <= mesh.zmax, "boundary point z out of bounds"

    print("  ✓ All integration checks passed")
    return True


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Mesh Implementation Verification")
    print("=" * 60)
    print()

    try:
        verify_cartesian_mesh()
        verify_boundary_point()
        verify_integration()

        print()
        print("=" * 60)
        print("✓ ALL CHECKS PASSED")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"✗ VERIFICATION FAILED: {e}")
        print("=" * 60)
        return 1

    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ ERROR: {e}")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

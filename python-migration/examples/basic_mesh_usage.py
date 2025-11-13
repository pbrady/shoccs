"""
Basic usage example for SHOCCS mesh structures.

This example demonstrates how to create and use CartesianMesh and BoundaryPoint
objects in the SHOCCS Python framework.
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
from shoccs.geometry import CartesianMesh, BoundaryPoint


def main():
    print("=" * 60)
    print("SHOCCS Mesh Usage Example")
    print("=" * 60)
    print()

    # Create a simple 3D Cartesian mesh
    print("1. Creating a 10x10x10 Cartesian mesh from [0,1]^3")
    mesh = CartesianMesh(
        nx=10, ny=10, nz=10,
        xmin=0.0, xmax=1.0,
        ymin=0.0, ymax=1.0,
        zmin=0.0, zmax=1.0
    )

    print(f"   Mesh shape: {mesh.shape}")
    print(f"   Grid spacing: dx={mesh.dx:.6f}, dy={mesh.dy:.6f}, dz={mesh.dz:.6f}")
    print(f"   Total points: {mesh.size()}")
    print(f"   Domain volume: {mesh.volume():.6f}")
    print()

    # Generate coordinates
    print("2. Generating coordinate arrays")
    x, y, z = mesh.coordinates()
    print(f"   x-coordinates: min={x[0]:.3f}, max={x[-1]:.3f}, n={len(x)}")
    print(f"   y-coordinates: min={y[0]:.3f}, max={y[-1]:.3f}, n={len(y)}")
    print(f"   z-coordinates: min={z[0]:.3f}, max={z[-1]:.3f}, n={len(z)}")
    print()

    # Create a non-uniform mesh
    print("3. Creating a non-uniform mesh with different scales")
    mesh2 = CartesianMesh(
        nx=21, ny=11, nz=11,
        xmin=-2.0, xmax=2.0,
        ymin=-1.0, ymax=1.0,
        zmin=-1.0, zmax=1.0
    )
    print(f"   Mesh shape: {mesh2.shape}")
    print(f"   Grid spacing: dx={mesh2.dx:.6f}, dy={mesh2.dy:.6f}, dz={mesh2.dz:.6f}")
    print()

    # Create boundary points
    print("4. Creating boundary points for embedded objects")

    # Boundary point on a sphere at origin
    bp1 = BoundaryPoint(
        position=(0.5, 0.0, 0.0),
        psi=0.25,
        solid_coord=(5, 5, 5),
        shape_id=0,
        ray_outside=True
    )
    print(f"   Boundary point 1:")
    print(f"     Position: {bp1.position}")
    print(f"     Psi (distance): {bp1.psi}")
    print(f"     Solid coordinate: {bp1.solid_coord}")
    print(f"     Shape ID: {bp1.shape_id}")
    print(f"     Ray outside: {bp1.ray_outside}")
    print()

    # Another boundary point
    bp2 = BoundaryPoint(
        position=(0.3, 0.3, 0.0),
        psi=0.6,
        solid_coord=(6, 6, 5),
        shape_id=0,
        ray_outside=False
    )
    print(f"   Boundary point 2:")
    print(f"     Position: {bp2.position}")
    print(f"     Psi (distance): {bp2.psi}")
    print()

    # Demonstrate working with coordinates
    print("5. Working with mesh coordinates")
    x, y, z = mesh.coordinates()

    # Create a 3D meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    print(f"   3D meshgrid created with shape: {X.shape}")

    # Calculate distance from origin for all points
    R = np.sqrt(X**2 + Y**2 + Z**2)
    print(f"   Distance from origin: min={R.min():.6f}, max={R.max():.6f}")
    print()

    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

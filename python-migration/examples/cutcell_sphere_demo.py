"""
Demonstration of cut-cell derivative operator with embedded sphere.

This example shows how to compute derivatives on a grid with an embedded
spherical boundary using E2_1 stencils.
"""

import numpy as np
import matplotlib.pyplot as plt

from shoccs.geometry.mesh import CartesianMesh
from shoccs.geometry.shapes import Sphere
from shoccs.geometry.geometry import cast_ray_through_grid
from shoccs.operators.cutcell_builder import build_cutcell_derivative
from shoccs.operators.derivative import DerivativeOperator
from shoccs.fields.field import ScalarField


def main():
    print("=" * 70)
    print("Cut-Cell Derivative Operator Demo")
    print("=" * 70)

    # Create a 1D-like mesh (extended in x, thin in y and z)
    mesh = CartesianMesh(
        nx=41, ny=3, nz=3,
        xmin=0.0, xmax=4.0,
        ymin=0.0, ymax=0.2,
        zmin=0.0, zmax=0.2
    )

    print(f"\nMesh configuration:")
    print(f"  Grid points: {mesh.nx} x {mesh.ny} x {mesh.nz}")
    print(f"  Domain: x=[{mesh.xmin}, {mesh.xmax}]")
    print(f"  Grid spacing: h = {mesh.dx:.4f}")

    # Create embedded sphere
    sphere_center = np.array([2.0, 0.1, 0.1])
    sphere_radius = 0.5
    sphere = Sphere(center=sphere_center, radius=sphere_radius)

    print(f"\nEmbedded sphere:")
    print(f"  Center: ({sphere_center[0]:.1f}, {sphere_center[1]:.1f}, {sphere_center[2]:.1f})")
    print(f"  Radius: {sphere_radius:.2f}")

    # Find boundary intersections in x-direction
    boundary_points = cast_ray_through_grid(mesh, [sphere], direction=0)

    print(f"\nBoundary intersection:")
    print(f"  Found {len(boundary_points)} cut-cell points")
    print(f"  Boundary point details (first 5):")
    for i, bp in enumerate(boundary_points[:5]):
        print(f"    BP {i}: position=({bp.position[0]:.3f}, {bp.position[1]:.3f}, {bp.position[2]:.3f}), "
              f"psi={bp.psi:.3f}, grid={bp.solid_coord}")

    # Build cut-cell derivative operator
    print(f"\nBuilding cut-cell derivative operator...")
    A, B = build_cutcell_derivative(mesh, boundary_points, direction=0)

    print(f"  A matrix shape: {A.shape}, nnz: {A.nnz}")
    print(f"  B matrix shape: {B.shape}, nnz: {B.nnz}")

    # Create derivative operator
    op = DerivativeOperator(
        mesh=mesh,
        direction=0,
        bc_type='cutcell',
        A=A,
        B=B
    )

    # Test 1: Linear function f(x) = x
    print(f"\n" + "=" * 70)
    print("Test 1: Linear function f(x) = x")
    print("=" * 70)

    x_coords = np.linspace(mesh.xmin, mesh.xmax, mesh.nx)
    u_values = np.zeros((mesh.nx, mesh.ny, mesh.nz))
    for i in range(mesh.nx):
        u_values[i, :, :] = x_coords[i]

    # Boundary values
    boundary_values = np.zeros(len(boundary_points))
    for idx, bp in enumerate(boundary_points):
        boundary_values[idx] = bp.position[0]

    u = ScalarField(
        D=u_values,
        Rx=boundary_values,
        Ry=np.zeros(0),
        Rz=np.zeros(0)
    )

    # Apply operator
    du_dx = op(u)

    # Check accuracy
    expected = 1.0
    interior_vals = du_dx.D[:, 1, 1]  # Middle slice in y and z

    print(f"\nExpected derivative: df/dx = {expected}")
    print(f"Computed derivative at selected points:")
    for i in [5, 10, 15, 20, 25, 30, 35]:
        if i < len(x_coords):
            print(f"  x = {x_coords[i]:.2f}: df/dx = {interior_vals[i]:.4f}")

    # Test 2: Quadratic function f(x) = x^2
    print(f"\n" + "=" * 70)
    print("Test 2: Quadratic function f(x) = x^2")
    print("=" * 70)

    u_values = np.zeros((mesh.nx, mesh.ny, mesh.nz))
    for i in range(mesh.nx):
        u_values[i, :, :] = x_coords[i] ** 2

    # Boundary values
    boundary_values = np.zeros(len(boundary_points))
    for idx, bp in enumerate(boundary_points):
        boundary_values[idx] = bp.position[0] ** 2

    u = ScalarField(
        D=u_values,
        Rx=boundary_values,
        Ry=np.zeros(0),
        Rz=np.zeros(0)
    )

    # Apply operator
    du_dx = op(u)

    # Check accuracy
    interior_vals = du_dx.D[:, 1, 1]

    print(f"\nExpected derivative: df/dx = 2x")
    print(f"Computed derivative at selected points:")
    for i in [5, 10, 15, 20, 25, 30, 35]:
        if i < len(x_coords):
            expected_val = 2.0 * x_coords[i]
            error = abs(interior_vals[i] - expected_val)
            print(f"  x = {x_coords[i]:.2f}: df/dx = {interior_vals[i]:.4f}, "
                  f"expected = {expected_val:.4f}, error = {error:.4f}")

    # Test 3: Sinusoidal function
    print(f"\n" + "=" * 70)
    print("Test 3: Sinusoidal function f(x) = sin(2πx/L)")
    print("=" * 70)

    L = mesh.xmax - mesh.xmin
    u_values = np.zeros((mesh.nx, mesh.ny, mesh.nz))
    for i in range(mesh.nx):
        u_values[i, :, :] = np.sin(2 * np.pi * x_coords[i] / L)

    # Boundary values
    boundary_values = np.zeros(len(boundary_points))
    for idx, bp in enumerate(boundary_points):
        boundary_values[idx] = np.sin(2 * np.pi * bp.position[0] / L)

    u = ScalarField(
        D=u_values,
        Rx=boundary_values,
        Ry=np.zeros(0),
        Rz=np.zeros(0)
    )

    # Apply operator
    du_dx = op(u)

    # Check accuracy
    interior_vals = du_dx.D[:, 1, 1]

    print(f"\nExpected derivative: df/dx = (2π/L) * cos(2πx/L)")
    print(f"Computed derivative at selected points:")
    for i in [5, 10, 15, 20, 25, 30, 35]:
        if i < len(x_coords):
            expected_val = (2 * np.pi / L) * np.cos(2 * np.pi * x_coords[i] / L)
            error = abs(interior_vals[i] - expected_val)
            print(f"  x = {x_coords[i]:.2f}: df/dx = {interior_vals[i]:.4f}, "
                  f"expected = {expected_val:.4f}, error = {error:.4f}")

    print(f"\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

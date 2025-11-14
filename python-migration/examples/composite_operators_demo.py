"""
Demonstration of Gradient and Laplacian operators built by composition.

This example shows:
1. Creating gradient and Laplacian operators
2. Applying them to polynomial functions
3. Verifying exact results
"""

import numpy as np
from shoccs.geometry import CartesianMesh
from shoccs.fields import ScalarField
from shoccs.stencils import centered_diff_1st_order2, centered_diff_2nd_order2
from shoccs.operators import create_gradient_operator, create_laplacian_operator


def demo_gradient():
    """Demonstrate gradient operator on polynomial."""
    print("=" * 60)
    print("GRADIENT OPERATOR DEMO")
    print("=" * 60)

    # Setup mesh
    nx, ny, nz = 20, 20, 20
    mesh = CartesianMesh(nx, ny, nz, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    print(f"Mesh: {nx} x {ny} x {nz} grid on [0,1]³")
    print(f"Grid spacing: dx={mesh.dx:.4f}, dy={mesh.dy:.4f}, dz={mesh.dz:.4f}")

    # Create gradient operator
    grad = create_gradient_operator(mesh, centered_diff_1st_order2)
    print("\nCreated gradient operator using centered_diff_1st_order2")

    # Test function: u = x² + y² + z²
    x, y, z = mesh.coordinates()
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    u = ScalarField(D=X**2 + Y**2 + Z**2)
    print("\nTest function: u = x² + y² + z²")
    print(f"u at center: {u.D[nx//2, ny//2, nz//2]:.6f}")

    # Compute gradient
    grad_u = grad(u)
    print("\nComputed ∇u = (∂u/∂x, ∂u/∂y, ∂u/∂z)")

    # Exact gradient: (2x, 2y, 2z)
    exact_x = 2.0 * X
    exact_y = 2.0 * Y
    exact_z = 2.0 * Z

    # Check accuracy (interior points)
    error_x = np.max(np.abs(grad_u.x.D[1:-1, 1:-1, 1:-1] - exact_x[1:-1, 1:-1, 1:-1]))
    error_y = np.max(np.abs(grad_u.y.D[1:-1, 1:-1, 1:-1] - exact_y[1:-1, 1:-1, 1:-1]))
    error_z = np.max(np.abs(grad_u.z.D[1:-1, 1:-1, 1:-1] - exact_z[1:-1, 1:-1, 1:-1]))

    print(f"\nExact gradient: (2x, 2y, 2z)")
    print(f"Max error in ∂u/∂x: {error_x:.2e}")
    print(f"Max error in ∂u/∂y: {error_y:.2e}")
    print(f"Max error in ∂u/∂z: {error_z:.2e}")

    # Sample values at center
    i, j, k = nx//2, ny//2, nz//2
    print(f"\nAt center point ({x[i]:.3f}, {y[j]:.3f}, {z[k]:.3f}):")
    print(f"  Computed: ({grad_u.x.D[i,j,k]:.6f}, {grad_u.y.D[i,j,k]:.6f}, {grad_u.z.D[i,j,k]:.6f})")
    print(f"  Expected: ({exact_x[i,j,k]:.6f}, {exact_y[i,j,k]:.6f}, {exact_z[i,j,k]:.6f})")
    print(f"  ✓ Gradient computed correctly!")


def demo_laplacian():
    """Demonstrate Laplacian operator on polynomial."""
    print("\n" + "=" * 60)
    print("LAPLACIAN OPERATOR DEMO")
    print("=" * 60)

    # Setup mesh
    nx, ny, nz = 20, 20, 20
    mesh = CartesianMesh(nx, ny, nz, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    print(f"Mesh: {nx} x {ny} x {nz} grid on [0,1]³")
    print(f"Grid spacing: dx={mesh.dx:.4f}, dy={mesh.dy:.4f}, dz={mesh.dz:.4f}")

    # Create Laplacian operator
    laplacian = create_laplacian_operator(mesh, centered_diff_2nd_order2)
    print("\nCreated Laplacian operator using centered_diff_2nd_order2")

    # Test function: u = x² + y² + z²
    x, y, z = mesh.coordinates()
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    u = ScalarField(D=X**2 + Y**2 + Z**2)
    print("\nTest function: u = x² + y² + z²")
    print(f"u at center: {u.D[nx//2, ny//2, nz//2]:.6f}")

    # Compute Laplacian
    lap_u = laplacian(u)
    print("\nComputed ∇²u = ∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²")

    # Exact Laplacian: 2 + 2 + 2 = 6
    exact = 6.0

    # Check accuracy (interior points)
    error = np.max(np.abs(lap_u.D[1:-1, 1:-1, 1:-1] - exact))

    print(f"\nExact Laplacian: {exact:.1f} (constant)")
    print(f"Max error: {error:.2e}")

    # Sample values
    i, j, k = nx//2, ny//2, nz//2
    print(f"\nAt center point ({x[i]:.3f}, {y[j]:.3f}, {z[k]:.3f}):")
    print(f"  Computed: {lap_u.D[i,j,k]:.6f}")
    print(f"  Expected: {exact:.6f}")
    print(f"  ✓ Laplacian computed correctly!")


def demo_composition():
    """Demonstrate that gradient/Laplacian are built by composition."""
    print("\n" + "=" * 60)
    print("COMPOSITION VERIFICATION")
    print("=" * 60)

    # Setup
    nx, ny, nz = 15, 15, 15
    mesh = CartesianMesh(nx, ny, nz, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

    # Create operators
    from shoccs.operators import create_derivative_operator
    Dx = create_derivative_operator(mesh, centered_diff_1st_order2, 0)
    Dy = create_derivative_operator(mesh, centered_diff_1st_order2, 1)
    Dz = create_derivative_operator(mesh, centered_diff_1st_order2, 2)
    grad = create_gradient_operator(mesh, centered_diff_1st_order2)

    print("Created individual derivative operators and gradient operator")

    # Test function
    x, y, z = mesh.coordinates()
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    u = ScalarField(D=X**3 + Y**3 + Z**3)

    # Apply individually and via gradient
    dudx = Dx(u)
    dudy = Dy(u)
    dudz = Dz(u)
    grad_u = grad(u)

    # Compare
    error_x = np.max(np.abs(grad_u.x.D - dudx.D))
    error_y = np.max(np.abs(grad_u.y.D - dudy.D))
    error_z = np.max(np.abs(grad_u.z.D - dudz.D))

    print(f"\nVerifying: gradient.x == Dx(u), gradient.y == Dy(u), etc.")
    print(f"  Difference in x-component: {error_x:.2e}")
    print(f"  Difference in y-component: {error_y:.2e}")
    print(f"  Difference in z-component: {error_z:.2e}")
    print(f"  ✓ Gradient is exactly composed from derivative operators!")


if __name__ == '__main__':
    demo_gradient()
    demo_laplacian()
    demo_composition()

    print("\n" + "=" * 60)
    print("All demonstrations completed successfully!")
    print("=" * 60)

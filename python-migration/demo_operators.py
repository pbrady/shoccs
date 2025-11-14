"""
Demo of Phase 3: Operator Implementation

Shows how to build and use derivative operators with sparse matrices.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from shoccs.stencils import centered_diff_1st_order2
from shoccs.operators.matrix_builders import (
    build_circulant_operator,
    build_banded_matrix,
)

print("=" * 60)
print("Phase 3: Operators - Matrix Construction Demo")
print("=" * 60)

# Example 1: Derivative of x^2 with periodic BC
print("\n1. Derivative of x^2 (periodic BC)")
print("-" * 40)
h = 0.1
xmin, xmax = 0.0, 1.0
n = int((xmax - xmin) / h)
x = np.linspace(xmin, xmax - h, n)

stencil = centered_diff_1st_order2(h)
D = build_circulant_operator(stencil, n)

u = x**2
du = D @ u
du_exact = 2 * x

print(f"Grid points: {n}")
print(f"Grid spacing h: {h}")
print(f"Matrix shape: {D.shape}")
print(f"Matrix sparsity: {D.nnz} non-zeros")
print(f"\nInterior points (avoiding wrap-around):")
print(f"  x: {x[2:5]}")
print(f"  du/dx (computed): {du[2:5]}")
print(f"  du/dx (exact):    {du_exact[2:5]}")
print(f"  Error: {np.abs(du[2:5] - du_exact[2:5])}")

# Example 2: Derivative of x^2 with Dirichlet BC
print("\n2. Derivative of x^2 (Dirichlet BC)")
print("-" * 40)
n_full = int((xmax - xmin) / h) + 1
x_full = np.linspace(xmin, xmax, n_full)

D_dir = build_banded_matrix(stencil, n_full)

u_full = x_full**2
du_dir = D_dir @ u_full
du_exact_interior = 2 * x_full[1:-1]

print(f"Grid points (total): {n_full}")
print(f"Interior points: {n_full - 2}")
print(f"Matrix shape: {D_dir.shape}")
print(f"Matrix sparsity: {D_dir.nnz} non-zeros")
print(f"\nSample interior points:")
print(f"  x: {x_full[1:4]}")
print(f"  du/dx (computed): {du_dir[0:3]}")
print(f"  du/dx (exact):    {du_exact_interior[0:3]}")
print(f"  Error: {np.abs(du_dir[0:3] - du_exact_interior[0:3])}")

# Example 3: Convergence test
print("\n3. Convergence test for sin(x)")
print("-" * 40)
grid_sizes = [0.1, 0.05, 0.025]
errors = []

for h_test in grid_sizes:
    n_test = int(1.0 / h_test)
    x_test = np.linspace(0, 1 - h_test, n_test)
    
    stencil_test = centered_diff_1st_order2(h_test)
    D_test = build_circulant_operator(stencil_test, n_test)
    
    u_test = np.sin(2 * np.pi * x_test)
    du_test = D_test @ u_test
    du_exact_test = 2 * np.pi * np.cos(2 * np.pi * x_test)
    
    error = np.max(np.abs(du_test - du_exact_test))
    errors.append(error)
    
print(f"Grid size h | Max Error")
for i, (h_test, err) in enumerate(zip(grid_sizes, errors)):
    if i > 0:
        rate = np.log2(errors[i-1] / err) / np.log2(grid_sizes[i-1] / h_test)
        print(f"  {h_test:6.4f}   | {err:.6e}  (rate: {rate:.2f})")
    else:
        print(f"  {h_test:6.4f}   | {err:.6e}")

print("\nExpected convergence rate: ~2.0 (2nd order method)")
print("\n" + "=" * 60)
print("Success! Phase 3 operators are working correctly.")
print("=" * 60)

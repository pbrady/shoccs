#!/usr/bin/env python3
"""
Demonstration of ray-tracing and cut-cell geometry.

This script demonstrates the Phase 4 Part 1 implementation:
- Creating shapes (Sphere, Rectangle)
- Ray-sphere and ray-rectangle intersection
- Casting rays through a grid to find boundary points
"""

import numpy as np
from shoccs.geometry import (
    CartesianMesh, Sphere, Rectangle, Ray,
    cast_ray_through_grid, find_boundary_intersections
)


def demo_basic_ray_sphere():
    """Demonstrate basic ray-sphere intersection."""
    print("=" * 60)
    print("DEMO 1: Basic Ray-Sphere Intersection")
    print("=" * 60)

    # Create a unit sphere at origin
    sphere = Sphere(center=np.array([0.0, 0.0, 0.0]), radius=1.0)
    print(f"Sphere: center={sphere.center}, radius={sphere.radius}")

    # Create a ray from (-2, 0, 0) in +x direction
    ray = Ray(
        origin=np.array([-2.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0])
    )
    print(f"Ray: origin={ray.origin}, direction={ray.direction}")

    # Find intersection
    hit = sphere.intersect(ray)
    if hit:
        print(f"\nIntersection found:")
        print(f"  Distance (t): {hit.t:.6f}")
        print(f"  Position: {hit.position}")
        print(f"  Normal: {hit.normal}")
        print(f"  Ray from outside: {hit.ray_outside}")
    else:
        print("No intersection found")


def demo_ray_rectangle():
    """Demonstrate ray-rectangle intersection."""
    print("\n" + "=" * 60)
    print("DEMO 2: Ray-Rectangle Intersection")
    print("=" * 60)

    # Create a rectangle in XY plane at z=0
    rect = Rectangle(
        axis=2,  # Perpendicular to z-axis
        offset=0.0,
        bounds=((0.0, 1.0), (0.0, 1.0))  # x in [0,1], y in [0,1]
    )
    print(f"Rectangle: axis={rect.axis}, offset={rect.offset}")
    print(f"           bounds={rect.bounds}")

    # Create a ray from below
    ray = Ray(
        origin=np.array([0.5, 0.5, -1.0]),
        direction=np.array([0.0, 0.0, 1.0])
    )
    print(f"Ray: origin={ray.origin}, direction={ray.direction}")

    # Find intersection
    hit = rect.intersect(ray)
    if hit:
        print(f"\nIntersection found:")
        print(f"  Distance (t): {hit.t:.6f}")
        print(f"  Position: {hit.position}")
        print(f"  Normal: {hit.normal}")
    else:
        print("No intersection found")


def demo_grid_ray_casting():
    """Demonstrate ray-casting through a grid."""
    print("\n" + "=" * 60)
    print("DEMO 3: Grid Ray-Casting with Sphere")
    print("=" * 60)

    # Create a 10x10x10 grid
    mesh = CartesianMesh(
        nx=20, ny=20, nz=20,
        xmin=0.0, xmax=1.0,
        ymin=0.0, ymax=1.0,
        zmin=0.0, zmax=1.0
    )
    print(f"Mesh: {mesh.nx}x{mesh.ny}x{mesh.nz}")
    print(f"      domain=[{mesh.xmin},{mesh.xmax}] x [{mesh.ymin},{mesh.ymax}] x [{mesh.zmin},{mesh.zmax}]")
    print(f"      spacing=({mesh.dx:.4f}, {mesh.dy:.4f}, {mesh.dz:.4f})")

    # Create a sphere at center
    sphere = Sphere(center=np.array([0.5, 0.5, 0.5]), radius=0.2)
    print(f"\nSphere: center={sphere.center}, radius={sphere.radius}")

    # Cast rays in x-direction
    print("\nCasting rays in x-direction...")
    boundary_points = cast_ray_through_grid(mesh, [sphere], direction=0)

    print(f"Found {len(boundary_points)} boundary intersections")

    # Show first few boundary points
    print("\nFirst 5 boundary points:")
    for i, bp in enumerate(boundary_points[:5]):
        print(f"  {i+1}. position={bp.position}")
        print(f"     psi={bp.psi:.6f}, solid_coord={bp.solid_coord}")
        print(f"     ray_outside={bp.ray_outside}")

    # Verify psi range
    psi_values = [bp.psi for bp in boundary_points]
    print(f"\nPsi statistics:")
    print(f"  min: {min(psi_values):.6f}")
    print(f"  max: {max(psi_values):.6f}")
    print(f"  mean: {np.mean(psi_values):.6f}")


def demo_multiple_shapes():
    """Demonstrate multiple shapes."""
    print("\n" + "=" * 60)
    print("DEMO 4: Multiple Shapes")
    print("=" * 60)

    # Create mesh
    mesh = CartesianMesh(
        nx=20, ny=20, nz=20,
        xmin=0.0, xmax=1.0,
        ymin=0.0, ymax=1.0,
        zmin=0.0, zmax=1.0
    )

    # Create two spheres
    sphere1 = Sphere(center=np.array([0.3, 0.5, 0.5]), radius=0.15)
    sphere2 = Sphere(center=np.array([0.7, 0.5, 0.5]), radius=0.15)
    print(f"Sphere 1: center={sphere1.center}, radius={sphere1.radius}")
    print(f"Sphere 2: center={sphere2.center}, radius={sphere2.radius}")

    # Find all intersections in all directions
    print("\nFinding intersections in all directions...")
    intersections = find_boundary_intersections(mesh, [sphere1, sphere2])

    for direction, bps in intersections.items():
        print(f"\nDirection {direction}: {len(bps)} intersections")
        shape_counts = {}
        for bp in bps:
            shape_counts[bp.shape_id] = shape_counts.get(bp.shape_id, 0) + 1
        print(f"  Shape 0: {shape_counts.get(0, 0)} points")
        print(f"  Shape 1: {shape_counts.get(1, 0)} points")


def main():
    """Run all demonstrations."""
    print("\n")
    print("*" * 60)
    print("* Phase 4 Part 1: Shapes and Ray-Tracing Demonstration")
    print("*" * 60)

    demo_basic_ray_sphere()
    demo_ray_rectangle()
    demo_grid_ray_casting()
    demo_multiple_shapes()

    print("\n" + "*" * 60)
    print("* All demonstrations complete!")
    print("*" * 60)
    print()


if __name__ == "__main__":
    main()

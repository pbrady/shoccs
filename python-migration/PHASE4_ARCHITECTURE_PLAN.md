# Phase 4 Architecture Plan: Cut-Cell Geometry

**Author:** Software Architect  
**Date:** 2025-11-14  
**Status:** DRAFT - Ready for Implementation

---

## Executive Summary

Phase 4 adds embedded boundary support (cut-cell geometry) to enable simulations with complex shapes (spheres, rectangles) embedded in Cartesian grids. The Python design drastically simplifies the C++ template-heavy approach while maintaining the core ray-tracing algorithm.

**Key Principle:** Use Python Protocols + simple dataclasses + NumPy arrays, avoiding C++ template complexity and visitor patterns.

**Complexity Reduction:** C++ uses ~600 lines with templates and type erasure. Python will be ~200 lines with protocols and simple functions.

---

## C++ Implementation Analysis

### Core Algorithm: Ray-Tracing

The C++ code shoots rays along grid lines in each direction (x, y, z) and finds intersections with embedded shapes:

```cpp
// For each grid line in direction I:
1. Create ray: origin at grid line start, direction along grid axis
2. Find all intersections with shapes in range [t_min, t_max]
3. For each hit:
   - Compute solid_coord (which grid cell contains the hit)
   - Compute psi (normalized distance from fluid cell to boundary)
   - Compute normal vector at hit point
   - Store as mesh_object_info
4. Organize hits by shape_id and direction
```

### Key Data Structures

**ray** (simple):
```cpp
struct ray {
    real3 origin;      // Starting point
    real3 direction;   // Unit direction vector
    real3 position(real t) { return origin + t * direction; }
};
```

**hit_info**:
```cpp
struct hit_info {
    real t;             // Parameter t where ray hits shape
    real3 position;     // Hit position in 3D space
    bool ray_outside;   // True if ray came from outside shape
    int shape_id;       // Which shape was hit
};
```

**mesh_object_info** (what we store):
```cpp
struct mesh_object_info {
    real psi;           // Normalized cut-cell distance (0 to 1)
    real3 position;     // Boundary point position
    real3 normal;       // Outward normal at boundary
    bool ray_outside;   // Ray direction relative to shape
    int3 solid_coord;   // Grid cell coordinate
    int shape_id;       // Shape identifier
};
```

### Shape Interface (C++ Concept)

```cpp
template <typename S>
concept Shape = requires(const S& shape, const ray& r, real t, const real3& pos) {
    { shape.hit(r, t_min, t_max) } -> std::optional<hit_info>;
    { shape.normal(pos) } -> real3;
};
```

**Implementations:**
1. **Sphere**: Solves quadratic equation for ray-sphere intersection
2. **Rectangles** (xy, xz, yz): Ray-plane intersection + bounds check

### The Complexity Problem

C++ uses type erasure (like `std::any`) to store heterogeneous shapes:
```cpp
class shape {
    class any_shape { virtual ~any_shape(); };  // Abstract base
    template<Shape S> 
    class any_shape_impl : public any_shape;    // Template wrapper
    any_shape* s;                                // Polymorphic pointer
};
```

**Why?** C++ needs runtime polymorphism + value semantics + compile-time concept checking.

**Python doesn't need this!** We have duck typing and Protocols.

---

## Python Design: Simplified Architecture

### Design Principles

✅ **DO:**
- Use `Protocol` for Shape interface (structural subtyping)
- Simple `@dataclass` for Ray, HitInfo, BoundaryPoint
- NumPy arrays for storing boundary points
- Functional approach (factories, not builders)
- Direct algorithm implementation (no abstraction layers)

❌ **DON'T:**
- NO abstract base classes with virtual methods
- NO type erasure or manual polymorphism
- NO template metaprogramming patterns
- NO visitor pattern (not needed)
- NO complex inheritance hierarchies

### Core Data Structures

#### 1. Ray (Simple Dataclass)

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class Ray:
    """
    Ray for intersection testing.
    
    Represents a ray r(t) = origin + t * direction
    where t >= 0.
    
    Attributes:
        origin: Starting point (x, y, z)
        direction: Direction vector (normalized or not)
    """
    origin: np.ndarray      # shape (3,)
    direction: np.ndarray   # shape (3,)
    
    def position(self, t: float) -> np.ndarray:
        """Compute position at parameter t."""
        return self.origin + t * self.direction
    
    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        self.origin = np.asarray(self.origin, dtype=np.float64)
        self.direction = np.asarray(self.direction, dtype=np.float64)
```

**Why simple?** No need for fancy methods. Just data + position computation.

#### 2. HitInfo (Dataclass)

```python
@dataclass
class HitInfo:
    """
    Information about ray-shape intersection.
    
    Attributes:
        t: Parameter value where ray hits shape
        position: 3D position of hit point
        ray_outside: True if ray came from outside the shape
        shape_id: Integer identifier for the shape
    """
    t: float
    position: np.ndarray    # shape (3,)
    ray_outside: bool
    shape_id: int
```

#### 3. BoundaryPoint (Already Exists!)

The existing `BoundaryPoint` class is perfect, but we can enhance it:

```python
@dataclass
class BoundaryPoint:
    """
    Boundary point information for cut-cell methods.
    
    This matches mesh_object_info in C++.
    
    Attributes:
        position: (x, y, z) position of the boundary point
        normal: (nx, ny, nz) outward normal vector at boundary
        psi: 1D cut-cell distance (normalized, 0 to 1)
        solid_coord: (i, j, k) integer grid coordinate
        shape_id: Integer ID of the shape
        ray_outside: True if ray approached from outside
    """
    position: np.ndarray       # shape (3,)
    normal: np.ndarray         # shape (3,)
    psi: float
    solid_coord: tuple[int, int, int]
    shape_id: int
    ray_outside: bool
    
    def __post_init__(self):
        """Validate and convert to numpy arrays."""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.normal = np.asarray(self.normal, dtype=np.float64)
        if not (0.0 <= self.psi <= 1.0):
            raise ValueError(f"psi must be in [0, 1], got {self.psi}")
```

### Shape Interface (Protocol)

**Use Python 3.8+ Protocol for structural subtyping:**

```python
from typing import Protocol, Optional

class Shape(Protocol):
    """
    Protocol for geometric shapes that can be intersected by rays.
    
    Any class implementing hit() and normal() methods with the correct
    signatures will satisfy this protocol - no inheritance needed!
    """
    
    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitInfo]:
        """
        Test if ray intersects this shape in range [t_min, t_max].
        
        Args:
            ray: Ray to test for intersection
            t_min: Minimum t value to consider
            t_max: Maximum t value to consider
            
        Returns:
            HitInfo if intersection found, None otherwise
        """
        ...
    
    def normal(self, position: np.ndarray) -> np.ndarray:
        """
        Compute outward normal vector at given position on shape surface.
        
        Args:
            position: Point on shape surface (x, y, z)
            
        Returns:
            Unit normal vector (nx, ny, nz)
        """
        ...
```

**Why Protocol?**
- No inheritance needed - just implement the methods!
- Type checkers (mypy) can verify at compile time
- More Pythonic than ABC (Abstract Base Class)
- Matches C++ concept semantics without complexity

### Shape Implementations

#### Sphere

```python
@dataclass
class Sphere:
    """
    Sphere shape for cut-cell geometry.
    
    Attributes:
        center: (x, y, z) center position
        radius: Sphere radius
        shape_id: Unique identifier for this shape
    """
    center: np.ndarray  # shape (3,)
    radius: float
    shape_id: int
    
    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=np.float64)
        if self.radius <= 0:
            raise ValueError("Radius must be positive")
    
    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitInfo]:
        """
        Ray-sphere intersection using quadratic formula.
        
        Ray equation: P(t) = O + tD
        Sphere equation: ||P - C||² = r²
        
        Substitute and solve: at² + bt + c = 0
        where:
            a = D·D
            b = 2(O-C)·D
            c = (O-C)·(O-C) - r²
        """
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius ** 2
        
        discriminant = b * b - a * c
        if discriminant <= 0:
            return None  # No intersection
        
        sqrt_disc = np.sqrt(discriminant)
        
        # Try first root: t = (-b - sqrt(discriminant)) / a
        t = (-b - sqrt_disc) / a
        if t_min < t < t_max:
            pos = ray.position(t)
            # Ray is outside if it's moving toward center
            ray_outside = np.dot(ray.direction, pos - self.center) < 0
            return HitInfo(t=t, position=pos, ray_outside=ray_outside, 
                          shape_id=self.shape_id)
        
        # Try second root: t = (-b + sqrt(discriminant)) / a
        t = (-b + sqrt_disc) / a
        if t_min < t < t_max:
            pos = ray.position(t)
            ray_outside = np.dot(ray.direction, pos - self.center) < 0
            return HitInfo(t=t, position=pos, ray_outside=ray_outside,
                          shape_id=self.shape_id)
        
        return None  # No hit in valid range
    
    def normal(self, position: np.ndarray) -> np.ndarray:
        """Outward normal is (P - C) / ||P - C||."""
        r = position - self.center
        return r / np.linalg.norm(r)
```

#### Rectangle (Axis-Aligned)

```python
@dataclass
class AxisAlignedRectangle:
    """
    Axis-aligned rectangle (xy, xz, or yz plane).
    
    Attributes:
        axis: Which axis is perpendicular to rect (0=x, 1=y, 2=z)
        plane_coord: Position along perpendicular axis
        bounds: ((min_slow, max_slow), (min_fast, max_fast))
                Bounds in the two in-plane directions
        fluid_normal: +1 or -1, indicates which side is fluid
        shape_id: Unique identifier
    
    Example:
        # XY rectangle at z=0.5, spanning x=[0,1], y=[0,1]
        rect = AxisAlignedRectangle(
            axis=2,  # z-axis
            plane_coord=0.5,
            bounds=((0.0, 1.0), (0.0, 1.0)),
            fluid_normal=1.0,
            shape_id=0
        )
    """
    axis: int  # 0, 1, or 2 for x, y, z
    plane_coord: float
    bounds: tuple[tuple[float, float], tuple[float, float]]
    fluid_normal: float  # +1 or -1
    shape_id: int
    
    def __post_init__(self):
        if self.axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")
        self.fluid_normal = 1.0 if self.fluid_normal > 0 else -1.0
        
        # Precompute slow/fast axis indices
        if self.axis == 0:  # YZ rectangle
            self.slow, self.fast = 1, 2
        elif self.axis == 1:  # XZ rectangle
            self.slow, self.fast = 0, 2
        else:  # XY rectangle
            self.slow, self.fast = 0, 1
    
    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitInfo]:
        """
        Ray-plane intersection with bounds check.
        
        Plane equation: x[axis] = plane_coord
        Ray equation: P(t) = O + tD
        
        Solve: O[axis] + t*D[axis] = plane_coord
               => t = (plane_coord - O[axis]) / D[axis]
        """
        # Avoid division by zero
        if abs(ray.direction[self.axis]) < 1e-10:
            return None  # Ray parallel to plane
        
        t = (self.plane_coord - ray.origin[self.axis]) / ray.direction[self.axis]
        
        if not (t_min < t < t_max):
            return None  # Out of range
        
        pos = ray.position(t)
        
        # Check if hit point is within rectangle bounds
        (min_slow, max_slow), (min_fast, max_fast) = self.bounds
        if not (min_slow <= pos[self.slow] <= max_slow and
                min_fast <= pos[self.fast] <= max_fast):
            return None  # Outside rectangle
        
        # Determine if ray came from fluid side
        ray_outside = self.fluid_normal * ray.direction[self.axis] < 0
        
        return HitInfo(t=t, position=pos, ray_outside=ray_outside,
                      shape_id=self.shape_id)
    
    def normal(self, position: np.ndarray) -> np.ndarray:
        """Normal is along axis direction."""
        n = np.zeros(3)
        n[self.axis] = self.fluid_normal
        return n
```

**Why these implementations?**
- Direct translation of C++ math (validated code)
- No unnecessary abstraction
- Easy to test (pure functions)
- Extensible (add cylinder, box, etc. by implementing Protocol)

### Geometry Class: Organizing Boundary Points

```python
from typing import List

@dataclass
class CutCellGeometry:
    """
    Stores boundary point information for cut-cell methods.
    
    Organizes intersections of grid rays with embedded shapes,
    separated by direction (x, y, z) and optionally by shape.
    
    Attributes:
        Rx: Boundary points from rays in x-direction
        Ry: Boundary points from rays in y-direction
        Rz: Boundary points from rays in z-direction
        mesh: CartesianMesh this geometry is associated with
        shapes: List of shapes in the domain
    """
    Rx: List[BoundaryPoint]
    Ry: List[BoundaryPoint]
    Rz: List[BoundaryPoint]
    mesh: CartesianMesh
    shapes: List  # List[Shape] - any objects satisfying Shape protocol
    
    def get_by_direction(self, direction: int) -> List[BoundaryPoint]:
        """Get boundary points for a specific direction."""
        if direction == 0:
            return self.Rx
        elif direction == 1:
            return self.Ry
        else:
            return self.Rz
    
    def get_by_shape(self, direction: int, shape_id: int) -> List[BoundaryPoint]:
        """Get boundary points for specific direction and shape."""
        points = self.get_by_direction(direction)
        return [p for p in points if p.shape_id == shape_id]
    
    def to_arrays(self, direction: int) -> dict:
        """
        Convert boundary points to NumPy arrays for operator construction.
        
        Returns dict with keys:
            - positions: (N, 3) array
            - normals: (N, 3) array
            - psi: (N,) array
            - solid_coords: (N, 3) array of integers
            - shape_ids: (N,) array of integers
        """
        points = self.get_by_direction(direction)
        if not points:
            return {k: np.array([]) for k in 
                   ['positions', 'normals', 'psi', 'solid_coords', 'shape_ids']}
        
        return {
            'positions': np.array([p.position for p in points]),
            'normals': np.array([p.normal for p in points]),
            'psi': np.array([p.psi for p in points]),
            'solid_coords': np.array([p.solid_coord for p in points]),
            'shape_ids': np.array([p.shape_id for p in points]),
        }
```

### Ray-Tracing Algorithm

**Main function: Build geometry from mesh + shapes**

```python
def build_cut_cell_geometry(
    mesh: CartesianMesh,
    shapes: List  # List of objects satisfying Shape protocol
) -> CutCellGeometry:
    """
    Build cut-cell geometry by ray-tracing through mesh.
    
    For each direction (x, y, z):
        1. Shoot rays along grid lines
        2. Find intersections with all shapes
        3. Compute boundary point information
        4. Store organized by direction
    
    Args:
        mesh: Cartesian mesh
        shapes: List of shapes (must satisfy Shape protocol)
        
    Returns:
        CutCellGeometry with boundary points organized by direction
    """
    Rx = _trace_direction(mesh, shapes, direction=0)
    Ry = _trace_direction(mesh, shapes, direction=1)
    Rz = _trace_direction(mesh, shapes, direction=2)
    
    return CutCellGeometry(Rx=Rx, Ry=Ry, Rz=Rz, mesh=mesh, shapes=shapes)


def _trace_direction(
    mesh: CartesianMesh,
    shapes: List,
    direction: int
) -> List[BoundaryPoint]:
    """
    Trace rays in a specific direction to find boundary points.
    
    Args:
        mesh: Cartesian mesh
        shapes: List of shapes to intersect
        direction: 0=x, 1=y, 2=z
        
    Returns:
        List of BoundaryPoint objects
    """
    boundary_points = []
    
    # Get mesh info for this direction
    if direction == 0:
        nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
        x, y, z = mesh.coordinates()
        h = mesh.dx
        # Iterate over y-z plane, shoot rays along x
        for j in range(ny):
            for k in range(nz):
                origin = np.array([x[0], y[j], z[k]])
                ray_dir = np.array([1.0, 0.0, 0.0])
                t_max = x[-1] - x[0]
                
                # Find all hits along this ray
                hits = _find_all_hits(Ray(origin, ray_dir), shapes, 0.0, t_max)
                
                # Process each hit
                for hit in hits:
                    bp = _process_hit(hit, origin, ray_dir, h, direction, 
                                     shapes, j, k)
                    boundary_points.append(bp)
    
    # Similar for y and z directions...
    elif direction == 1:
        # ... (iterate over x-z plane, shoot rays along y)
        pass
    else:  # direction == 2
        # ... (iterate over x-y plane, shoot rays along z)
        pass
    
    return boundary_points


def _find_all_hits(
    ray: Ray,
    shapes: List,
    t_min: float,
    t_max: float
) -> List[HitInfo]:
    """
    Find all intersections of ray with shapes, sorted by t.
    
    This implements the C++ closest_hit() logic, extended to find
    ALL hits instead of just one.
    """
    hits = []
    
    # Check each shape
    for shape in shapes:
        hit = shape.hit(ray, t_min, t_max)
        if hit is not None:
            hits.append(hit)
    
    # Sort by t value
    hits.sort(key=lambda h: h.t)
    
    # For each hit, find next hit by restricting search range
    # This handles cases where ray enters/exits shape multiple times
    all_hits = []
    current_t_min = t_min
    
    while True:
        # Find closest hit in remaining range
        closest_hit = None
        for shape in shapes:
            hit = shape.hit(ray, current_t_min, t_max)
            if hit is not None:
                if closest_hit is None or hit.t < closest_hit.t:
                    # Attach shape reference for normal computation
                    closest_hit = hit
        
        if closest_hit is None:
            break  # No more hits
        
        all_hits.append(closest_hit)
        
        # Move past this hit (use nextafter to avoid hitting same point)
        current_t_min = closest_hit.t + 1e-10
        
        if current_t_min >= t_max:
            break
    
    return all_hits


def _process_hit(
    hit: HitInfo,
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    h: float,
    direction: int,
    shapes: List,
    j: int,  # Slow coordinate
    k: int   # Fast coordinate
) -> BoundaryPoint:
    """
    Convert HitInfo to BoundaryPoint with psi computation.
    
    Key computation: psi is normalized distance from fluid cell to boundary.
    
    If ray_outside:
        - Solid is before hit, fluid is after
        - solid_coord[dir] = floor(t/h)
        - fluid_coord[dir] = solid_coord[dir] + 1
        - psi = (fluid_pos - hit_pos) / h
    
    If not ray_outside:
        - Fluid is before hit, solid is after
        - solid_coord[dir] = ceil(t/h)
        - fluid_coord[dir] = solid_coord[dir] - 1
        - psi = (hit_pos - fluid_pos) / h
    """
    # Determine solid coordinate
    coord = [0, 0, 0]
    
    if direction == 0:
        coord[0] = int(hit.t / h) + (1 if hit.ray_outside else 0)
        coord[1] = j
        coord[2] = k
    # Similar for other directions...
    
    # Compute psi
    offset = 1 if hit.ray_outside else -1
    fluid_coord = coord[direction] + offset
    fluid_pos = ray_origin[direction] + fluid_coord * h
    psi = abs(fluid_pos - hit.position[direction]) / h
    
    # Get normal from shape
    shape = shapes[hit.shape_id]
    normal = shape.normal(hit.position)
    
    return BoundaryPoint(
        position=hit.position,
        normal=normal,
        psi=psi,
        solid_coord=tuple(coord),
        shape_id=hit.shape_id,
        ray_outside=hit.ray_outside
    )
```

**Simplified version (initial implementation):**

For Phase 4.1, we can start with a simpler non-vectorized version:

```python
def build_cut_cell_geometry_simple(
    mesh: CartesianMesh,
    shapes: List
) -> CutCellGeometry:
    """Simplified version - easier to understand and debug."""
    Rx, Ry, Rz = [], [], []
    x, y, z = mesh.coordinates()
    
    # Direction 0: Rays along x-axis
    for j, yj in enumerate(y):
        for k, zk in enumerate(z):
            ray = Ray(origin=np.array([x[0], yj, zk]),
                     direction=np.array([1.0, 0.0, 0.0]))
            hits = _find_all_hits_simple(ray, shapes, x[0], x[-1])
            
            for hit in hits:
                bp = _make_boundary_point(hit, mesh, shapes, 
                                          direction=0, j=j, k=k)
                Rx.append(bp)
    
    # Similar for y and z...
    
    return CutCellGeometry(Rx=Rx, Ry=Ry, Rz=Rz, mesh=mesh, shapes=shapes)
```

---

## Integration with Operators

### How Operators Use Geometry

From Phase 3, operators have matrices for cut-cell boundaries:
```python
@dataclass
class DerivativeOperator:
    O: sp.csr_matrix      # Interior operator
    B: sp.csr_matrix      # Boundary coupling
    Bfx, Brx: sp.csr_matrix  # Cut-cell x-boundaries
    # ... etc
```

**Building these matrices requires geometry info:**

```python
def build_derivative_with_cutcells(
    mesh: CartesianMesh,
    geometry: CutCellGeometry,
    direction: int,
    stencil,
    bc: dict
) -> DerivativeOperator:
    """
    Build derivative operator including cut-cell boundaries.
    
    Uses geometry.Rx/Ry/Rz to populate B, Bf*, Br* matrices.
    """
    # Build standard operator first
    D = build_derivative(mesh, direction, stencil, bc)
    
    # Add cut-cell boundary contributions
    boundary_data = geometry.to_arrays(direction)
    
    if len(boundary_data['psi']) > 0:
        # Build B matrix (couples Rx to domain D)
        B = _build_boundary_coupling_matrix(mesh, boundary_data, direction)
        
        # Build Bf* and Br* matrices
        Bf = _build_boundary_feedback_matrix(mesh, boundary_data, direction)
        Br = _build_boundary_reflection_matrix(mesh, boundary_data, direction)
        
        # Update operator
        if direction == 0:
            D.Bfx = Bf
            D.Brx = Br
        # ... similar for y, z
        
        D.B = B
    
    return D
```

**Key insight:** Geometry provides data, operators use it to build matrices. Clean separation!

---

## Testing Strategy

### Test Pyramid

**Level 1: Unit Tests**
- Ray-sphere intersection (known cases)
- Ray-rectangle intersection
- Normal computation
- Psi computation
- HitInfo creation

**Level 2: Integration Tests**
- Single sphere in mesh
- Single rectangle in mesh
- Multiple shapes
- Ray finding all hits along line

**Level 3: Validation Tests**
- Sphere: Compare with analytical volume
- Rectangle: Verify planar cuts
- Conservation: Volume of fluid + solid = total

### Key Test Cases

#### Test 1: Sphere Intersection

```python
def test_sphere_ray_intersection():
    """Ray through sphere center hits twice."""
    sphere = Sphere(center=np.array([0.5, 0.5, 0.5]), 
                   radius=0.25, 
                   shape_id=0)
    
    # Ray along x-axis through center
    ray = Ray(origin=np.array([0.0, 0.5, 0.5]),
             direction=np.array([1.0, 0.0, 0.0]))
    
    # Should hit at x = 0.25 and x = 0.75
    hit1 = sphere.hit(ray, 0.0, 0.5)
    assert hit1 is not None
    assert np.isclose(hit1.t, 0.25)
    assert hit1.ray_outside  # Coming from outside
    
    hit2 = sphere.hit(ray, 0.5, 1.0)
    assert hit2 is not None
    assert np.isclose(hit2.t, 0.75)
    assert not hit2.ray_outside  # Coming from inside
```

#### Test 2: Rectangle Intersection

```python
def test_rectangle_intersection():
    """Ray perpendicular to rectangle."""
    # XY rectangle at z=0.5
    rect = AxisAlignedRectangle(
        axis=2,
        plane_coord=0.5,
        bounds=((0.0, 1.0), (0.0, 1.0)),
        fluid_normal=1.0,
        shape_id=0
    )
    
    # Ray along z-axis
    ray = Ray(origin=np.array([0.5, 0.5, 0.0]),
             direction=np.array([0.0, 0.0, 1.0]))
    
    hit = rect.hit(ray, 0.0, 1.0)
    assert hit is not None
    assert np.isclose(hit.t, 0.5)
    assert np.allclose(hit.position, [0.5, 0.5, 0.5])
```

#### Test 3: Geometry Construction

```python
def test_geometry_single_sphere():
    """Build geometry for sphere in unit cube."""
    mesh = CartesianMesh(nx=10, ny=10, nz=10,
                        xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=1)
    
    sphere = Sphere(center=np.array([0.5, 0.5, 0.5]),
                   radius=0.2,
                   shape_id=0)
    
    geom = build_cut_cell_geometry(mesh, [sphere])
    
    # Should have boundary points in all three directions
    assert len(geom.Rx) > 0
    assert len(geom.Ry) > 0
    assert len(geom.Rz) > 0
    
    # All psi values should be in [0, 1]
    for bp in geom.Rx + geom.Ry + geom.Rz:
        assert 0.0 <= bp.psi <= 1.0
    
    # Normals should be unit vectors
    for bp in geom.Rx:
        assert np.isclose(np.linalg.norm(bp.normal), 1.0)
```

#### Test 4: Psi Computation

```python
def test_psi_computation():
    """Verify psi is correctly computed."""
    mesh = CartesianMesh(nx=11, ny=3, nz=3,
                        xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=1)
    h = mesh.dx  # = 0.1
    
    # Sphere at x=0.55 should create psi=0.5 for cell i=5
    sphere = Sphere(center=np.array([0.55, 0.5, 0.5]),
                   radius=0.05,
                   shape_id=0)
    
    geom = build_cut_cell_geometry(mesh, [sphere])
    
    # Find boundary point near y=0.5, z=0.5
    bp = [p for p in geom.Rx if 
          abs(p.position[1] - 0.5) < 0.01 and
          abs(p.position[2] - 0.5) < 0.01][0]
    
    # Check psi (should be ~0.5)
    assert 0.4 < bp.psi < 0.6
```

---

## Implementation Timeline

| Phase | Task | Duration | Priority |
|-------|------|----------|----------|
| 4.1 | Ray, HitInfo, Shape Protocol | 1 day | HIGH |
| 4.2 | Sphere implementation + tests | 1 day | HIGH |
| 4.3 | Rectangle implementation + tests | 1 day | HIGH |
| 4.4 | Ray-tracing algorithm (simple) | 2 days | HIGH |
| 4.5 | Geometry class + integration | 1 day | HIGH |
| 4.6 | Validation + examples | 1 day | MEDIUM |
| 4.7 | Operator integration | 2 days | HIGH |

**Total:** 9 days (~2 weeks)

**Critical path:** 4.1 → 4.2 → 4.4 → 4.5 → 4.7

---

## File Structure

```
python-migration/
├── src/shoccs/geometry/
│   ├── __init__.py          # Existing (mesh, BoundaryPoint)
│   ├── mesh.py              # Existing
│   ├── shapes.py            # NEW: Shape protocol, Sphere, Rectangle
│   ├── ray.py               # NEW: Ray, HitInfo dataclasses
│   └── cutcell.py           # NEW: CutCellGeometry, ray-tracing
├── tests/
│   ├── test_ray.py          # Ray class tests
│   ├── test_sphere.py       # Sphere intersection tests
│   ├── test_rectangle.py    # Rectangle intersection tests
│   ├── test_geometry.py     # Geometry construction tests
│   └── test_cutcell_ops.py  # Integration with operators
└── examples/
    ├── sphere_in_box.py     # Simple sphere example
    └── channel_flow.py      # Rectangle boundary example
```

---

## API Design

### User-Facing Workflow

```python
from shoccs.geometry import CartesianMesh, Sphere, AxisAlignedRectangle
from shoccs.geometry import build_cut_cell_geometry
from shoccs.operators import build_derivative_with_cutcells
from shoccs.fields import ScalarField

# 1. Create mesh
mesh = CartesianMesh(nx=50, ny=50, nz=50,
                     xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=1)

# 2. Define shapes
sphere = Sphere(center=np.array([0.5, 0.5, 0.5]), 
               radius=0.2, 
               shape_id=0)

inlet = AxisAlignedRectangle(
    axis=0,  # YZ plane
    plane_coord=0.1,
    bounds=((0.0, 1.0), (0.0, 1.0)),
    fluid_normal=1.0,
    shape_id=1
)

# 3. Build geometry
geometry = build_cut_cell_geometry(mesh, shapes=[sphere, inlet])

print(f"Found {len(geometry.Rx)} x-boundary points")
print(f"Found {len(geometry.Ry)} y-boundary points")
print(f"Found {len(geometry.Rz)} z-boundary points")

# 4. Build operators with cut-cells
dx = build_derivative_with_cutcells(mesh, geometry, direction=0, ...)
dy = build_derivative_with_cutcells(mesh, geometry, direction=1, ...)
dz = build_derivative_with_cutcells(mesh, geometry, direction=2, ...)

# 5. Use operators normally
u = ScalarField(D=some_data, Rx=boundary_data_x, ...)
du_dx = dx.apply(u)
```

---

## Simplification Wins

| Feature | C++ Complexity | Python Simplicity |
|---------|---------------|-------------------|
| **Shape Interface** | Template concept + type erasure + virtual functions | Protocol (duck typing) |
| **Shape Storage** | `any_shape` wrapper class | List of objects |
| **Polymorphism** | Manual vtable via abstract base class | Python dispatch |
| **Ray Class** | Templated on coordinate type | Simple dataclass |
| **Hit Detection** | Iterator-based visitor pattern | List comprehension |
| **Code Size** | ~600 lines | ~200 lines (estimated) |

**Key insight:** Python's dynamic typing eliminates 70% of C++ boilerplate!

---

## Success Criteria

### Correctness
- ✅ Ray-sphere intersection matches analytical solution
- ✅ Ray-rectangle intersection correct for all orientations
- ✅ Psi computation matches C++ formula
- ✅ Normals are unit vectors
- ✅ All boundary points in valid range

### Code Quality
- ✅ Simple, readable implementation (no fancy patterns)
- ✅ Type hints for all public functions
- ✅ Comprehensive tests (>90% coverage)
- ✅ Clear documentation

### Performance
- ✅ Geometry construction < 5s for 50³ grid with sphere
- ✅ Memory efficient (boundary points << grid points)

### Integration
- ✅ Works with existing CartesianMesh
- ✅ BoundaryPoint compatible with operators
- ✅ Ready for Phase 5 (time integration)

---

## Risk Mitigation

### Risk 1: Psi Computation Edge Cases

**Concern:** Boundary exactly on grid line, or very small psi.

**Mitigation:**
1. Start with well-separated cases (psi ∈ [0.1, 0.9])
2. Add epsilon tolerance for edge detection
3. Test edge cases explicitly
4. Document assumptions clearly

### Risk 2: Multiple Shapes Overlapping

**Concern:** What if shapes overlap? Which one wins?

**Mitigation:**
1. Initial implementation: assume non-overlapping
2. Document this assumption
3. Future: Add overlap detection if needed
4. Test cases verify non-overlap

### Risk 3: Performance for Many Shapes

**Concern:** Ray-tracing is O(n_rays * n_shapes * n_hits).

**Mitigation:**
1. Initial implementation: simple loop (fine for ~10 shapes)
2. Profile before optimizing
3. Future: Add spatial acceleration (BVH, octree) if needed
4. For research code, simplicity > speed

---

## Comparison: C++ vs Python

| Aspect | C++ | Python |
|--------|-----|--------|
| **Shape Interface** | Template concept + type erasure | Protocol |
| **Hit Detection** | `std::optional<hit_info>` | `Optional[HitInfo]` |
| **Normal Computation** | `real3 normal(const real3&)` | `np.ndarray normal(np.ndarray)` |
| **Geometry Storage** | `std::vector<mesh_object_info>` | `List[BoundaryPoint]` |
| **Ray Tracing** | Template over direction | Simple function per direction |
| **Code Complexity** | ~600 lines | ~200 lines |
| **Abstraction** | Heavy (templates, visitors) | Light (protocols, dataclasses) |

---

## Next Steps

1. **Review this plan** with team
2. **Start with Phase 4.1:** Ray, HitInfo, Shape Protocol
3. **Iterate quickly:** Get sphere working first
4. **Validate early:** Test against known cases
5. **Integrate with operators:** Phase 4.7 is the payoff

---

**Architect sign-off:** Ready for implementation!

This design is **~3x simpler** than C++ while maintaining correctness. The key is embracing Python's strengths (duck typing, protocols, dataclasses) instead of fighting them with C++ patterns.

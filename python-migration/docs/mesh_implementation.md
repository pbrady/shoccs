# Mesh Implementation Documentation

## Overview

This document describes the mesh and geometry structures implemented for the SHOCCS Python migration.

## Implemented Classes

### CartesianMesh

A dataclass representing a uniform 3D Cartesian grid for computational domains.

**File**: `/home/user/shoccs/python-migration/src/shoccs/geometry/mesh.py`

#### Attributes

- `nx`, `ny`, `nz`: Number of grid points in each direction (int)
- `xmin`, `xmax`: Domain bounds in x-direction (float)
- `ymin`, `ymax`: Domain bounds in y-direction (float)
- `zmin`, `zmax`: Domain bounds in z-direction (float)

#### Properties

- `dx`, `dy`, `dz`: Grid spacing in each direction
- `shape`: Tuple of (nx, ny, nz)

#### Methods

- `coordinates()`: Returns (x, y, z) coordinate arrays as numpy arrays
- `size()`: Returns total number of grid points
- `volume()`: Returns total domain volume

#### Validation

- Grid dimensions must be positive
- Min bounds must be less than max bounds for each dimension

#### Example Usage

```python
from shoccs.geometry import CartesianMesh

# Create a 10x10x10 mesh on unit cube
mesh = CartesianMesh(
    nx=10, ny=10, nz=10,
    xmin=0.0, xmax=1.0,
    ymin=0.0, ymax=1.0,
    zmin=0.0, zmax=1.0
)

# Access properties
print(f"Grid spacing: {mesh.dx}")
print(f"Total points: {mesh.size()}")

# Get coordinates
x, y, z = mesh.coordinates()
```

### BoundaryPoint

A dataclass representing boundary point information for cut-cell methods.

**File**: `/home/user/shoccs/python-migration/src/shoccs/geometry/mesh.py`

#### Attributes

- `position`: Tuple of (x, y, z) coordinates (float, float, float)
- `psi`: 1D cut-cell distance, normalized [0, 1] (float)
- `solid_coord`: Tuple of (i, j, k) integer grid coordinates (int, int, int)
- `shape_id`: Integer ID of the shape (int)
- `ray_outside`: Boolean indicating if ray is outside object (bool)

#### Validation

- Position must be a 3-tuple
- Solid coordinate must be a 3-tuple
- Psi must be in range [0, 1]

#### Example Usage

```python
from shoccs.geometry import BoundaryPoint

# Create a boundary point
bp = BoundaryPoint(
    position=(0.5, 0.5, 0.5),
    psi=0.25,
    solid_coord=(5, 5, 5),
    shape_id=0,
    ray_outside=True
)

print(f"Position: {bp.position}")
print(f"Distance: {bp.psi}")
```

## Testing

Comprehensive unit tests are provided in `/home/user/shoccs/python-migration/tests/test_mesh.py`

### Test Coverage

#### CartesianMesh Tests (11 tests)
- Basic mesh creation
- Shape property
- Grid spacing calculation (uniform and non-uniform)
- Coordinate generation
- Coordinate value validation
- Mesh size and volume
- Invalid dimension handling
- Invalid bounds handling
- Coordinate monotonicity

#### BoundaryPoint Tests (6 tests)
- Basic boundary point creation
- Attribute access
- Psi range validation
- Invalid position handling
- Invalid solid coordinate handling
- Negative coordinate support

#### Integration Tests (2 tests)
- Mesh and boundary point interaction
- Multiple boundary points

### Running Tests

```bash
cd /home/user/shoccs/python-migration
python -m pytest tests/test_mesh.py -v
```

All 19 tests pass successfully.

## Design Decisions

1. **Dataclasses**: Used Python dataclasses for simplicity and clean syntax
2. **NumPy Integration**: Coordinate generation uses NumPy's `linspace` for accuracy
3. **Validation**: Input validation in `__post_init__` methods
4. **Type Hints**: Full type annotations for better IDE support
5. **Documentation**: Comprehensive docstrings with examples

## Future Enhancements

The current implementation provides basic mesh structures. Future additions may include:

- Ray tracing functionality for boundary detection
- Object geometry integration
- Advanced boundary condition handling
- Mesh refinement capabilities
- Parallel processing with Numba JIT compilation

## Compatibility

This implementation is compatible with the C++ SHOCCS codebase structures:

- `CartesianMesh` corresponds to C++ `cartesian` class
- `BoundaryPoint` corresponds to C++ `mesh_object_info` struct

## Dependencies

- Python 3.11+
- NumPy >= 1.20
- pytest (for testing)

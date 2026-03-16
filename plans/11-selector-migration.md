# Phase 11: Selector Migration to Handle Patterns

**Goal:** Replace the iterator-based `plane_view`, `multi_slice_view`, `predicate_view`, and `optional_view` with handle-based patterns using pre-computed index arrays and strided access descriptors. These patterns are compatible with future GPU dispatch.

**Depends on:** Phase 8 (registry, handles), Phase 9 (field lifecycle)

**Read first:**
- `src/fields/selector.hpp` (1113 lines — `plane_view<0/1/2>`, `multi_slice_view`, `predicate_view`, `optional_view`, `sel::D`, `sel::R`)
- `src/fields/selector_fwd.hpp` (`si::*`, `vi::*` index types)
- `src/fields/handle.hpp` (handle types, `handle_sel::*`)
- `src/fields/field_registry.hpp` (registry)
- `src/mesh/mesh.hpp` (`fluid`, `dirichlet`, `fluid_all`, `neumann` selectors)
- `src/mesh/mesh.cpp` (`init_slices` — fluid slice construction)
- `src/mesh/selections.hpp` (`YPlaneView`, `FView`)
- `src/mesh/cartesian.hpp` (mesh extents, `index_extents`)
- `src/indexing.hpp` (`stride<I>`, `dir<I>`)
- `src/systems/heat.cpp` (BC application: `u | m.dirichlet(...) = sol`)
- `src/systems/scalar_wave.cpp` (BC application patterns)

**Test commands:**
```bash
cmake --build build
ctest --test-dir build -R t-selector_v2
ctest --test-dir build -R t-mesh
ctest --test-dir build
```

---

## Design

Each runtime selection pattern maps to a **selection descriptor** — a small, trivially-copyable struct that describes which elements to access. The descriptor is constructed once at mesh construction time and reused every time step.

| Current Pattern | New Descriptor | Data |
|---|---|---|
| `plane_view<0>` (x-plane) | `contiguous_selection` | `{offset, count}` — 8 bytes |
| `plane_view<1>` (y-plane) | `strided_selection` | `{offset, inner_count, outer_count, outer_stride}` — 16 bytes |
| `plane_view<2>` (z-plane) | `strided_selection` | `{offset, stride, count}` — 12 bytes |
| `multi_slice_view` (fluid) | `gather_selection` | index into mesh-owned `Kokkos::View<int*>` index array |
| `predicate_view` (object BCs) | `gather_selection` | index into mesh-owned `Kokkos::View<int*>` index array (built via prefix scan) |
| `optional_view` | `contiguous_selection` | `count = 0` when inactive |

All descriptors are trivially copyable and can be captured in `KOKKOS_LAMBDA`.

---

## Items

### 11.1 — Selection descriptor types (TDD)

- [ ] **11.1a** Create `src/fields/selection_desc.t.cpp` testing selection descriptors:
  - `contiguous_selection{offset, count}`: `element(i)` returns `offset + i`.
  - `strided_selection`: for y-plane: `element(i)` returns `offset + (i / inner_count) * outer_stride + (i % inner_count)`.
  - `gather_selection{Kokkos::View<const int*> indices}`: `element(i)` returns `indices(i)`.
  - All are trivially copyable (except `gather_selection` which holds a View).
  - Test arithmetic correctness for 8×8×8 mesh geometry.
  - File: `src/fields/selection_desc.t.cpp` (new)
  - Test: `ctest --test-dir build -R t-selection_desc`

- [ ] **11.1b** Implement `src/fields/selection_desc.hpp`:
  - `contiguous_selection`, `strided_selection`, `gather_selection`.
  - File: `src/fields/selection_desc.hpp` (new)
  - Test: `ctest --test-dir build -R t-selection_desc`

### 11.2 — Selected assign / fill (TDD)

- [ ] **11.2a** Add tests for `assign_selected()` and `fill_selected()`:
  - `assign_selected(real* dst, selection_desc, expr)`: writes `expr(i)` to `dst[desc.element(i)]` for `i in [0, desc.count)`.
  - `fill_selected(real* dst, selection_desc, real value)`: fills selected elements with a constant.
  - Test with `contiguous_selection`: assign to a subrange.
  - Test with `strided_selection`: assign to a y-plane pattern.
  - Test with `gather_selection`: assign to scattered indices.
  - All dispatch via `Kokkos::parallel_for`.
  - File: `src/fields/selection_desc.t.cpp` (extend)
  - Test: `ctest --test-dir build -R t-selection_desc`

- [ ] **11.2b** Implement `assign_selected()` and `fill_selected()`.
  - File: `src/fields/selection_desc.hpp`
  - Test: `ctest --test-dir build -R t-selection_desc`

### 11.3 — Mesh builds selection descriptors at construction time

- [ ] **11.3a** Add methods to `mesh` that return selection descriptors:
  - `mesh::x_plane_desc(int i)` → `contiguous_selection`.
  - `mesh::y_plane_desc(int j)` → `strided_selection`.
  - `mesh::z_plane_desc(int k)` → `strided_selection`.
  - `mesh::fluid_desc()` → `gather_selection` (pre-computed index array from `init_slices`).
  - `mesh::dirichlet_desc(grid_bcs, object_bcs)` → collection of selection descriptors.
  - These are built once and cached in the mesh.
  - File: `src/mesh/mesh.hpp`, `src/mesh/mesh.cpp`
  - Test: `ctest --test-dir build -R t-mesh`

- [ ] **11.3b** Build the `gather_selection` index arrays for fluid and predicate selections:
  - Fluid: flatten the `index_slice` list into a `Kokkos::View<int*>`.
  - Predicate (Dirichlet/Neumann object BCs): scan the `mesh_object_info` arrays, build index arrays for matching entries.
  - Cache these in the mesh object.
  - File: `src/mesh/mesh.cpp`
  - Test: `ctest --test-dir build -R t-mesh`

### 11.4 — Replace BC application in systems

- [ ] **11.4a** Replace `u | m.dirichlet(grid_bcs, object_bcs) = sol` in `heat.cpp` with:
  - `fill_selected(registry, field_ref, scalar_handle.D(), mesh.dirichlet_desc(...), sol_expr)`.
  - Or: iterate the descriptor collection and call `assign_selected` per descriptor.
  - File: `src/systems/heat.cpp`
  - Test: `ctest --test-dir build -R t-heat`

- [ ] **11.4b** Same for `scalar_wave.cpp`, `inviscid_vortex.cpp`.
  - Files: respective `.cpp` files
  - Test: `ctest --test-dir build -L systems`

### 11.5 — Replace fluid_all selection in systems

- [ ] **11.5a** Replace `u_rhs | m.fluid_all(object_bcs) += src` with:
  - `plus_assign_selected(registry, ..., mesh.fluid_all_desc(object_bcs), src_expr)`.
  - File: `src/systems/heat.cpp`
  - Test: `ctest --test-dir build -R t-heat`

---

## Completion Criteria

- Selection descriptors correctly encode x/y/z-plane, fluid, and predicate patterns.
- `assign_selected` and `fill_selected` dispatch via `Kokkos::parallel_for`.
- Mesh builds and caches index arrays at construction time.
- BC application in all systems uses descriptors instead of iterator-based views.
- All existing tests pass.
- The iterator-based `plane_view<1>` (70-line custom iterator) is no longer used in hot paths.

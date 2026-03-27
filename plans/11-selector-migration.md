# Phase 11: Selector Migration to Handle Patterns

**Goal:** Replace the iterator-based `plane_view`, `multi_slice_view`, `predicate_view`, and `optional_view` with handle-based patterns using pre-computed index arrays and strided access descriptors. These patterns are compatible with future GPU dispatch.

**Depends on:** Phase 8 (registry, handles), Phase 9 (field lifecycle)

**Read first:**
- `src/fields/selector.hpp` (1113 lines — `plane_view<0/1/2>`, `multi_slice_view`, `predicate_view`, `optional_view`, `sel::D`, `sel::R`)
- `src/fields/selector_fwd.hpp` (`si::*`, `vi::*` index types)
- `src/fields/handle.hpp` (handle types, `handle_sel::*`)
- `src/fields/field_registry.hpp` (registry)
- `src/fields/expr.hpp` (expression templates: `handle_expr`, `assign()`, `plus_assign()`)
- `src/mesh/mesh.hpp` (`fluid`, `dirichlet`, `fluid_all`, `neumann` selectors)
- `src/mesh/mesh.cpp` (`init_slices` — fluid slice construction)
- `src/mesh/selections.hpp` (`YPlaneView`, `FView`)
- `src/mesh/cartesian.hpp` (mesh extents, `index_extents`)
- `src/indexing.hpp` (`stride<I>`, `dir<I>`)
- `src/operators/boundaries.hpp` (`bcs::Grid`, `bcs::Object`, `bcs::type`)
- `src/mesh/mesh_types.hpp` (`mesh_object_info`, `index_slice`)
- `src/systems/heat.cpp` (BC application: `u | m.dirichlet(...) = sol`)
- `src/systems/scalar_wave.cpp` (BC application patterns)

**Test commands:**
```bash
cmake --build build
ctest --test-dir build -R t-selection_desc
ctest --test-dir build -R t-mesh
ctest --test-dir build -R t-heat
ctest --test-dir build
```

---

## Design

Each runtime selection pattern maps to a **selection descriptor** — a small struct that describes which elements to access. The descriptor is constructed once (or cheaply on-demand) and reused every time step.

### Descriptor types

| Current Pattern | New Descriptor | Data | `element(i)` formula |
|---|---|---|---|
| `plane_view<0>` (x-plane) | `contiguous_selection` | `{offset, count}` — 8 bytes | `offset + i` |
| `plane_view<1>` (y-plane) | `strided_selection` | `{offset, inner_count, outer_count, outer_stride}` — 16 bytes | `offset + (i / inner_count) * outer_stride + (i % inner_count)` |
| `plane_view<2>` (z-plane) | `strided_selection` | same struct, `inner_count = 1` | `offset + i * outer_stride` (degenerate case) |
| `multi_slice_view` (fluid) | `gather_selection` | `Kokkos::View<const int*>` index array | `indices(i)` |
| `predicate_view` (object BCs) | `gather_selection` | `Kokkos::View<const int*>` index array | `indices(i)` |
| `optional_view` | `contiguous_selection` | `count = 0` when inactive | N/A (zero iterations) |

The `strided_selection` struct unifies both y-plane and z-plane patterns (see D-R15 in meta.md). For a mesh with extents `{nx, ny, nz}`:
- **x-plane at i**: `contiguous_selection{i * ny * nz, ny * nz}`
- **y-plane at j**: `strided_selection{j * nz, nz, nx, ny * nz}`
- **z-plane at k**: `strided_selection{k, 1, nx * ny, nz}`

`contiguous_selection` and `strided_selection` are trivially copyable. `gather_selection` holds a `Kokkos::View` (capturable in `KOKKOS_LAMBDA` but may not satisfy `std::is_trivially_copyable_v`).

### Source expression strategy (D-R16)

`assign_selected(dst, desc, expr)` evaluates `dst[desc.element(i)] = expr(desc.element(i))` — the expression is evaluated at the **absolute** flat index, not the relative selection index. This means source expressions (e.g., `handle_expr{sol_ptr}`) must be indexed on the same domain as the destination buffer.

For BC application, source values are pre-evaluated into scratch buffers using the existing tuple DSL (D-R6 coexistence), then `assign_selected` copies the selected elements.

### Scratch buffer strategy for source pre-evaluation

The `assign_selected` / `plus_assign_selected` functions take an `Expr` callable indexed by flat integer. Source expressions in BC application (e.g., manufactured solution values) are currently lazy views produced by the tuple DSL (`m.xyz | m_sol(time)`). To bridge the two worlds:

1. **Allocate a local `scalar_real`** with the field's sizes: `scalar_real sol{m.ss()};`
2. **Fill via old DSL**: `sol = m.xyz | m_sol(time);` — this evaluates the lazy view into all 4 buffers (D, Rx, Ry, Rz).
3. **Extract raw `real*` pointers** using the project's multi-level `get`:
   - `get<si::D>(sol).data()` → D buffer pointer
   - `get<si::Rx>(sol).data()` → Rx buffer pointer
   - `get<si::Ry>(sol).data()` / `get<si::Rz>(sol).data()` → Ry/Rz buffer pointers
4. **Wrap in `handle_expr`**: `handle_expr{sol_D_ptr}` — indexed by absolute flat index.

For **registry-backed** destination fields, pointers come from `reg.data(ref, sh.D())` etc.

For **fill-zero** operations (`u_rhs | m.dirichlet(...) = 0`), no scratch buffer is needed — `fill_selected` takes a scalar value directly.

Note: `heat::rhs` is `const`, so scratch buffers in that method must be local variables (stack-allocated `scalar_real`). The allocation overhead is negligible for host-only execution (D-R1).

### Scope: hot-path writes only

Per D-R6, the old selector infrastructure remains for read-only operations (stats, write, initialize). Phase 11 replaces **write operations** in `rhs` and `update_boundary`:
- `u | m.dirichlet(...) = sol` → `assign_selected` per descriptor
- `u_rhs | m.dirichlet(...) = 0` → `fill_selected` per descriptor
- `u_rhs | m.fluid_all(...) += src` → `plus_assign_selected` per descriptor

Other uses (`initialize`, `stats`, `write`, `neumann`, constructor setup) remain on the old tuple DSL and migrate in Phase 12.

---

## Items

### 11.1 — Selection descriptor types (TDD)

- [x] **11.1a** Create `src/fields/selection_desc.t.cpp` testing selection descriptors:
  - `contiguous_selection{offset, count}`: verify `element(i) == offset + i` and `count()`.
  - `strided_selection{offset, inner_count, outer_count, outer_stride}`: verify `element(i) == offset + (i / inner_count) * outer_stride + (i % inner_count)` and `count() == inner_count * outer_count`.
  - Test y-plane pattern: `strided_selection{j*nz, nz, nx, ny*nz}` for 8×6×4 mesh (use distinct dimensions to catch axis bugs).
  - Test z-plane pattern: `strided_selection{k, 1, nx*ny, nz}` for same mesh.
  - `gather_selection{Kokkos::View<const int*> indices}`: verify `element(i) == indices(i)` and `count() == indices.extent(0)`.
  - Verify `std::is_trivially_copyable_v<contiguous_selection>` and `std::is_trivially_copyable_v<strided_selection>`.
  - Test plane factory functions: `make_x_plane_desc`, `make_y_plane_desc`, `make_z_plane_desc` (see 11.1b).
  - Cross-check: for each plane descriptor, verify the selected indices match the set of flat indices `{i*ny*nz + j*nz + k}` that the plane should select.
  - File: `src/fields/selection_desc.t.cpp` (new)
  - CMake: Kokkos-aware test (custom `main()` with `Kokkos::ScopeGuard`, link `Catch2::Catch2` not `Catch2::Catch2WithMain`, per D-R9)
  - Test: `ctest --test-dir build -R t-selection_desc`

- [x] **11.1b** Implement `src/fields/selection_desc.hpp`:
  - Structs: `contiguous_selection`, `strided_selection`, `gather_selection`.
  - Each provides `KOKKOS_INLINE_FUNCTION int element(int i) const` and `int count() const`.
  - Plane descriptor factory functions (depend only on `index_extents`):
    - `make_x_plane_desc(index_extents ext, int i)` → `contiguous_selection{i * ext[1] * ext[2], ext[1] * ext[2]}`
    - `make_y_plane_desc(index_extents ext, int j)` → `strided_selection{j * ext[2], ext[2], ext[0], ext[1] * ext[2]}`
    - `make_z_plane_desc(index_extents ext, int k)` → `strided_selection{k, 1, ext[0] * ext[1], ext[2]}`
  - Include: `kokkos_types.hpp`, `index_extents.hpp`
  - File: `src/fields/selection_desc.hpp` (new)
  - CMake: add test executable to `src/fields/CMakeLists.txt`:
    ```cmake
    add_executable(t-selection_desc selection_desc.t.cpp)
    target_link_libraries(t-selection_desc Catch2::Catch2 fields Kokkos::kokkos)
    add_test(NAME t-selection_desc COMMAND t-selection_desc)
    set_tests_properties(t-selection_desc PROPERTIES LABELS "fields")
    ```
  - Test: `ctest --test-dir build -R t-selection_desc`

### 11.1-fix — Test gap: gather_selection::element() untested

- [x] **11.1-fix** Fix `gather_selection` test to actually call `element()`:
  - Current test ("gather_selection element and count" in `selection_desc.t.cpp` lines 90-107) verifies `count()` and then checks the mirror host view values (`h(0) == 3` etc.), which are trivially true — the mirror was just filled with those values. `sel.element(i)` is never called.
  - Fix: replace `REQUIRE(h(0) == 3)` etc. with `REQUIRE(sel.element(0) == 3)`, `REQUIRE(sel.element(1) == 7)`, `REQUIRE(sel.element(2) == 1)`, `REQUIRE(sel.element(3) == 42)`. This works because `memory_space` is host memory.
  - File: `src/fields/selection_desc.t.cpp`
  - Test: `ctest --test-dir build -R t-selection_desc`

### 11.2 — Selected assign / fill / plus_assign (TDD)

- [x] **11.2a** Add tests for `assign_selected()`, `fill_selected()`, and `plus_assign_selected()`:
  - `assign_selected(real* dst, Desc desc, Expr expr)`: sets `dst[desc.element(i)] = expr(desc.element(i))` for `i ∈ [0, desc.count())` via `Kokkos::parallel_for`.
  - `fill_selected(real* dst, Desc desc, real value)`: sets `dst[desc.element(i)] = value`.
  - `plus_assign_selected(real* dst, Desc desc, Expr expr)`: sets `dst[desc.element(i)] += expr(desc.element(i))`.
  - Test each function with all three descriptor types:
    - `contiguous_selection`: assign to a subrange of a larger buffer, verify only selected elements changed.
    - `strided_selection`: assign to a y-plane-like pattern, verify correct scattered writes.
    - `gather_selection`: assign to scattered indices, verify correct elements updated.
  - Use `handle_expr{src_ptr}` and `scalar_literal_expr{val}` from `src/fields/expr.hpp` as test expressions.
  - File: `src/fields/selection_desc.t.cpp` (extend)
  - Test: `ctest --test-dir build -R t-selection_desc`

- [x] **11.2b** Implement `assign_selected()`, `fill_selected()`, and `plus_assign_selected()`:
  - Template functions in `src/fields/selection_desc.hpp`.
  - All dispatch via `Kokkos::parallel_for(Kokkos::RangePolicy<execution_space>(0, desc.count()), ...)`.
  - Pattern: `KOKKOS_LAMBDA(int i) { int idx = desc.element(i); dst[idx] = expr(idx); }`
  - Expressions follow the same `operator()(int i) -> real` interface as Phase 10 expression templates.
  - File: `src/fields/selection_desc.hpp`
  - Test: `ctest --test-dir build -R t-selection_desc`

### 11.3 — Mesh builds selection descriptors

- [x] **11.3a** Implement `make_gather_from_slices()` — build gather_selection from `index_slice` arrays:
  - Free function: `gather_selection make_gather_from_slices(std::span<const index_slice> slices)`.
  - Count total elements across all slices, allocate `Kokkos::View<int*>`, fill with flattened indices: for each `{first, last}` slice, write indices `[first, last)`.
  - This replaces `multi_slice_view` for the fluid selection.
  - Add tests: construct slices `{{0,5}, {10,15}}`, verify gather indices are `[0,1,2,3,4,10,11,12,13,14]`.
  - File: `src/fields/selection_desc.hpp` (extend), `src/fields/selection_desc.t.cpp` (extend)
  - Test: `ctest --test-dir build -R t-selection_desc`

- [x] **11.3b** Implement `make_gather_from_predicate()` — build gather_selection from predicate over arrays:
  - Template function: `gather_selection make_gather_from_predicate(std::span<const mesh_object_info> infos, Pred pred)`.
  - Scan `infos`, collect indices `i` where `pred(infos[i])` is true into a `Kokkos::View<int*>`.
  - Used for object Dirichlet: `pred = [&o](auto& info) { return o[info.shape_id] == bcs::Dirichlet; }`.
  - Used for non-Dirichlet (fluid_all): `pred = [&o](auto& info) { return o[info.shape_id] != bcs::Dirichlet; }`.
  - Add tests with a small mock `mesh_object_info` array and known predicate.
  - File: `src/fields/selection_desc.hpp` (extend), `src/fields/selection_desc.t.cpp` (extend)
  - Depends on: `src/mesh/mesh_types.hpp` for `mesh_object_info`
  - Test: `ctest --test-dir build -R t-selection_desc`

- [x] **11.3c** Cache fluid descriptor in `mesh` and add on-demand BC descriptor methods:
  - Add member: `gather_selection fluid_desc_` — built from `fluid_slices` using `make_gather_from_slices` in `mesh` constructor (after `init_slices` populates `fluid_slices`).
  - Add accessor: `const gather_selection& fluid_desc() const { return fluid_desc_; }`
  - Add on-demand methods (allocate per call; cheap for host execution per D1):
    - `gather_selection dirichlet_object_desc(int dir, const bcs::Object& o) const` — calls `make_gather_from_predicate(R(dir), [&](auto& info) { return o[info.shape_id] == bcs::Dirichlet; })`.
    - `gather_selection non_dirichlet_object_desc(int dir, const bcs::Object& o) const` — complement predicate.
  - Include `selection_desc.hpp` in `mesh.hpp`.
  - File: `src/mesh/mesh.hpp` (add members + methods), `src/mesh/mesh.cpp` (constructor init)
  - Test: `ctest --test-dir build -R t-mesh` — verify `fluid_desc()` produces indices matching the existing `fluid` multi_slice selector for a known mesh geometry.

- [x] **11.3d** Add grid BC descriptor helper — `for_each_grid_bc_desc()`:
  - Template function: `for_each_grid_bc_desc<bcs::type B>(const bcs::Grid& g, index_extents ext, Fn fn)`.
  - For each of 6 faces (xmin, xmax, ymin, ymax, zmin, zmax), check if BC type matches `B`. If so, create the appropriate plane descriptor (`make_x/y/z_plane_desc`) and call `fn(desc)`.
  - Mapping: `g[0].left` → xmin (x=0), `g[0].right` → xmax (x=nx-1), `g[1].left` → ymin (y=0), `g[1].right` → ymax (y=ny-1), `g[2].left` → zmin (z=0), `g[2].right` → zmax (z=nz-1).
  - `fn` is a generic lambda `[&](auto desc) { ... }` — template dispatch handles contiguous vs strided automatically.
  - File: `src/fields/selection_desc.hpp` (extend) — depends on `boundaries.hpp` for `bcs::Grid`, `bcs::type`.
  - Test: add test in `selection_desc.t.cpp` — verify that `for_each_grid_bc_desc<bcs::Dirichlet>(grid, ext, collector)` produces the correct set of descriptors for a known BC configuration.
  - Test: `ctest --test-dir build -R t-selection_desc`

### 11.4 — Replace BC application in rhs / update_boundary

**Pattern:** Pre-evaluate source values into scratch buffers using existing tuple DSL, then use selection descriptors + `assign_selected`/`fill_selected` for the actual writes.

- [x] **11.4a** Replace dirichlet assignment in `heat::update_boundary`:
  - Current (line 129): `u | m.dirichlet(grid_bcs, object_bcs) = l | m_sol(time);`
  - Replacement code pattern:
    ```cpp
    // Pre-evaluate manufactured solution into local scratch (old DSL)
    scalar_real sol{m.ss()};
    sol = m.xyz | m_sol(time);

    // Extract destination pointers from registry
    real* u_D = reg.data(ref, sh.D());

    // Extract source pointers from scratch (see Design: Scratch buffer strategy)
    real* sol_D  = get<si::D>(sol).data();
    real* sol_Rx = get<si::Rx>(sol).data();
    real* sol_Ry = get<si::Ry>(sol).data();
    real* sol_Rz = get<si::Rz>(sol).data();

    // Grid Dirichlet: assign plane subsets of D buffer
    for_each_grid_bc_desc<bcs::Dirichlet>(grid_bcs, m.extents(), [&](auto desc) {
        assign_selected(u_D, desc, handle_expr{sol_D});
    });

    // Object Dirichlet: assign predicate subsets of Rx/Ry/Rz buffers
    auto R = sh.R();  // std::array<buf_handle, 3> = {Rx, Ry, Rz}
    real* sol_R[] = {sol_Rx, sol_Ry, sol_Rz};
    for (int dir = 0; dir < 3; ++dir) {
        auto gd = m.dirichlet_object_desc(dir, object_bcs);
        assign_selected(reg.data(ref, R[dir]), gd, handle_expr{sol_R[dir]});
    }
    ```
  - Include `fields/selection_desc.hpp` in `heat.cpp` (for `assign_selected`, `for_each_grid_bc_desc`).
  - Dead code cleanup: after replacement, `auto u = extract_scalar_span(reg, ref, sh)` (line 126) is no longer referenced — remove it. Keep `auto l = m.xyz` (line 127): still needed for neumann assignments on lines 132-134.
  - Note: Neumann assignment (lines 132-134) operates on `neumann_u` (old `scalar_real` type, not in registry). Unchanged; deferred to Phase 12.
  - File: `src/systems/heat.cpp`
  - Test: `ctest --test-dir build -R t-heat`

- [x] **11.4b** Replace dirichlet fill-zero in `heat::rhs`:
  - Current (line 119): `u_rhs | m.dirichlet(grid_bcs, object_bcs) = 0;`
  - No scratch buffer needed — `fill_selected` takes a scalar value directly.
  - Replacement code pattern:
    ```cpp
    // Extract destination pointers from output registry
    real* rhs_D = out_reg.data(output, sh.D());

    // Grid Dirichlet: fill plane subsets of D buffer with zero
    for_each_grid_bc_desc<bcs::Dirichlet>(grid_bcs, m.extents(), [&](auto desc) {
        fill_selected(rhs_D, desc, 0.0);
    });

    // Object Dirichlet: fill predicate subsets of Rx/Ry/Rz buffers
    auto R = sh.R();
    for (int dir = 0; dir < 3; ++dir) {
        auto gd = m.dirichlet_object_desc(dir, object_bcs);
        fill_selected(out_reg.data(output, R[dir]), gd, 0.0);
    }
    ```
  - Note: `rhs` receives `out_reg` and `output` for the RHS field (not the same ref as the input).
  - Context: `auto u = extract_scalar_view(reg, input, sh)` and `auto u_rhs = extract_scalar_span(out_reg, output, sh)` (lines 107-108) are still needed for lines 111-112 (`u_rhs = lap(u, neumann_u)` and `times_assign_scalar`). Do not remove them. The replacement code for lines 118-119 stays inside the existing `if (m_sol) { ... }` guard (lines 114-120).
  - File: `src/systems/heat.cpp`
  - Test: `ctest --test-dir build -R t-heat`
  - Ordering: after 11.4a (same file, both in heat.cpp)

- [x] **11.4c** Replace dirichlet assignment in `scalar_wave::update_boundary`:
  - Current (line 185): `u | m.dirichlet(grid_bcs, object_bcs) = sol;`
  - Context: `sol` is `m.xyz | solution(center, radius, time)` (lazy view, line 183).
  - Same scratch pattern as 11.4a:
    ```cpp
    // Pre-evaluate solution into local scratch
    scalar_real sol_buf{m.ss()};
    sol_buf = m.xyz | solution(center, radius, time);

    // Grid Dirichlet on D buffer
    real* u_D = reg.data(ref, sh.D());
    real* sol_D = get<si::D>(sol_buf).data();
    for_each_grid_bc_desc<bcs::Dirichlet>(grid_bcs, m.extents(), [&](auto desc) {
        assign_selected(u_D, desc, handle_expr{sol_D});
    });

    // Object Dirichlet on R buffers
    auto R = sh.R();
    real* sol_R[] = { get<si::Rx>(sol_buf).data(),
                      get<si::Ry>(sol_buf).data(),
                      get<si::Rz>(sol_buf).data() };
    for (int dir = 0; dir < 3; ++dir) {
        auto gd = m.dirichlet_object_desc(dir, object_bcs);
        assign_selected(reg.data(ref, R[dir]), gd, handle_expr{sol_R[dir]});
    }
    ```
  - Include `fields/selection_desc.hpp` in `scalar_wave.cpp`.
  - Note: `solution()` is a file-local constexpr function (lines 33-37), not a member `m_sol`.
  - Dead code cleanup: after replacement, `auto u = extract_scalar_span(reg, ref, sh)` (line 182) and `auto sol = m.xyz | solution(center, radius, time)` (line 183) are no longer referenced — remove both. The method body becomes just the scratch buffer + descriptor code.
  - File: `src/systems/scalar_wave.cpp`
  - Test: `ctest --test-dir build -R t-scalar_wave`

- [x] **11.4c-test** Add integration test for `scalar_wave::update_boundary`:
  - 11.4c references `ctest --test-dir build -R t-scalar_wave` but no such test exists. The `update_boundary` changes are untested.
  - Create a minimal `scalar_wave` system test (similar to `t-heat`) that constructs a `scalar_wave` system, runs `update_boundary`, and verifies the Dirichlet values match the manufactured solution at boundary nodes.
  - File: `src/systems/scalar_wave.t.cpp` (new)
  - CMake: add to `src/systems/CMakeLists.txt` with label `systems`.
  - Test: `ctest --test-dir build -R t-scalar_wave` — PASSED

- [x] **11.4c-test-fix** Strengthen `scalar_wave::update_boundary` test:
  - The current test calls `update_boundary` at t=0 where `initialize` already wrote the exact solution. `stats` only checks `fluid_all` (non-Dirichlet) points, so the test passes even if `update_boundary` is a no-op.
  - Fix: call `update_boundary` at t≠0 (e.g., t=0.25) so boundary values differ from `initialize`'s t=0 values, then directly verify that D-buffer entries on Dirichlet faces match `solution(center, radius, 0.25)`. Use `make_x_plane_desc` / `make_z_plane_desc` to read back the boundary entries and compare against the expected solution values.
  - Alternatively: zero out or perturb the D-buffer Dirichlet face entries before calling `update_boundary(t=0)`, then verify they are restored to the correct solution values.
  - File: `src/systems/scalar_wave.t.cpp`
  - Test: `ctest --test-dir build -R t-scalar_wave`

- [x] **11.4d** Verify `inviscid_vortex.cpp` — no changes needed:
  - All methods are stubs with empty implementations (lines 219-255). No BC application to replace.
  - Verify: `ctest --test-dir build -L systems`

### 11.5 — Replace fluid_all selection in heat::rhs

- [x] **11.5a** Replace fluid_all `+=` in `heat::rhs`:
  - Current (line 118): `u_rhs | m.fluid_all(object_bcs) += src;`
  - `fluid_all` = fluid (D buffer) + non-dirichlet objects (Rx/Ry/Rz buffers).
  - `src` is `(m.xyz | m_sol.ddt(time)) - (diffusivity * (m.xyz | m_sol.laplacian(time)))`.
  - Note: `rhs` is `const`, so scratch must be a local variable (no mutable members).
  - Replacement code pattern:
    ```cpp
    // Pre-evaluate source expression into local scratch (old DSL)
    scalar_real src_buf{m.ss()};
    src_buf = (m.xyz | m_sol.ddt(time)) - (diffusivity * (m.xyz | m_sol.laplacian(time)));

    // Extract destination pointers from output registry
    real* rhs_D = out_reg.data(output, sh.D());

    // Extract source pointers from scratch
    real* src_D  = get<si::D>(src_buf).data();
    real* src_Rx = get<si::Rx>(src_buf).data();
    real* src_Ry = get<si::Ry>(src_buf).data();
    real* src_Rz = get<si::Rz>(src_buf).data();

    // Fluid on D buffer: plus_assign from gather_selection of fluid indices
    plus_assign_selected(rhs_D, m.fluid_desc(), handle_expr{src_D});

    // Non-dirichlet objects on Rx/Ry/Rz buffers
    auto R = sh.R();
    real* src_R[] = {src_Rx, src_Ry, src_Rz};
    for (int dir = 0; dir < 3; ++dir) {
        auto gd = m.non_dirichlet_object_desc(dir, object_bcs);
        plus_assign_selected(out_reg.data(output, R[dir]), gd,
                             handle_expr{src_R[dir]});
    }
    ```
  - File: `src/systems/heat.cpp`
  - Ordering: implement after 11.4b, but in the method body 11.5a's `plus_assign_selected` code must appear **before** 11.4b's `fill_selected` code — matching the original source ordering (line 118 `fluid_all +=` before line 119 `dirichlet = 0`). The Dirichlet zero-fill overrides the accumulated values, so it must run last.
  - Note: 11.4b and 11.5a share the same method and both extract `rhs_D`, `R`, etc. Merge variable declarations — do not duplicate them. The `src` lazy expression (line 115-116) is replaced by `src_buf` scratch — remove the old `const auto src = ...` declaration. The `if (m_sol)` guard remains.
  - Test: `ctest --test-dir build -R t-heat`

---

## Deferred to Phase 12

The following uses of selectors remain on the old tuple DSL in Phase 11 (per D-R6 coexistence):
- `heat::initialize` / `scalar_wave::initialize`: `u | sel::D = 0`, `u | m.fluid = sol`, `u | sel::R = sol`
- `heat::stats` / `scalar_wave::stats`: read-only selections for error computation
- `heat::write` / `scalar_wave::write`: `error | m.fluid_all(...) = ...`, `error | m.dirichlet(...) = 0`
- Neumann BC: `neumann_u | m.neumann<I>(grid_bcs) = ...` (operates on old `field` type)
- `scalar_wave` constructor: `grad_G | m.fluid = ...`, `grad_G | m.dirichlet(...) = 0`

---

## Completion Criteria

- Selection descriptors correctly encode x/y/z-plane, fluid, and predicate patterns.
- `assign_selected`, `fill_selected`, and `plus_assign_selected` dispatch via `Kokkos::parallel_for`.
- Mesh builds and caches the fluid gather_selection at construction time.
- Mesh provides on-demand object BC descriptors (`dirichlet_object_desc`, `non_dirichlet_object_desc`).
- BC write operations in `rhs` and `update_boundary` use descriptors instead of iterator-based views.
- All existing tests pass.
- The iterator-based `plane_view<1>` and `multi_slice_view` are no longer used in `rhs`/`update_boundary` hot paths.

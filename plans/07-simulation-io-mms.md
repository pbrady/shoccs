# Phase 7: Simulation, I/O, MMS, and Mesh

**Goal:** Migrate remaining subsystems and remove the range-v3 dependency entirely.

**Depends on:** All previous phases

**Read first:**
- `src/simulation/simulation_cycle.hpp` + `simulation_cycle.cpp` (no range-v3 in impl)
- `src/simulation/simulation_builder.hpp` + `simulation_builder.cpp` (no range-v3)
- `src/io/field_io.hpp` + `field_io.cpp` (moderate: vs::transform, rs::to)
- `src/io/field_data.hpp` + `field_data.cpp` (moderate: rs::for_each, vs::zip, vs::transform)
- `src/io/xdmf.hpp` + `xdmf.cpp` (light: vs::zip, vs::repeat_n, rs::size)
- `src/io/logging.hpp` + `logging.cpp` (no range-v3)
- `src/io/interval.hpp` (no range-v3)
- `src/mms/manufactured_solutions.hpp` (heavy: 6 vs::transform view-adaptor methods — public API)
- `src/mms/mms.cpp` (light: rs::action::transform for string lowercasing)
- `src/mms/gauss1d.cpp`, `gauss2d.cpp`, `gauss3d.cpp` (no range-v3)
- `src/mms/lua_mms.hpp` + `lua_mms.cpp` (no range-v3)
- `src/mesh/cartesian.hpp` + `cartesian.cpp` (heavy: linear_distribute, zip_with, cartesian_product, copy, count_if, to)
- `src/mesh/selections.hpp` (heaviest in codebase: custom view adaptors YPlaneView, FView)
- `src/mesh/mesh.hpp` + `mesh.cpp` (light: rs::begin/end)
- `src/mesh/mesh_view.hpp` (uses cppcoro, no range-v3)
- `src/mesh/object_geometry.hpp` + `object_geometry.cpp` (light: vs::transform)
- `plans/meta.md`

**Test commands:**
```bash
cmake --build build
ctest --test-dir build
```

---

## Items

### Mesh (High Complexity)

- [ ] **7.1** Migrate `cartesian.cpp`: Replace `vs::linear_distribute` with manual computation (`min + i*(max-min)/(n-1)`). Replace `vs::zip_with` for grid spacing. Replace `rs::copy`, `rs::count_if`, `rs::to<vector>`. Replace `vs::cartesian_product` in `domain()`.
  - Test: `ctest --test-dir build -R t-cartesian`

- [ ] **7.2** Migrate `selections.hpp`: Replace `YPlaneView` and `FView` custom `rs::view_adaptor` classes. Replace all `rs::make_view_closure`, `rs::bind_back`, `rs::adaptor_base`, `rs::range_access` usage. This is one of the two highest-effort files in the codebase (alongside `fields/selector.hpp`).
  - Test: `ctest --test-dir build -R t-mesh`

- [ ] **7.3** Migrate `object_geometry.cpp`: Replace `vs::transform(&mesh_object_info::position)`.
  - Test: `ctest --test-dir build -R t-object_geometry`

- [ ] **7.4** Migrate `mesh.cpp`: Replace `rs::begin`/`rs::end` and any `vs::transform` in `object_boundaries()`.
  - Test: `ctest --test-dir build -R t-mesh`

- [ ] **7.5** Migrate `mesh_view.hpp`: If cppcoro is still in use here after Phase 0, replace with plain loops.
  - Test: `ctest --test-dir build -R t-mesh_view`

### MMS (Medium Complexity)

- [ ] **7.6** Migrate `manufactured_solutions.hpp`: The 6 methods returning `vs::transform(...)` view adaptors are the MMS public API. Replace with `std::views::transform` or change the API to take a location range and return a vector of values.
  - Test: `ctest --test-dir build -R t-mms`

- [ ] **7.7** Migrate `mms.cpp`: Replace `rs::action::transform(tolower)` with `std::transform` for string lowercasing.
  - Test: `ctest --test-dir build -R t-mms`

### I/O (Low-Medium Complexity)

- [ ] **7.8** Migrate `field_io.cpp`: Replace `vs::transform` + `rs::to<vector<string>>()` for filename generation.
  - Test: `ctest --test-dir build -R t-field_io`

- [ ] **7.9** Migrate `field_data.cpp`: Replace `rs::for_each`, `vs::zip`, `vs::transform`, `rs::size`.
  - Test: `ctest --test-dir build -R t-field_io`

- [ ] **7.10** Migrate `xdmf.cpp`: Replace `vs::zip`, `vs::repeat_n`, `rs::size`.
  - Test: `ctest --test-dir build -R t-xdmf`

### Simulation (Low Complexity)

- [ ] **7.11** Verify `simulation_cycle.cpp` and `simulation_builder.cpp` have no range-v3 usage.
  - Test: build succeeds.

### Test Migration

- [ ] **7.12** Migrate all remaining test files:
  - `mesh/mesh.t.cpp` (heavy range-v3)
  - `mesh/cartesian.t.cpp`, `mesh_view.t.cpp`, `object_geometry.t.cpp`, `shapes.t.cpp`
  - `simulation/simulation_cycle.t.cpp`
  - `io/field_io.t.cpp`, `xdmf.t.cpp`, `interval.t.cpp`, `logging.t.cpp`
  - `mms/mms.t.cpp`
  - Root: `src/indexing.t.cpp`, `src/index_view.t.cpp`, `src/real3_operators.t.cpp`
  - Test: `ctest --test-dir build` — all pass.

### Final Cleanup

- [ ] **7.13** Remove `find_package(range-v3)` from top-level `CMakeLists.txt`.
- [ ] **7.14** Remove `range-v3` from `config/shoccsConfig.cmake.in`.
- [ ] **7.15** Remove `range-v3` from `.devcontainer/spack.yaml`.
- [ ] **7.16** Remove any remaining `rs`/`vs` namespace aliases from `src/types.hpp`.
- [ ] **7.17** Remove `#include <range/v3/...>` from any remaining files.
- [ ] **7.18** Full build and test: `cmake --build build && ctest --test-dir build` — all pass.

---

## Completion Criteria

- **Zero** references to range-v3 remain anywhere in the codebase.
- `find_package(range-v3)` is removed from CMake.
- All tests pass.
- The `shoccs` executable runs successfully with all 4 Lua config files.

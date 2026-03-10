# Phase 4: Operators Subsystem

**Goal:** Migrate spatial discretization operators from range-v3 to std/Kokkos.

**Depends on:** Phases 0–3 (operators use fields, matrices, stencils, and mesh)

**Read first:**
- `src/operators/derivative.hpp` + `derivative.cpp` (19 rs/vs uses — heaviest operator file)
- `src/operators/gradient.hpp` + `gradient.cpp` (1 rs/vs use — trivial)
- `src/operators/laplacian.hpp` + `laplacian.cpp` (1 rs/vs use — trivial)
- `src/operators/eigenvalue_visitor.hpp` + `eigenvalue_visitor.cpp` (0 rs/vs uses)
- `src/operators/boundaries.hpp` + `boundaries.cpp` (0 rs/vs uses)
- `src/operators/identity_stencil.hpp` (test fixture, no range-v3)
- `src/operators/CMakeLists.txt`
- `plans/meta.md`

**Test commands:**
```bash
cmake --build build
ctest --test-dir build -L operators
ctest --test-dir build -L bcs
```

---

## Items

### Core Derivative Operator

- [ ] **4.1** Migrate `derivative.cpp`: This is the critical path. Replace:
  - `vs::enumerate` on shapes → plain indexed loop
  - `vs::drop`/`vs::take`/`vs::drop_exactly`/`vs::take_exactly` on coefficient arrays → `std::span` subviews
  - `vs::reverse` on coefficient arrays → `std::ranges::reverse` or reversed copy
  - `vs::chunk | vs::for_each` on stencil rows → explicit row iteration
  - `vs::stride` for non-X-direction operations → manual stride indexing
  - `rs::accumulate` → `std::accumulate` or manual loop
  - `rs::begin` → `std::ranges::begin`
  - `rs::random_access_range` concept → `std::ranges::random_access_range`
  - Files: `src/operators/derivative.hpp`, `src/operators/derivative.cpp`
  - Test: `ctest --test-dir build -R t-derivative`

- [ ] **4.2** Migrate `derivative.hpp`: Replace any remaining `rs::` references in the header.
  - Test: `ctest --test-dir build -R t-derivative`

### Gradient and Laplacian (Trivial)

- [ ] **4.3** Migrate `gradient.cpp`: Replace the single `vs::repeat_n` use (log header formatting) with `std::string(n, '-')` or similar.
  - Test: `ctest --test-dir build -R t-gradient`

- [ ] **4.4** Migrate `laplacian.cpp`: Same as gradient — single `vs::repeat_n` for log formatting.
  - Test: `ctest --test-dir build -R t-laplacian`

### Eigenvalue Visitor (No range-v3)

- [ ] **4.5** Verify `eigenvalue_visitor.cpp` has no range-v3 usage. If clean, mark complete.
  - Test: `ctest --test-dir build -R t-eigenvalue_visitor`

### Boundaries (No range-v3)

- [ ] **4.6** Verify `boundaries.cpp` has no range-v3 usage. If clean, mark complete.
  - Test: `ctest --test-dir build -R t-boundaries`

### Test Migration

- [ ] **4.7** Migrate operator test files to remove range-v3:
  - `derivative.t.cpp` (26 rs/vs uses — heaviest test)
  - `gradient.t.cpp` (12 uses)
  - `laplacian.t.cpp` (18 uses)
  - `eigenvalue_visitor.t.cpp` (2 uses)
  - `boundaries.t.cpp` (0 uses)
  - Test: `ctest --test-dir build -L operators` — all pass.

### Remove range-v3 from CMake

- [ ] **4.8** Remove range-v3 link dependency from `src/operators/CMakeLists.txt` if present.
  - Test: full build succeeds.

---

## Completion Criteria

- All 5 operator test files pass (excluding disabled directional).
- No `#include <range/v3/...>` remains in `src/operators/` (excluding disabled files).
- `shoccs-operators` and `shoccs-bcs` libraries no longer depend on range-v3.

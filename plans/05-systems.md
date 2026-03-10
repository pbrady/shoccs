# Phase 5: Systems Subsystem

**Goal:** Migrate PDE system implementations from range-v3 to std/Kokkos.

**Depends on:** Phases 0–4 (systems use fields, operators, mesh selectors)

**Read first:**
- `src/systems/heat.hpp` + `heat.cpp` (moderate: rs::max_element, rs::min, rs::distance, vs::transform)
- `src/systems/scalar_wave.hpp` + `scalar_wave.cpp` (moderate: rs::max_element, rs::min, vs::transform)
- `src/systems/inviscid_vortex.hpp` + `inviscid_vortex.cpp` (light: rs::max, vs::transform — partially #if 0)
- `src/systems/hyperbolic_eigenvalues.hpp` + `hyperbolic_eigenvalues.cpp` (light: rs::min, vs::transform)
- `src/systems/system.hpp` + `system.cpp` (no range-v3)
- `src/systems/empty_system.hpp` + `empty_system.cpp` (no range-v3)
- `src/systems/CMakeLists.txt`
- `plans/meta.md`

**Test commands:**
```bash
cmake --build build
ctest --test-dir build -L systems
```

---

## Items

- [ ] **5.1** Migrate `heat.cpp`: Replace `rs::max_element` with `std::ranges::max_element`. Replace `rs::min` with `std::ranges::min`. Replace `rs::distance`/`rs::begin`/`rs::end` with std equivalents. Remove `#include <range/v3/algorithm/max_element.hpp>`.
  - Test: `ctest --test-dir build -R t-heat`

- [ ] **5.2** Migrate `scalar_wave.cpp`: Same algorithm replacements as heat. Also replace `vs::transform` usage for initial condition generation with `std::views::transform` or a manual loop.
  - Test: `ctest --test-dir build -R t-scalar_wave` (if test exists; otherwise verify build)

- [ ] **5.3** Migrate `inviscid_vortex.cpp`: Replace `rs::max` with `std::ranges::max`. Replace `vs::transform`. Note: much of this code is `#if 0` — only migrate active code.
  - Test: build succeeds (no dedicated test for this system).

- [ ] **5.4** Migrate `hyperbolic_eigenvalues.cpp`: Replace `rs::min` and `vs::transform`.
  - Test: `ctest --test-dir build -R t-hyperbolic_eigenvalues`

- [ ] **5.5** Verify `system.cpp` and `empty_system.cpp` have no range-v3 usage.
  - Test: build succeeds.

- [ ] **5.6** Migrate system test files:
  - `heat.t.cpp` (uses `rs::count`, `rs::accumulate`, `#include <range/v3/all.hpp>`)
  - `hyperbolic_eigenvalues.t.cpp`
  - Test: `ctest --test-dir build -L systems` — all pass.

- [ ] **5.7** Exclude `cc_elliptic.hpp/cpp` — dead code (see `meta.md`).

---

## Completion Criteria

- All system tests pass.
- No `#include <range/v3/...>` remains in `src/systems/` (excluding dead cc_elliptic).

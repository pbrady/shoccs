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

### 5.1 — Migrate `heat.cpp`

- [x] **5.1** Migrate `src/systems/heat.cpp` (1 file, ~10 changed lines). ✓ Compiled (object file). Full build/test blocked by pre-existing `manufactured_solutions.hpp` errors (Phase 7 dependency).
  - **Includes:**
    - Remove: `#include <range/v3/algorithm/max_element.hpp>` (line 12).
    - Add: `#include <iterator>` (for `std::ranges::distance`). Note: `<algorithm>` and `<ranges>` are already available transitively via `fields/algorithms.hpp`.
  - **Replacements:**
    - Line 76: `transform(rs::max_element, fluid_error)` → `transform(std::ranges::max_element, fluid_error)`. (`std::ranges::max_element` is a niebloid/function object; it can be passed as a callable.)
    - Line 79: `rs::end(rng)` → `std::ranges::end(rng)`.
    - Line 81: `rs::distance(rs::begin(rng.base()), max_el.base())` → `std::ranges::distance(std::ranges::begin(rng.base()), max_el.base())`.
    - Line 116: `rs::min(m.h())` → `std::ranges::min(m.h())`. (`m.h()` returns `real3` = `std::array<real,3>`, a valid `std::ranges::range`.)
  - **Header file:** `heat.hpp` has no range-v3 usage — no changes needed.
  - **Ordering:** Independent; can run before or after any other 5.x item.
  - Test: `cmake --build build && ctest --test-dir build -R t-heat`

### 5.2 — Migrate `scalar_wave.cpp`

- [x] **5.2** Migrate `src/systems/scalar_wave.cpp` — partially migrated. ✓ Algorithm calls (`rs::min`, `rs::max_element`, `rs::end`, `rs::distance`) migrated to `std::ranges::`. Pre-existing compilation errors from `manufactured_solutions.hpp` (Phase 7) and `common_tuple`/`real3` mismatch.
  - **Includes:**
    - Remove: `#include <range/v3/algorithm/max_element.hpp>` (line 12).
    - Keep: `#include <range/v3/view/transform.hpp>` — still needed; `vs::transform` in `neg_G<I>` and `solution` cannot be replaced with `std::views::transform` because the mesh views (`m.xyz`, `m.vxyz`) produce range-v3 types (`cartesian_product_view`, `transform_view`) that don't satisfy `std::ranges::viewable_range`. This will be resolvable once mesh views are migrated.
    - Add: `#include <iterator>` (for `std::ranges::distance`).
    - Add: `#include <fmt/ranges.h>` (for `fmt::join`, previously provided transitively by removed range-v3 include).
  - **Replacements — `vs::transform` NOT migrated (blocked):**
    - `neg_G<I>` and `solution` functions still use `vs::transform` because `std::views::transform` can't pipe through range-v3 view types from mesh. Deferred to Phase 7 or mesh migration.
  - **Replacements — algorithm calls in `stats()` (lines 84, 118–123): ✓ done**
    - `rs::min(m.h())` → `std::ranges::min(m.h())`.
    - `transform(rs::max_element, fluid_error)` → `transform(std::ranges::max_element, fluid_error)`.
    - `rs::end(rng)` → `std::ranges::end(rng)`.
    - `rs::distance(...)` → `std::ranges::distance(...)`.
  - **Header file:** `scalar_wave.hpp` has no range-v3 usage — no changes needed.
  - Test: `cmake --build build` (pre-existing errors block full build)

### 5.3 — Migrate `inviscid_vortex.cpp`

- [x] **5.3** Migrate `src/systems/inviscid_vortex.cpp` (1 file, ~5 changed lines). ✓ Compiles successfully.
  - **Includes:**
    - Remove: `#include <range/v3/algorithm/max.hpp>` (line 9).
    - Remove: `#include <range/v3/view/transform.hpp>` (line 10). (Not used by active code; all `vs::transform` references are inside `#if 0` blocks.)
    - Add: `#include <algorithm>` (for `std::ranges::max`). Check if it's transitively available first; `fields/selector.hpp` may provide it, but add explicitly to be safe.
  - **Replacements (active code only — do NOT modify `#if 0` blocks):**
    - Line 237: `rs::max((max_abs(rhoU / rho, rhoV / rho) + sqrt(g * P / rho)) | sel::D)` → `std::ranges::max((max_abs(rhoU / rho, rhoV / rho) + sqrt(g * P / rho)) | sel::D)`.
    - The expression `... | sel::D` produces a domain-component range; `std::ranges::max` operates on it identically.
  - **Header file:** `inviscid_vortex.hpp` has no range-v3 usage — no changes needed.
  - **Ordering:** Independent.
  - Test: `cmake --build build` (no dedicated test for inviscid_vortex)

### 5.4 — Migrate `hyperbolic_eigenvalues.cpp`

- [x] **5.4** Migrate `src/systems/hyperbolic_eigenvalues.cpp` (1 file, ~5 changed lines). ✓ Compiles successfully.
  - **Includes:**
    - Remove: `#include <range/v3/algorithm/max.hpp>` (line 10). (Despite the name, `rs::min` is resolved through transitive range-v3 includes or this header.)
    - Add: `#include <algorithm>` (for `std::ranges::min`).
    - Add: `#include <ranges>` (for `std::views::transform`).
  - **Replacements:**
    - Line 39: `auto p = m.Rx() | vs::transform([this](auto&& info) {` → `auto p = m.Rx() | std::views::transform([this](auto&& info) {`. (`m.Rx()` returns `std::span<const mesh_object_info>`; `std::views::transform` works on any range.)
    - Line 45: `rs::min(v.eigenvalues_real())` → `std::ranges::min(v.eigenvalues_real())`. (`eigenvalues_real()` returns `std::span<const real>`, a valid `std::ranges::range`.)
  - **Header file:** `hyperbolic_eigenvalues.hpp` has no range-v3 usage — no changes needed.
  - **Ordering:** Independent.
  - Test: `cmake --build build && ctest --test-dir build -R t-hyperbolic_eigenvalues`

### 5.5 — Verify `system.cpp` and `empty_system.cpp`

- [x] **5.5** Verify `system.cpp` and `empty_system.cpp` have no range-v3 usage (0 changes needed). ✓ Verified.
  - Confirmed: neither file contains `#include <range/v3/...>`, `rs::`, or `vs::` calls.
  - `system.hpp` also clean (no range-v3).
  - `empty_system.hpp` also clean (no range-v3).
  - Test: build succeeds (covered by other items).

### 5.6 — Migrate test file `heat.t.cpp`

- [x] **5.6** Migrate `src/systems/heat.t.cpp` (1 file, ~12 changed lines). ✓ Compiles successfully (object file). Full link/test blocked by pre-existing `manufactured_solutions.hpp` errors in shoccs-system library (Phase 7).
  - **Includes:**
    - Remove: `#include <range/v3/all.hpp>` (line 9).
    - Add: `#include <algorithm>` (for `std::ranges::count`).
    - Add: `#include <numeric>` (for `std::accumulate`).
    - Add: `#include "fields/selector.hpp"` (for `sel::D` etc.; currently pulled in transitively through range/v3/all.hpp or system.hpp — verify, and add explicitly if needed).
  - **Replacements — `rs::count` (6 occurrences):**
    - Lines 84, 98, 192, 206, 301, 315: `rs::count(EXPR, VAL)` → `std::ranges::count(EXPR, VAL)`. Direct drop-in; `std::ranges::count` takes a range + value.
  - **Replacements — `rs::accumulate` (3 occurrences):**
    - C++20 has no `std::ranges::accumulate`; use `std::accumulate` with begin/end iterators.
    - Line 105: `real sum = rs::accumulate(u_rhs | sel::D, 0.0);` → introduce a named range variable, then:
      ```cpp
      auto d_rng = u_rhs | sel::D;
      real sum = std::accumulate(std::ranges::begin(d_rng), std::ranges::end(d_rng), 0.0);
      ```
    - Lines 213, 321 (inside `transform` lambda): `rs::accumulate(FWD(ui), 0.0)` → `std::accumulate(std::ranges::begin(ui), std::ranges::end(ui), 0.0)`.
  - **`hyperbolic_eigenvalues.t.cpp`** — already clean: no range-v3 includes or `rs::`/`vs::` usage. No changes needed.
  - **Ordering:** Should be done after or concurrently with 5.1 (heat.cpp), since both contribute to `t-heat` passing.
  - Test: `cmake --build build && ctest --test-dir build -R t-heat`

### 5.7 — Exclude `cc_elliptic.hpp/cpp`

- [x] **5.7** Exclude `cc_elliptic.hpp/cpp` — dead code (see `meta.md`). No action needed; these files are not in `CMakeLists.txt` and are excluded from migration scope. ✓ Verified.

---

## Completion Criteria

- All system tests pass: `ctest --test-dir build -L systems`
- No `#include <range/v3/...>` remains in `src/systems/` (excluding dead cc_elliptic).
- `grep -r 'range/v3' src/systems/` returns only cc_elliptic hits (or nothing if cc_elliptic is not present).

## Status Notes

- **Pre-existing build breakage:** `heat.cpp` and `scalar_wave.cpp` have pre-existing compilation errors from `manufactured_solutions.hpp` (Phase 7 — `vs::transform` view closures from MMS don't pipe through custom `tuple` type with range-v3 `view_closure`). These errors exist on the main branch before any Phase 5 changes. Full build and system tests (`t-heat`, `t-hyperbolic_eigenvalues`) cannot link until Phase 7 is completed.
- **Residual range-v3 in `scalar_wave.cpp`:** `#include <range/v3/view/transform.hpp>` remains because `neg_G<I>()` and `solution()` use `vs::transform` piped through mesh views (`m.xyz`, `m.vxyz`) that produce range-v3 types (`cartesian_product_view`). `std::views::transform` requires `std::ranges::viewable_range` which range-v3 views don't satisfy. This will be resolved when mesh views are migrated to standard types.
- **All migrated object files compile:** `heat.cpp.o`, `scalar_wave.cpp.o`, `inviscid_vortex.cpp.o`, `hyperbolic_eigenvalues.cpp.o`, `heat.t.cpp.o`, `hyperbolic_eigenvalues.t.cpp.o` all compile without new errors.

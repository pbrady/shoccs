# Phase 3: Stencils Subsystem

**Goal:** Migrate stencil coefficient construction from range-v3 view pipelines to standard C++ equivalents.

**Depends on:** Phase 0

**Read first:**
- `src/stencils/stencil.hpp` (no range-v3)
- `src/stencils/stencil.cpp` (no range-v3)
- `src/stencils/E2_1.cpp` (2509 lines — but only 1 zero-pad in constructor + 2 `ranges::reverse`; the bulk is coefficient tables)
- `src/stencils/E2_2.cpp` (light: `ranges::reverse` only; `fill.hpp` include is unused)
- `src/stencils/E4_2.cpp` (light: `ranges::reverse` only; `fill.hpp` include is unused)
- `src/stencils/E4u_1.cpp` (light: 1 zero-pad constructor + `ranges::reverse`; `fill.hpp`/`transform.hpp` includes unused)
- `src/stencils/E6u_1.cpp` (same pattern as E4u_1)
- `src/stencils/E8u_1.cpp` (same pattern as E4u_1)
- `src/stencils/polyE2_1.cpp` (moderate: 3 zero-pad constructors + `ranges::reverse`; unused includes)
- `src/stencils/CMakeLists.txt`
- `plans/meta.md`

**Test commands:**
```bash
cmake --build build
ctest --test-dir build -L stencils
```

---

## Actual range-v3 Usage Summary

After reading all source files, the actual range-v3 usage in library code is minimal:

| File | `concat\|repeat\|take` (zero-pad) | `ranges::reverse` | Unused includes |
|------|---|----|---|
| E2_2.cpp | — | 4 calls | `fill.hpp` |
| E4_2.cpp | — | 4 calls | `fill.hpp` |
| E4u_1.cpp | 1 (constructor) | 2 calls | `fill.hpp`, `transform.hpp` |
| E6u_1.cpp | 1 (constructor) | 2 calls | `fill.hpp`, `transform.hpp` |
| E8u_1.cpp | 1 (constructor) | 2 calls | `fill.hpp`, `transform.hpp` |
| polyE2_1.cpp | 3 (constructor) | 2 calls | `fill.hpp`, `transform.hpp` |
| E2_1.cpp | 1 (constructor) | 2 calls | `fill.hpp`, `transform.hpp` |

Test file range-v3 usage:
- **E4u_1.t.cpp, E6u_1.t.cpp, E8u_1.t.cpp, E2_1.t.cpp**: Only unused `#include <range/v3/view/zip.hpp>`.
- **E2_2.t.cpp, E4_2.t.cpp, polyE2_1.t.cpp**: Heavy usage (`vs::linear_distribute`, `rs::inner_product`, `vs::transform`, `vs::concat`, `rs::to<T>()`, `vs::drop`, `vs::take_exactly`). Per D2, tests keep range-v3.

---

## Items

### Common Pattern: `rs::copy(vs::concat(a, vs::repeat(0.0)) | vs::take(n), rs::begin(out))`

This pattern zero-pads a coefficient array `a` to length `n`. It appears in E2_1, E4u_1, E6u_1, E8u_1, and polyE2_1 constructors. Replace with a simple helper function:
```cpp
inline void copy_zero_padded(std::span<const real> src, std::span<real> dst) {
    auto n = std::min(src.size(), dst.size());
    std::copy_n(src.begin(), n, dst.begin());
    std::fill(dst.begin() + n, dst.end(), 0.0);
}
```

- [ ] **3.1** Add `copy_zero_padded` inline function to `src/stencils/stencil.hpp`.
  - Add `#include <algorithm>` to `stencil.hpp`.
  - Place the function in `namespace ccs::stencils`, before the `stencil` class.
  - Signature: `inline void copy_zero_padded(std::span<const real> src, std::span<real> dst)`.
  - `std::array<real, N>` implicitly converts to `std::span<real>` in C++20, so callers can pass arrays directly.
  - No ordering constraint: items 3.2–3.3 don't need this; items 3.4–3.8 depend on it.
  - Test: `cmake --build build` succeeds (no callers yet).

- [ ] **3.2** Migrate `E2_2.cpp`: replace `ranges::reverse` with `std::ranges::reverse`.
  - File: `src/stencils/E2_2.cpp`.
  - Remove: `#include <range/v3/algorithm/fill.hpp>` (unused), `#include <range/v3/algorithm/reverse.hpp>`.
  - Add: `#include <algorithm>`.
  - Replace all 4 occurrences of `ranges::reverse(...)` → `std::ranges::reverse(...)` (lines 134, 156, 188, 189).
  - Test: `ctest --test-dir build -R t-E2_2`

- [ ] **3.3** Migrate `E4_2.cpp`: replace `ranges::reverse` with `std::ranges::reverse`.
  - File: `src/stencils/E4_2.cpp`.
  - Remove: `#include <range/v3/algorithm/fill.hpp>` (unused), `#include <range/v3/algorithm/reverse.hpp>`.
  - Add: `#include <algorithm>`.
  - Replace all 4 occurrences of `ranges::reverse(...)` → `std::ranges::reverse(...)` (lines 235, 263, 306, 307 — in `nbs_floating`, `nbs_dirichlet`, `nbs_neumann`).
  - Test: `ctest --test-dir build -R t-E4_2`

- [ ] **3.4** Migrate `E4u_1.cpp`: replace zero-pad + `ranges::reverse`.
  - File: `src/stencils/E4u_1.cpp`.
  - **Depends on: 3.1.**
  - Remove all 7 range-v3 includes (lines 3–9).
  - Add: `#include <algorithm>`.
  - Replace constructor body (lines 29–30): `rs::copy(vs::concat(a, vs::repeat(0.0)) | vs::take(alpha.size()), rs::begin(alpha));` → `copy_zero_padded(a, alpha);`.
  - Replace 2 occurrences of `ranges::reverse(c)` → `std::ranges::reverse(c)` (in `nbs_floating` line 106, `nbs_dirichlet` line 129).
  - Test: `ctest --test-dir build -R t-E4u_1`

- [ ] **3.5** Migrate `E6u_1.cpp`: same pattern as 3.4.
  - File: `src/stencils/E6u_1.cpp`.
  - **Depends on: 3.1.**
  - Remove all 7 range-v3 includes (lines 3–9).
  - Add: `#include <algorithm>`.
  - Replace constructor body (lines 29–30): → `copy_zero_padded(a, alpha);`.
  - Replace 2 `ranges::reverse(c)` → `std::ranges::reverse(c)` (in `nbs_floating` line 146, `nbs_dirichlet` line 213).
  - Test: `ctest --test-dir build -R t-E6u_1`

- [ ] **3.6** Migrate `E8u_1.cpp`: same pattern as 3.4.
  - File: `src/stencils/E8u_1.cpp`.
  - **Depends on: 3.1.**
  - Remove all 7 range-v3 includes (lines 3–9).
  - Add: `#include <algorithm>`.
  - Replace constructor body (lines 29–30): → `copy_zero_padded(a, alpha);`.
  - Replace 2 `ranges::reverse(c)` → `std::ranges::reverse(c)` (in `nbs_floating` line 197, `nbs_dirichlet` line 301).
  - Test: `ctest --test-dir build -R t-E8u_1`

- [ ] **3.7** Migrate `polyE2_1.cpp`: 3 zero-pad replacements + `ranges::reverse`.
  - File: `src/stencils/polyE2_1.cpp`.
  - **Depends on: 3.1.**
  - Remove all 7 range-v3 includes (lines 3–9).
  - Add: `#include <algorithm>`.
  - Replace 3 constructor lines (31–33):
    - `rs::copy(vs::concat(fa_, vs::repeat(0.0)) | vs::take(fa.size()), rs::begin(fa));` → `copy_zero_padded(fa_, fa);`
    - `rs::copy(vs::concat(da_, vs::repeat(0.0)) | vs::take(da.size()), rs::begin(da));` → `copy_zero_padded(da_, da);`
    - `rs::copy(vs::concat(ia_, vs::repeat(0.0)) | vs::take(ia.size()), rs::begin(ia));` → `copy_zero_padded(ia_, ia);`
  - Replace 2 `ranges::reverse(c)` → `std::ranges::reverse(c)` (in `nbs_floating` line 209, `nbs_dirichlet` line 253).
  - Test: `ctest --test-dir build -R t-polyE2_1`

- [ ] **3.8** Migrate `E2_1.cpp` (2509 lines — largest file, but simple range-v3 usage).
  - File: `src/stencils/E2_1.cpp`.
  - **Depends on: 3.1.**
  - Despite 2509 lines, the range-v3 usage is only: 1 zero-pad in constructor (line 27–28) + 2 `ranges::reverse` calls (lines 1311, 2498). The rest is hand-written coefficient tables.
  - Remove all 7 range-v3 includes (lines 3–9).
  - Add: `#include <algorithm>`.
  - Replace constructor body: → `copy_zero_padded(a, alpha);`.
  - Replace 2 `ranges::reverse(c)` → `std::ranges::reverse(c)` (line 1311 in `nbs_floating`, line 2498 in `nbs_dirichlet`).
  - Test: `ctest --test-dir build -R t-E2_1`

- [ ] **3.9** Clean up CMakeLists.txt and test file includes. Verify no library dependency on range-v3 remains.
  - **Depends on: 3.2–3.8 all complete.**
  - **3.9a** Remove unused `#include <range/v3/view/zip.hpp>` from 4 test files:
    - `src/stencils/E4u_1.t.cpp` (line 9)
    - `src/stencils/E6u_1.t.cpp` (line 9)
    - `src/stencils/E8u_1.t.cpp` (line 9)
    - `src/stencils/E2_1.t.cpp` (line 9)
  - **3.9b** Keep range-v3 includes in 3 test files that actively use them (per D2):
    - `src/stencils/E2_2.t.cpp` — uses `vs::linear_distribute`, `rs::inner_product`, `vs::transform`, `vs::concat`, `rs::to`, `rs::fill`
    - `src/stencils/E4_2.t.cpp` — uses `vs::transform`, `vs::linear_distribute`, `rs::inner_product`, `vs::concat`, `vs::drop`, `vs::take_exactly`, `rs::to`
    - `src/stencils/polyE2_1.t.cpp` — uses `vs::transform`, `vs::linear_distribute`, `rs::inner_product`, `vs::concat`, `vs::drop`, `vs::take_exactly`, `rs::to`
  - **3.9c** Update `src/stencils/CMakeLists.txt`:
    - Remove `PRIVATE range-v3::range-v3` from `target_link_libraries(shoccs-stencils ...)`.
    - Remove `range-v3::range-v3` from `add_unit_test()` calls for E2_1, E4u_1, E6u_1, E8u_1 (they no longer use it).
    - Keep `range-v3::range-v3` in `add_unit_test()` calls for E2_2, E4_2, polyE2_1 (tests still use it).
  - Test: `ctest --test-dir build -L stencils` — all 7 pass.

---

## Completion Criteria

- All 7 stencil test files pass.
- No `#include <range/v3/...>` remains in stencil **library** source files (`E2_1.cpp`, `E2_2.cpp`, `E4_2.cpp`, `E4u_1.cpp`, `E6u_1.cpp`, `E8u_1.cpp`, `polyE2_1.cpp`, `stencil.hpp`, `stencil.cpp`).
- No `#include <range/v3/...>` remains in 4 test files that had only unused includes (`E2_1.t.cpp`, `E4u_1.t.cpp`, `E6u_1.t.cpp`, `E8u_1.t.cpp`).
- 3 test files (`E2_2.t.cpp`, `E4_2.t.cpp`, `polyE2_1.t.cpp`) retain range-v3 per D2 (tests can keep range-v3).
- The `shoccs-stencils` library target no longer links `range-v3::range-v3`.

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

- [ ] **4.1** Migrate `derivative.cpp` — `OB_builder` struct (lines 69–154):
  - File: `src/operators/derivative.cpp`
  - Remove `#include <range/v3/all.hpp>` (line 4), add `<algorithm>`, `<numeric>`, `<ranges>`.
  - Line 73: `template <rs::random_access_range R>` → `template <std::ranges::random_access_range R>`
  - Line 77: `for (auto&& [i, v] : vs::enumerate(r) | vs::drop(1))` → indexed `for` loop starting at index 1:
    ```cpp
    for (int i = 1; i < std::ranges::ssize(r); ++i)
        O.add_point(shape_row, solid_ic + i * stride, r[i]);
    ```
  - Line 88: `template <rs::random_access_range R>` → `template <std::ranges::random_access_range R>`
  - Line 101: `auto it = rs::begin(interp_coeffs);` → `auto it = std::ranges::begin(interp_coeffs);`
  - Test: `ctest --test-dir build -R t-derivative`

- [ ] **4.2** Migrate `derivative.cpp` — `cut_discretization` function (lines 156–253):
  - Line 170: `rs::accumulate(obj_bcs, true, [](auto&& acc, auto&& cur) { return acc && (cur == bcs::Dirichlet); })` → `std::ranges::all_of(obj_bcs, [](auto bc) { return bc == bcs::Dirichlet; })`
  - Line 188: `for (auto&& [shape_row, obj] : vs::enumerate(shapes))` → `for (integer shape_row = 0; shape_row < (integer)sz; ++shape_row) { const auto& obj = shapes[shape_row]; ... }`
  - Lines 197–198: `c | vs::drop_exactly((rObj - 1) * tObj) | vs::take_exactly(tObj) | vs::reverse` → create a reversed copy:
    ```cpp
    auto sub = std::span{c}.subspan((rObj - 1) * tObj, tObj);
    std::vector<real> rng(sub.rbegin(), sub.rend());
    ```
    (Pass `std::span{rng}` to `add_cut_row`, which accepts any `random_access_range`.)
  - Line 201: `c | vs::take_exactly(tObj)` → `std::span{c}.subspan(0, tObj)` (already a `random_access_range`).
  - Line 206: `for (auto&& [shape_row, obj] : vs::enumerate(shapes))` → same indexed loop as line 188.
  - Test: `ctest --test-dir build -R t-derivative`

- [ ] **4.3** Migrate `derivative.cpp` — `domain_discretization` function (lines 301–438):
  - **Left boundary with object (lines 347–356):**
    - Line 347: `auto lc = left | vs::drop(s * tLeft);` → `auto lc = std::span{left}.subspan(s * tLeft);`
    - Lines 348–349: `lc | vs::chunk(tLeft) | vs::for_each(vs::drop(1))` (skip first column of each row, flatten) → explicit vector construction:
      ```cpp
      std::vector<real> dense_data;
      dense_data.reserve(rLeft * (tLeft - 1));
      for (int row = 0; row < rLeft; ++row) {
          auto row_span = lc.subspan(row * tLeft + 1, tLeft - 1);
          dense_data.insert(dense_data.end(), row_span.begin(), row_span.end());
      }
      leftMat = matrix::dense{rLeft, tLeft - 1, dense_data};
      ```
    - Lines 354–356: `lc | vs::stride(tLeft) | vs::take(rLeft)` with `vs::enumerate` → explicit strided loop:
      ```cpp
      for (int row = 0; row < rLeft; ++row) {
          B_builder.add_point(sub.left_row(row), obj->object_coordinate, lc[row * tLeft]);
      }
      ```
  - **Right boundary with object (lines 387–401):**
    - Line 387: `right | vs::take_exactly(rRight * tRight)` → `auto rc = std::span{right}.subspan(0, rRight * tRight);`
    - Lines 391–392: `rc | vs::chunk(tRight) | vs::for_each(vs::take(tRight - 1))` (take first (tRight-1) columns of each row, flatten) → explicit vector:
      ```cpp
      std::vector<real> dense_data;
      dense_data.reserve(rRight * (tRight - 1));
      for (int row = 0; row < rRight; ++row) {
          auto row_span = rc.subspan(row * tRight, tRight - 1);
          dense_data.insert(dense_data.end(), row_span.begin(), row_span.end());
      }
      rightMat = matrix::dense{rRight, tRight - 1, dense_data};
      ```
    - Lines 396–398: `rc | vs::drop(tRight - 1) | vs::stride(tRight) | vs::take(rRight)` with `vs::enumerate` → explicit strided loop (take last element of each row):
      ```cpp
      for (int row = 0; row < rRight; ++row) {
          auto val = rc[row * tRight + tRight - 1];
          B_builder.add_point(sub.right_row(row - rRight), obj->object_coordinate, val);
      }
      ```
  - Test: `ctest --test-dir build -R t-derivative`

- [ ] **4.4** Verify `derivative.hpp` has no range-v3 usage:
  - File: `src/operators/derivative.hpp`
  - **Already verified clean**: No `#include <range/v3/...>`, no `rs::`, no `vs::` references.
  - No changes needed; mark complete after confirming build.
  - Test: `ctest --test-dir build -R t-derivative`

### Gradient and Laplacian (Trivial)

- [ ] **4.5** Migrate `gradient.cpp` — replace `vs::repeat_n`:
  - File: `src/operators/gradient.cpp`
  - `gradient.cpp` has no explicit `#include <range/v3/...>` — `vs::repeat_n` resolves transitively.
  - Line 18: `fmt::join(vs::repeat_n("wall,psi", st_info.t - 1), ",")` → build a `std::vector<std::string>`:
    ```cpp
    std::vector<std::string> hdr(st_info.t - 1, "wall,psi");
    fmt::join(hdr, ",")
    ```
    Alternatively, use a simple loop to build the comma-separated string directly.
  - Add `#include <string>` and `#include <vector>` if not already present.
  - Test: `ctest --test-dir build -R t-gradient`

- [ ] **4.6** Migrate `laplacian.cpp` — replace `vs::repeat_n`:
  - File: `src/operators/laplacian.cpp`
  - Remove `#include <range/v3/view/repeat_n.hpp>` (line 5).
  - Line 22: same `vs::repeat_n("wall,psi", st_info.t - 1)` pattern → same replacement as 4.5.
  - Test: `ctest --test-dir build -R t-laplacian`

### Eigenvalue Visitor (No range-v3)

- [ ] **4.7** Verify `eigenvalue_visitor.hpp` and `eigenvalue_visitor.cpp` have no range-v3 usage:
  - **Already verified clean**: Neither file includes range-v3 or uses `rs::`/`vs::`.
  - No changes needed; mark complete after confirming build.
  - Test: `ctest --test-dir build -R t-eigenvalue_visitor`

### Boundaries (No range-v3)

- [ ] **4.8** Verify `boundaries.hpp` and `boundaries.cpp` have no range-v3 usage:
  - **Already verified clean**: Neither file includes range-v3 or uses `rs::`/`vs::`.
  - No changes needed; mark complete after confirming build.
  - Test: `ctest --test-dir build -R t-boundaries`

### Test Migration

- [ ] **4.9** Migrate `derivative.t.cpp`:
  - File: `src/operators/derivative.t.cpp`
  - Remove `#include <range/v3/all.hpp>` (line 12), add `#include <ranges>` and `#include <algorithm>`.
  - **`vs::transform` (7 declarations, lines 59–81):** `constexpr auto f2 = vs::transform(...)` → `constexpr auto f2 = std::views::transform(...)`. Same for f2_dx, f2_dy, f2_dz, f2_ddx, f2_ddy, f2_ddz.
  - **`vs::transform` inline (line 332):** `m.xyz | vs::transform([](auto&&) { return pick(); })` → `m.xyz | std::views::transform([](auto&&) { return pick(); })`.
  - **`vs::generate_n` (lines 148, 206):** `u | sel::D = vs::generate_n(g, m.size())` → generate into a temporary vector:
    ```cpp
    { std::vector<real> tmp(m.size()); std::generate_n(tmp.begin(), m.size(), g); u | sel::D = tmp; }
    ```
  - **`rs::size` (10 uses, lines 151, 179, 219, 241, 275, 300, 334, 339, 371, 386):** `rs::size(...)` → `std::ranges::size(...)`.
  - Test: `ctest --test-dir build -R t-derivative`

- [ ] **4.10** Migrate `gradient.t.cpp`:
  - File: `src/operators/gradient.t.cpp`
  - Remove `#include <range/v3/all.hpp>` (line 10), add `#include <ranges>`.
  - **`vs::transform` (7 declarations, lines 21–53):** `vs::transform(...)` → `std::views::transform(...)`. For f2, f2_dx, f2_dy, f2_dz, g, gx, gy.
  - **`rs::size` (4 uses, lines 81, 142, 219, 297):** `rs::size(...)` → `std::ranges::size(...)`.
  - Test: `ctest --test-dir build -R t-gradient`

- [ ] **4.11** Migrate `laplacian.t.cpp`:
  - File: `src/operators/laplacian.t.cpp`
  - Remove `#include <range/v3/all.hpp>` (line 15), add `#include <ranges>`.
  - **`vs::transform` (13 declarations, lines 23–83):** `vs::transform(...)` → `std::views::transform(...)`. For f2, f2_dx, f2_dy, f2_dz, f2_ddx, f2_ddy, f2_ddz, g2, g2_dx, g2_dy, g2_ddx, g2_ddy.
  - **`rs::size` (5 uses, lines 111, 133, 195, 269, 347):** `rs::size(...)` → `std::ranges::size(...)`.
  - Test: `ctest --test-dir build -R t-laplacian`

- [ ] **4.12** Migrate `eigenvalue_visitor.t.cpp`:
  - File: `src/operators/eigenvalue_visitor.t.cpp`
  - No `#include <range/v3/...>` present, but uses `rs::max` (line 114).
  - Line 114: `rs::max(eigs)` → `std::ranges::max(eigs)`. Add `#include <algorithm>` and `#include <ranges>` if needed.
  - Note: `to<T>(...)` on lines 57, 111 is project-local (`fields/tuple_utils.hpp`), NOT range-v3. No change needed.
  - Test: `ctest --test-dir build -R t-eigenvalue_visitor`

- [ ] **4.13** Verify `boundaries.t.cpp` has no range-v3 usage:
  - **Already verified clean**: No range-v3 includes, no `rs::`/`vs::` usage.
  - No changes needed.
  - Test: `ctest --test-dir build -R t-boundaries`

### Remove range-v3 from CMake

- [ ] **4.14** Verify no range-v3 link dependency in `src/operators/CMakeLists.txt`:
  - **Already verified**: `shoccs-operators` links `shoccs-mesh shoccs-matrices shoccs-logging lapackpp`. No direct `range-v3::range-v3` dependency.
  - `shoccs-bcs` links `sol2::sol2 lua shoccs-logging`. No range-v3 dependency.
  - Range-v3 headers were only pulled in via explicit `#include <range/v3/...>` in source files (derivative.cpp, laplacian.cpp) and test files. Once those includes are removed in items 4.1–4.12, no CMake changes are needed.
  - Test: `cmake --build build && ctest --test-dir build -L operators && ctest --test-dir build -L bcs`

---

## Ordering Constraints

- Items 4.1–4.3 (derivative.cpp) must be done together or sequentially — the `#include <range/v3/all.hpp>` removal in 4.1 affects all three. Recommended: do 4.1 first (it removes the include and adds std replacements), then 4.2 and 4.3 in the same pass.
- Items 4.5, 4.6 (gradient.cpp, laplacian.cpp) are independent of 4.1–4.3.
- Items 4.9–4.12 (test migration) should be done after their corresponding source files (4.1–4.8), since tests must compile against the migrated headers.
- Items 4.4, 4.7, 4.8, 4.13, 4.14 are verification-only and can be marked complete immediately.

---

## Completion Criteria

- All 5 operator test files pass (excluding disabled directional).
- No `#include <range/v3/...>` remains in `src/operators/` (excluding disabled files).
- No `rs::` or `vs::` usage remains in `src/operators/` source or test files.
- `shoccs-operators` and `shoccs-bcs` libraries no longer depend on range-v3.

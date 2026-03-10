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
- `src/mesh/mesh.hpp` + `mesh.cpp` (light: rs::begin/end, vs::transform)
- `src/mesh/mesh_view.hpp` (already migrated — plain loops, no range-v3 or cppcoro)
- `src/mesh/object_geometry.hpp` + `object_geometry.cpp` (light: vs::transform)
- `src/fields/lazy_views.hpp` (project-local utilities: ccs::zip_transform, ccs::repeat_n, ccs::stride)
- `src/fields/ccs_range_utils.hpp` (project-local utilities: ccs::make_view_closure, ccs::bind_back, ccs::semiregular_box)
- `plans/meta.md`

**Test commands:**
```bash
cmake --build build
ctest --test-dir build
```

---

## Items

### Mesh (High Complexity)

- [ ] **7.1** Migrate `cartesian.hpp` and `cartesian.cpp`
  - **7.1a** Add `ccs::cartesian_product_view` to `src/fields/lazy_views.hpp` (see decision D9 in `plans/meta.md`).
    - A lazy view over three ranges yielding `std::tuple<T1&, T2&, T3&>` in triple-nested-loop order (first range slowest, third fastest).
    - Model `std::ranges::view_interface` with at least forward iteration.
    - Add factory `ccs::cartesian_product(r1, r2, r3)`.
    - File: `src/fields/lazy_views.hpp` (append after `stride_view`).
    - Test: unit test or compile check in 7.1b.
  - **7.1b** Migrate `cartesian.hpp` (line 10, 74):
    - Remove `#include <range/v3/view/cartesian_product.hpp>`.
    - Add `#include "fields/lazy_views.hpp"`.
    - Line 74: Replace `vs::cartesian_product(x(), y(), z())` with `ccs::cartesian_product(x(), y(), z())`.
    - File: `src/mesh/cartesian.hpp`.
  - **7.1c** Add shared `ccs::linear_distribute` helper to `src/fields/lazy_views.hpp`:
    - A free function: `template<typename T> std::vector<T> linear_distribute(T mn, T mx, int n)` that generates `n` linearly-spaced values from `mn` to `mx`.
    - Implementation: `v[i] = n > 1 ? mn + i * (mx - mn) / (n - 1) : mn`.
    - File: `src/fields/lazy_views.hpp` (append after `stride_view`).
    - Used by: 7.1d (cartesian.cpp) and 7.13a/b/c (stencil tests).
    - Ordering: Must precede 7.1d and 7.13.
  - **7.1d** Migrate `cartesian.cpp` constructor (lines 3, 13–34):
    - Remove `#include <range/v3/all.hpp>` (line 3), add `#include <algorithm>`, `#include <numeric>`, and `#include "fields/lazy_views.hpp"`.
    - Lines 13–15 (`concat_copy` with `rs::copy`, `vs::concat`, `vs::repeat`, `vs::take`): Replace with an explicit loop that copies up to 3 values from `in`, padding with `val`:
      ```cpp
      auto concat_copy = [](auto&& in, auto val, auto&& out) {
          int i = 0;
          for (auto it = std::ranges::begin(in); i < 3 && it != std::ranges::end(in); ++it, ++i)
              out[i] = *it;
          for (; i < 3; ++i) out[i] = val;
      };
      ```
    - Line 20 (`n | vs::transform(…)`): Inline into the concat_copy call with a manual transform:
      ```cpp
      int3 clamped;
      for (int i = 0; i < std::min((int)std::ranges::size(n), 3); ++i)
          clamped[i] = n[i] > 0 ? n[i] : 1;
      for (int i = std::ranges::size(n); i < 3; ++i) clamped[i] = 1;
      n_ = clamped; // or use concat_copy
      ```
    - Lines 22–28 (`vs::zip_with` for h_): Replace with explicit loop:
      ```cpp
      for (int i = 0; i < 3; ++i)
          h_[i] = (n_[i] - 1) ? (max_[i] - min_[i]) / (n_[i] - 1) : null_v<>;
      ```
    - Line 30 (`rs::count_if`): Replace with `std::count_if(n_.begin(), n_.end(), …)`.
    - Lines 32–34 (`vs::linear_distribute | rs::to<vector>`): Replace with `ccs::linear_distribute(min_[i], max_[i], n_[i])` (from 7.1c).
    - File: `src/mesh/cartesian.cpp`.
  - Test: `ctest --test-dir build -R t-cartesian`
  - Ordering: 7.1a must precede 7.1b. 7.1c must precede 7.1d. 7.1d is independent of 7.1a/7.1b.

- [ ] **7.2** Migrate `selections.hpp`
  - **7.2a** Rewrite `YPlaneView` as a `std::ranges::view_interface` class (lines 26–163):
    - Replace `rs::view_adaptor<YPlaneView<Rng>, Rng>` inheritance with `std::ranges::view_interface<YPlaneView<Rng>>`.
    - Store the base range directly (not via `view_adaptor`).
    - Replace inner `adaptor : rs::adaptor_base` with a standalone `iterator` class implementing the same `next`/`prev`/`advance`/`distance_to` logic but as C++20 iterator operations (`operator++`, `operator--`, `operator+=`, `operator-`, `operator==`, `operator<=>`).
    - Replace `rs::begin(rng.base())` / `rs::advance(it, n)` with `std::ranges::begin(base_)` / `std::ranges::advance(it, n)`.
    - Replace `rs::range_difference_t<Rng>` with `std::ranges::range_difference_t<Rng>`.
    - Replace `rs::difference_type_t<I>` with `std::iter_difference_t<I>` (or use the iterator's own `difference_type`).
    - Update `y_plane_fn` (line 154–161): Replace `rs::make_view_closure(rs::bind_back(...))` with `ccs::make_view_closure(ccs::bind_back(...))`.
    - Add `#include "fields/ccs_range_utils.hpp"` to selections.hpp.
    - File: `src/mesh/selections.hpp` lines 26–163.
  - **7.2b** Replace `plane_fn<0>` and `plane_fn<2>` range-v3 adaptors (lines 170–202):
    - `plane_fn<0>` line 174: Replace `vs::take_exactly(n)` with `std::views::take(n)`.
    - `plane_fn<0>` line 179: Replace `vs::drop_exactly(n)` with `std::views::drop(n)`.
    - `plane_fn<2>` line 195: Replace `vs::stride(n)` with a pipeable adaptor using `ccs::stride`:
      ```cpp
      return ccs::make_view_closure([n = extents[2]](auto&& rng) {
          return ccs::stride(FWD(rng), n);
      });
      ```
    - `plane_fn<2>` line 200: Replace `vs::drop_exactly(k)` with `std::views::drop(k)`.
    - Add `#include "fields/lazy_views.hpp"` to selections.hpp.
    - File: `src/mesh/selections.hpp` lines 170–202.
  - **7.2c** Rewrite `FView` as a `std::ranges::view_interface` class (lines 270–451):
    - Same approach as 7.2a: replace `rs::view_adaptor` inheritance, `rs::adaptor_base`, `rs::range_access` with `view_interface` + standalone iterator.
    - Replace `rs::begin(rng.base())`, `rs::advance(it, n)` with `std::ranges::begin(base_)`, `std::ranges::advance(it, n)`.
    - Update `fview_fn` (lines 441–448): Replace `rs::make_view_closure(rs::bind_back(...))` with `ccs::make_view_closure(ccs::bind_back(...))`.
    - File: `src/mesh/selections.hpp` lines 270–451.
  - **7.2d** Replace remaining `rs::make_view_closure` in utility functions (lines 207–264, 454–459):
    - Functions `xmin`, `xmax`, `ymin`, `ymax`, `zmin`, `zmax` (lines 207–247): Replace `rs::make_view_closure` with `ccs::make_view_closure`.
    - Function `location` (lines 249–264): Replace `rs::make_view_closure` with `ccs::make_view_closure`. Replace `vs::cartesian_product(…)` with `ccs::cartesian_product(…)` (from 7.1a). Replace `vs::transform(…)` with `std::views::transform(…)`.
    - Function `F` (lines 454–459): Replace `rs::make_view_closure` with `ccs::make_view_closure`.
    - Remove all 7 `#include <range/v3/…>` headers (lines 8–14).
    - File: `src/mesh/selections.hpp`.
  - Test: `ctest --test-dir build -R t-mesh`
  - Ordering: 7.2a/7.2b/7.2c are independent of each other. 7.2d depends on 7.1a (for `ccs::cartesian_product`) and should be done last.

- [ ] **7.3** Migrate `object_geometry.hpp` (lines 10, 65–66):
  - Remove `#include <range/v3/view/transform.hpp>`, add `#include <ranges>`.
  - Line 65: Replace `vs::transform(&mesh_object_info::position)` with `std::views::transform(&mesh_object_info::position)`.
  - Line 66: Replace pipe `Rx() | t` etc. — these use `std::views::transform` which is already pipeable in C++20.
  - File: `src/mesh/object_geometry.hpp`.
  - Test: `ctest --test-dir build -R t-object_geometry`

- [ ] **7.4** Migrate `mesh.cpp` and `mesh.hpp`
  - **7.4a** `mesh.cpp` `init_line` function (lines 47–48, 73, 81):
    - Replace `rs::begin(r)` with `r.begin()` (or `std::ranges::begin(r)`).
    - Replace `rs::end(r)` with `r.end()` (or `std::ranges::end(r)`).
    - Lines 73, 81: Replace `first - rs::begin(r)` with `first - r.begin()`.
    - File: `src/mesh/mesh.cpp`.
  - **7.4b** `mesh.hpp` `object_boundaries` method (line 48):
    - Replace `vs::transform(…)` with `std::views::transform(…)`.
    - No new include needed (mesh.hpp includes selections.hpp which will include `<ranges>`).
    - File: `src/mesh/mesh.hpp`.
  - Test: `ctest --test-dir build -R t-mesh`

- [ ] **7.5** Verify `mesh_view.hpp` — already migrated.
  - `mesh_view.hpp` uses plain loops returning `std::vector<real3>` — no range-v3 or cppcoro.
  - The `mesh_view.t.cpp` test is **commented out** in `src/mesh/CMakeLists.txt` (line 10). No action needed unless we uncomment it (see 7.12b).
  - Test: build succeeds (no test to run).

### MMS (Medium Complexity)

- [ ] **7.6** Migrate `manufactured_solutions.hpp` (lines 9, 205–245):
  - Remove `#include <range/v3/view/transform.hpp>`, add `#include <ranges>`.
  - Replace 6 methods that return `vs::transform(lambda)` with `std::views::transform(lambda)`:
    - `operator()(real time)` — line 208
    - `ddt(real time)` — line 215
    - `gradient(real time)` — line 222
    - `gradient(int i, real time)` — line 229
    - `divergence(real time)` — line 236
    - `laplacian(real time)` — line 243
  - Each is a one-line change: `vs::transform(…)` → `std::views::transform(…)`.
  - File: `src/mms/manufactured_solutions.hpp`.
  - Test: `ctest --test-dir build -R t-mms`

- [ ] **7.7** Migrate `mms.cpp` (lines 9, 43–44):
  - Remove `#include <range/v3/action/transform.hpp>`, add `#include <algorithm>` and `#include <cctype>`.
  - Line 43–44: Replace `str | rs::action::transform([](auto c) { return std::tolower(c); })` with:
    ```cpp
    std::transform(ms_t.begin(), ms_t.end(), ms_t.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    ```
    Note: cast to `unsigned char` for `std::tolower` safety.
  - Remove `PRIVATE range-v3::range-v3` from `src/mms/CMakeLists.txt` line 4.
  - Files: `src/mms/mms.cpp`, `src/mms/CMakeLists.txt`.
  - Test: `ctest --test-dir build -R t-mms`

### I/O (Low-Medium Complexity)

- [ ] **7.8** Migrate `field_io.cpp` (lines 10–11, 52–55, 64–66):
  - Remove `#include <range/v3/range/conversion.hpp>` and `#include <range/v3/view/transform.hpp>`.
  - Lines 52–55: Replace `names | vs::transform(…) | rs::to<vector<string>>()` with an explicit loop:
    ```cpp
    std::vector<std::string> xmf_file_names;
    xmf_file_names.reserve(names.size());
    for (auto&& name : names)
        xmf_file_names.push_back(fmt::format("{}.{:0{}d}", name, n, suffix_length));
    ```
  - Lines 64–66: Replace `xmf_file_names | vs::transform(…) | rs::to<vector<string>>()` with same pattern:
    ```cpp
    std::vector<std::string> data_file_names;
    data_file_names.reserve(xmf_file_names.size());
    for (auto&& name : xmf_file_names)
        data_file_names.push_back(io / name);
    ```
  - File: `src/io/field_io.cpp`.
  - Test: `ctest --test-dir build -R t-field_io`

- [ ] **7.9** Migrate `field_data.cpp` (lines 4–6, 19–31, 43, 50–52):
  - Remove all three `#include <range/v3/…>` headers (lines 4–6: `for_each`, `reverse_copy` (stale/unused), `transform`).
  - **`write_geom` method** (lines 16–37):
    - Lines 19–31: Replace `rs::for_each(rng | vs::transform(&mesh_object_info::position), lambda)` with a range-for loop:
      ```cpp
      for (auto&& info : rng) {
          auto&& pos = info.position;
          // ... same body as the lambda
      }
      ```
    - Lines 25, 29: Replace `rs::size(tmp)` and `rs::size(pos)` with `tmp.size()` and `pos.size()` (both are `real3` = `std::array<real,3>`, which has `.size()`).
  - **`write` method** (lines 39–59):
    - Line 43: Replace `vs::zip(filenames, f.scalars())` with an index-based loop:
      ```cpp
      auto& scalars = f.scalars();
      for (size_t idx = 0; idx < filenames.size(); ++idx) {
          auto& fname = filenames[idx];
          auto& sc = scalars[idx];
          // ... rest of body
      }
      ```
    - Lines 50, 52: Replace `rs::size(rng)` with `rng.size()` or `std::ranges::size(rng)`.
  - File: `src/io/field_data.cpp`.
  - Test: `ctest --test-dir build -R t-field_io`

- [ ] **7.10** Migrate `xdmf.cpp` (lines 12, 51, 69, 84, 86–88, 128–130):
  - Remove `#include <range/v3/view/zip.hpp>`, add `#include "fields/lazy_views.hpp"`.
  - Line 51 (`rs::size(rng)`): Replace with `std::ranges::size(rng)`.
  - Line 69 (`vs::zip(var_names, file_names)`): Replace with an index-based loop:
    ```cpp
    for (size_t i = 0; i < var_names.size(); ++i) {
        auto& v = var_names[i];
        auto& f = file_names[i];
        // ... rest of body
    }
    ```
  - Line 84 (`vs::repeat_n(0, f_sz)`): Replace with `ccs::repeat_n(0, f_sz)` (from `lazy_views.hpp`).
  - Lines 86–88, 128–130 (`rs::size(x)`, `rs::size(get<0>(t))`, etc.): Replace with `std::ranges::size(…)` or `.size()` on spans.
  - File: `src/io/xdmf.cpp`.
  - Test: `ctest --test-dir build -R t-xdmf`

### Simulation (Low Complexity)

- [ ] **7.11** Verify `simulation_cycle.cpp` and `simulation_builder.cpp` have no range-v3 usage.
  - Both files are confirmed clean — no `#include <range/v3/…>`, no `rs::`, no `vs::`.
  - Test: build succeeds.

### Earlier-Phase Leftover Cleanup

These files still have range-v3 usage from earlier phases and must be cleaned before final removal.

- [ ] **7.12** Migrate `src/systems/scalar_wave.cpp` (lines 14, 30, 37):
  - Remove `#include <range/v3/view/transform.hpp>`, add `#include <ranges>`.
  - Lines 30, 37: Replace `vs::transform(lambda)` with `std::views::transform(lambda)`.
  - These are used in the `initial_condition` and `exact` methods which return pipeable view closures.
  - File: `src/systems/scalar_wave.cpp`.
  - Test: `ctest --test-dir build -R t-simulation_cycle` (scalar_wave is used by simulation tests).

- [ ] **7.13** Migrate stencil test files (heavy range-v3: `vs::linear_distribute`, `rs::inner_product`, `vs::concat`, `vs::single`, `rs::to`, `rs::fill`, `vs::take_exactly`, `vs::drop`):
  - **7.13a** `src/stencils/polyE2_1.t.cpp`: Replace all range-v3 patterns with explicit loops / `std::inner_product` / manual vector construction. Remove `#include <range/v3/all.hpp>`, add `#include <numeric>`, `#include <ranges>`, `#include "fields/lazy_views.hpp"`.
    - Lines 19, 24: `constexpr auto gt = vs::transform(gf)` and `constexpr auto bt = vs::transform(bf)` — change `constexpr` to `const` and replace `vs::transform` with `std::views::transform`. These are namespace-scope pipeable closure variables used as `mesh | gt`.
    - `vs::linear_distribute(a, b, n) | rs::to<T>()` → `ccs::linear_distribute(a, b, n)` (from 7.1c; already returns `std::vector`).
    - `rs::inner_product(a, b, init)` → `std::inner_product(a.begin(), a.end(), b.begin(), init)` (from `<numeric>`).
    - `vs::concat(vs::single(x), mesh) | rs::to<T>()` → construct vector manually: `T m = {x}; m.insert(m.end(), mesh.begin(), mesh.end());`.
    - `c | vs::drop(i * t) | vs::take_exactly(t)` → `std::span(c).subspan(i * t, t)`.
    - Inline `vs::transform(f)` in pipelines → `std::views::transform(f)`.
    - Remove `range-v3::range-v3` link from `src/stencils/CMakeLists.txt` line 21.
  - **7.13b** `src/stencils/E2_2.t.cpp`: Same patterns as 7.13a (no constexpr globals, but has inline `vs::transform(f)` in pipelines at lines 127, 172).
    - Also replace `rs::fill(cw, 0.0)` → `std::ranges::fill(cw, 0.0)` (line 194).
    - Note: lines inside `#if 0` blocks (209–284, 287–453) can be migrated or left since they're compiled out.
    - Remove `range-v3::range-v3` link from `src/stencils/CMakeLists.txt` line 14.
  - **7.13c** `src/stencils/E4_2.t.cpp`: Same patterns as 7.13a.
    - Lines 23, 27, 30: `constexpr auto f4 = vs::transform(f4_f)`, `constexpr auto f3 = vs::transform(f3_f)`, `constexpr auto f2 = vs::transform(f2_f)` — change `constexpr` to `const` and replace `vs::transform` with `std::views::transform`. Used as pipeable closures in `mesh | f4`, `m | f2`, etc.
    - Remove `range-v3::range-v3` link from `src/stencils/CMakeLists.txt` line 15.
  - Files: `src/stencils/polyE2_1.t.cpp`, `src/stencils/E2_2.t.cpp`, `src/stencils/E4_2.t.cpp`, `src/stencils/CMakeLists.txt`.
  - Test: `ctest --test-dir build -L stencils`
  - Ordering: Depends on 7.1c (shared `ccs::linear_distribute` helper).

### Test Migration

- [ ] **7.14** Migrate `src/mesh/mesh.t.cpp` (heavy range-v3):
  - Remove `#include <range/v3/all.hpp>`, add `#include <ranges>`.
  - Line 252: `m.xyz | vs::transform(…)` → `m.xyz | std::views::transform(…)`.
  - Line 254: `rs::equal(a, b)` → `std::ranges::equal(a, b)`.
  - Lines 292–294, 300–302, 314, 316, 320: `rs::count(…)` → `std::ranges::count(…)`.
  - Lines 300–302, 320: `rs::size(…)` → `std::ranges::size(…)`.
  - Lines 308–311: `rs::bidirectional_range<F>`, `rs::contiguous_range<F>`, `rs::random_access_range<F>`, `rs::sized_range<F>` → `std::ranges::` equivalents.
  - Line 324: `vs::transform(…)` → `std::views::transform(…)`.
  - File: `src/mesh/mesh.t.cpp`.
  - Test: `ctest --test-dir build -R t-mesh`

- [ ] **7.15** Migrate `src/mms/mms.t.cpp` (lines 12–13, 52):
  - Remove `#include <range/v3/range/conversion.hpp>` and `#include <range/v3/view/single.hpp>`, add `#include <ranges>`.
  - Line 52: `vs::single(loc) | ms(time) | rs::to<std::vector<real>>()` → replace with explicit evaluation:
    ```cpp
    auto view = std::views::single(loc) | ms(time);
    auto t = std::vector<real>(std::ranges::begin(view), std::ranges::end(view));
    ```
  - File: `src/mms/mms.t.cpp`.
  - Test: `ctest --test-dir build -R t-mms`

- [ ] **7.16** Migrate `src/io/field_io.t.cpp` (lines 6, 68–69):
  - Remove `#include <range/v3/view/iota.hpp>`, add `#include <ranges>`.
  - Lines 68–69: `vs::iota(0, 24)` → `std::views::iota(0, 24)`.
  - File: `src/io/field_io.t.cpp`.
  - Test: `ctest --test-dir build -R t-field_io`

- [ ] **7.17** Migrate `src/io/xdmf.t.cpp` (line 27):
  - Line 27: `rs::size(get<0>(t))` → `std::ranges::size(get<0>(t))` or `get<0>(t).size()`.
  - No range-v3 include to remove (it gets `rs::size` through the `rs` alias in types.hpp which resolves to range-v3).
  - File: `src/io/xdmf.t.cpp`.
  - Test: `ctest --test-dir build -R t-xdmf`

- [ ] **7.18** Clean up test files with stale range-v3 includes (no actual usage):
  - `src/mesh/cartesian.t.cpp` line 6: Remove `#include <range/v3/view/single.hpp>` (unused).
  - `src/simulation/simulation_cycle.t.cpp` line 11: Remove `#include <range/v3/all.hpp>` (unused).
  - Note: `src/io/format_test.cpp` is handled by **7.24a** (deletion of dead code files).
  - Files that have NO range-v3 usage (confirmed clean, no action needed):
    - `src/mesh/object_geometry.t.cpp`, `src/mesh/shapes.t.cpp`
    - `src/io/interval.t.cpp`, `src/io/logging.t.cpp`
    - `src/indexing.t.cpp`, `src/index_view.t.cpp`, `src/real3_operators.t.cpp`
  - Test: build succeeds.

- [ ] **7.19** Migrate `src/mesh/mesh_view.t.cpp` (commented out test, optional):
  - This test is currently **commented out** in `src/mesh/CMakeLists.txt` line 10.
  - If re-enabling: replace `vs::zip(c, g)` (4 uses) with index-based comparison loops; replace `rs::to<vector<real3>>()` calls — but `location_view` already returns `std::vector<real3>` so the `| rs::to<>` is redundant; just use `auto r = location_view<2>(m);`.
  - Remove `#include <range/v3/range/conversion.hpp>` and `#include <range/v3/view/zip.hpp>`.
  - Uncomment CMake line 10 (adjust link targets since `cppcoro` and `range-v3::range-v3` are no longer needed):
    ```cmake
    add_unit_test(mesh_view "mesh" shoccs-mesh)
    ```
  - File: `src/mesh/mesh_view.t.cpp`, `src/mesh/CMakeLists.txt`.
  - Test: `ctest --test-dir build -R t-mesh_view`

### Final Cleanup

- [ ] **7.20** Remove `find_package(range-v3 REQUIRED)` from top-level `CMakeLists.txt` (line 26).
- [ ] **7.21** Remove `find_package(range-v3 REQUIRED)` from `config/shoccsConfig.cmake.in` (line 8).
- [ ] **7.22** Remove `"range-v3@0.12:"` from `.devcontainer/spack.yaml` (line 17).
- [ ] **7.23** Remove `rs`/`vs` namespace aliases and the `namespace ranges::views {}` forward declaration from `src/types.hpp` (lines 15–17, 28–29).
- [ ] **7.24** Sweep: Remove all remaining range-v3 references from the codebase.
  - **7.24a** Delete dead code files with heavy range-v3/cppcoro usage (see decision D10 in `plans/meta.md`):
    - Delete `src/operators/directional.cpp` — commented out of CMake line 21; uses `cppcoro::generator`, older `geometry`/`domain_boundaries` API that no longer exists, 5 range-v3 includes, 6+ API call sites. Not compiled or tested.
    - Delete `src/operators/directional.t.cpp` — same: 7 range-v3 includes, `vs::generate_n`, `vs::filter`, `vs::transform`, `rs::to`, `rs::fill`, `rs::equal`, old `mesh` constructor API.
    - Delete `src/operators/directional.hpp` — header for the dead directional code.
    - Delete `src/io/format_test.cpp` — standalone demo, not in the build. Uses `vs::iota`, `vs::repeat_n`, `range/v3/all.hpp`.
    - Files: `src/operators/directional.cpp`, `src/operators/directional.t.cpp`, `src/operators/directional.hpp`, `src/io/format_test.cpp`.
  - **7.24b** Clean up commented-out CMake references to range-v3 and deleted files:
    - `src/operators/CMakeLists.txt` line 21: Remove commented-out `#add_unit_test(directional ...)` line.
    - `src/geometry/CMakeLists.txt` lines 1–10: All content is commented out and references `range-v3::range-v3` on line 7. Delete this entire file (the geometry code was moved to `src/mesh/` in prior refactoring).
    - Files: `src/operators/CMakeLists.txt`, `src/geometry/CMakeLists.txt`.
  - **7.24c** Remove range-v3 comments from source files:
    - `src/fields/tuple_fwd.hpp` line 262: Remove commented-out `// concept AnyOutputRange = rs::range<T>&& ...` line. Lines 20–21 and 119 mention "range-v3" in descriptive comments — update to say "C++20" or remove the range-v3 reference.
    - `src/mesh/selections.hpp` lines 21–24: Update comment "range-v3 building blocks" to reflect the C++20 rewrite.
    - Files: `src/fields/tuple_fwd.hpp`, `src/mesh/selections.hpp`.
  - Test: build succeeds.
- [ ] **7.25** Full build and test: `cmake --build build && ctest --test-dir build` — all pass.

---

## Ordering Constraints

1. **7.1a** (add `ccs::cartesian_product_view`) must precede **7.1b**, **7.2d**, and any code depending on `ccs::cartesian_product`.
2. **7.1c** (add `ccs::linear_distribute`) must precede **7.1d** (cartesian.cpp migration) and **7.13** (stencil tests).
3. **7.2a–7.2c** (YPlaneView, plane_fn, FView rewrites) are independent of each other.
4. **7.2d** (replace `rs::make_view_closure` in utility functions) should be done after 7.2a–7.2c and 7.1a.
5. **7.3**, **7.4** depend on 7.2 (they include selections.hpp transitively).
6. **7.6** (manufactured_solutions.hpp) should precede **7.15** (mms.t.cpp) since the test pipes through `ms(time)`.
7. **7.7** (mms.cpp CMake cleanup) is independent.
8. **7.12** (scalar_wave.cpp) is independent of mesh/IO items. No CMake changes needed (shoccs-system doesn't link range-v3 directly).
9. **7.13** (stencil tests) depends on **7.1c** (shared `ccs::linear_distribute`).
10. **7.20–7.25** (Final Cleanup) must come last, after all code migration items.

---

## Completion Criteria

- **Zero** references to range-v3 remain anywhere in the codebase.
- `find_package(range-v3)` is removed from CMake.
- All tests pass.
- The `shoccs` executable runs successfully with all 4 Lua config files.

# Phase 2: Matrices Subsystem

**Goal:** Migrate matrix types and their matrix-vector product implementations from range-v3 view pipelines to explicit loops and std::ranges equivalents.

**Depends on:** Phase 0, Phase 1 (for `fields/tuple_fwd.hpp` concepts and `fields/lazy_views.hpp` utilities)

**Read first:**
- `src/matrices/dense.hpp` + `dense.cpp` (heavy: zip_with, inner_product, chunk, repeat_n, stride, zip)
- `src/matrices/circulant.hpp` + `circulant.cpp` (heavy: sliding, zip_with, inner_product, repeat_n, stride, zip)
- `src/matrices/csr.hpp` + `csr.cpp` (moderate: sort, sliding, enumerate, transform)
- `src/matrices/coefficient_visitor.hpp` + `coefficient_visitor.cpp` (moderate: chunk, drop, for_each, zip)
- `src/matrices/unit_stride_visitor.hpp` + `unit_stride_visitor.cpp` (light: rs::size, rs::begin/end)
- `src/matrices/inner_block.hpp` + `inner_block.cpp` (no range-v3)
- `src/matrices/block.hpp` (no range-v3)
- `src/matrices/common.hpp` (no range-v3)
- `src/matrices/matrix_visitor.hpp` (no range-v3)
- `src/matrices/CMakeLists.txt`
- `plans/meta.md` (decision D6)
- `src/fields/lazy_views.hpp` (Phase 1 utilities: `ccs::stride`, `ccs::zip_transform`, `ccs::repeat_n`)

**Test commands:**
```bash
cmake --build build
ctest --test-dir build -L matrices
```

---

## Items

### Resolve Decisions

- [ ] **2.1** Resolve Decision D6: Choose **(b) Write all custom loops (no KokkosKernels dependency).**
  - Update `plans/meta.md` with the decision and rationale.
  - Rationale: Per D1 (host-only) and D4 (keep std::vector), matrices are small per-line operators (not global sparse systems). Replace range-v3 view pipelines in `operator()` with explicit `for` loops using `std::inner_product` for dot products. No Kokkos kernels this phase; they can be added in a future GPU phase.
  - Phase 1 utilities (`ccs::stride`, `ccs::zip_transform`, `ccs::repeat_n` from `fields/lazy_views.hpp`) are available but explicit loops are simpler and clearer for these small matrix operations.

### Core Matrix Types

- [ ] **2.2** Migrate `dense.cpp`: Replace the MatVec `operator()` implementation with explicit loops.
  - Files: `src/matrices/dense.cpp`
  - Remove includes: `<range/v3/algorithm/copy.hpp>`, `<range/v3/numeric/inner_product.hpp>`, `<range/v3/range/concepts.hpp>`, `<range/v3/view/chunk.hpp>`, `<range/v3/view/repeat_n.hpp>`, `<range/v3/view/stride.hpp>`, `<range/v3/view/zip.hpp>`, `<range/v3/view/zip_with.hpp>`
  - Add include: `<numeric>` (for `std::inner_product`)
  - Replace `st == 1` branch: `vs::zip_with(inner_product, chunk(v, cols), repeat_n(x, rows))` + `vs::zip(b, rng)` → explicit row loop:
    ```cpp
    for (integer i = 0; i < rows(); i++) {
        auto dot = std::inner_product(
            v.data() + i * columns(), v.data() + (i + 1) * columns(),
            x.data(), 0.0);
        op(b[i], dot);
    }
    ```
  - Replace `st != 1` branch: same structure but index with stride:
    ```cpp
    for (integer i = 0; i < rows(); i++) {
        real dot = 0.0;
        for (integer j = 0; j < columns(); j++)
            dot += v[i * columns() + j] * x[j * st];
        op(b[i * st], dot);
    }
    ```
  - Test: `ctest --test-dir build -R t-dense`
  - Must come after: 2.1

- [ ] **2.3** Migrate `dense.hpp`: Replace range-v3 concepts and algorithms with std equivalents.
  - Files: `src/matrices/dense.hpp`
  - Remove includes: `<range/v3/algorithm/copy.hpp>`, `<range/v3/range/concepts.hpp>`, `<range/v3/view/take.hpp>`
  - Add includes: `<algorithm>`, `<ranges>`
  - Replace `rs::input_range` → `std::ranges::input_range` (2 constructor templates, lines 23 and 30)
  - Replace `rs::copy(rng | vs::take(v.size()), v.begin())` → `std::ranges::copy(rng | std::views::take(v.size()), v.begin())` (2 occurrences, lines 27 and 42)
  - Note: Can be done in the same work pass as 2.2
  - Test: `ctest --test-dir build -R t-dense`

- [ ] **2.4** Migrate `circulant.cpp`: Replace the convolution `operator()` with explicit loops.
  - Files: `src/matrices/circulant.cpp`
  - Remove includes: all 9 range-v3 includes (`copy`, `inner_product`, `concepts`, `drop`, `repeat_n`, `sliding`, `stride`, `zip`, `zip_with`)
  - Add include: `<numeric>` (for `std::inner_product`)
  - `circulant.hpp` has no range-v3 usage — no changes needed there.
  - Replace `st == 1` branch: `vs::zip_with(inner_product, repeat_n(v, rows), sliding(x, size))` + `vs::zip(b, rng)` → explicit row loop:
    ```cpp
    for (integer i = 0; i < rows(); i++) {
        auto dot = std::inner_product(v.begin(), v.end(), x.data() + i, 0.0);
        op(b[i], dot);
    }
    ```
  - Replace `st != 1` branch: nested loop with stride:
    ```cpp
    for (integer i = 0; i < rows(); i++) {
        real dot = 0.0;
        for (integer j = 0; j < size(); j++)
            dot += v[j] * x[(i + j) * st];
        op(b[i * st], dot);
    }
    ```
  - Test: `ctest --test-dir build -R t-circulant`
  - Must come after: 2.1

- [ ] **2.5** Migrate `csr.cpp`: Replace sort, sliding+enumerate, and transform with std/loops.
  - Files: `src/matrices/csr.cpp`
  - Remove includes: `<range/v3/algorithm/sort.hpp>`, `<range/v3/view/enumerate.hpp>`, `<range/v3/view/sliding.hpp>`, `<range/v3/view/transform.hpp>`
  - Add include: `<algorithm>` (for `std::ranges::sort`)
  - Replace `rs::sort(p)` → `std::ranges::sort(p)`
  - Replace `u | vs::sliding(2) | vs::enumerate` loop → explicit index loop:
    ```cpp
    for (integer i = 0; i < nrows; i++) {
        u[i + 1] = u[i];
        while (first != last && first->row == i) {
            ++u[i + 1];
            ++first;
        }
    }
    ```
  - Replace `p | vs::transform(...)` in `to_csr()` return → build `std::vector<real>` and `std::vector<integer>` with explicit loops over `p`, then pass to `csr` constructor:
    ```cpp
    std::vector<real> w_vec;
    std::vector<integer> v_vec;
    w_vec.reserve(p.size());
    v_vec.reserve(p.size());
    for (auto& pt : p) { w_vec.push_back(pt.v); v_vec.push_back(pt.col); }
    return csr{w_vec, v_vec, u};
    ```
  - Test: `ctest --test-dir build -R t-csr`
  - Must come after: 2.6 (csr constructor must accept std ranges first)

- [ ] **2.6** Migrate `csr.hpp`: Replace range-v3 concepts and iterator accessors.
  - Files: `src/matrices/csr.hpp`
  - Remove include: `<range/v3/range/concepts.hpp>`
  - Add include: `<ranges>`
  - Replace `ranges::input_range` → `std::ranges::input_range` (template constraint on constructor, line 23)
  - Replace `rs::begin(w)`, `rs::end(w)`, `rs::begin(v)`, `rs::end(v)`, `rs::begin(u)`, `rs::end(u)` → `std::ranges::begin(...)`, `std::ranges::end(...)` (6 occurrences in constructor initializer list, lines 25-27)
  - Test: `ctest --test-dir build -R t-csr`

### Visitors

- [ ] **2.7** Migrate `coefficient_visitor.cpp`: Replace chunk+for_each+drop/take and zip patterns with explicit loops.
  - Files: `src/matrices/coefficient_visitor.cpp` (`.hpp` has no range-v3 usage)
  - Remove includes: `<range/v3/view/chunk.hpp>`, `<range/v3/view/drop.hpp>`, `<range/v3/view/for_each.hpp>`, `<range/v3/view/zip.hpp>`
  - `visit(const dense&)` — LDD case: Replace `vs::chunk(c_n) | vs::for_each(vs::drop(1))` + `vs::zip(ind|t, d|t)` → nested loop skipping first column of each row:
    ```cpp
    for (integer r = 0; r < r_n; r++)
        for (integer c = 1; c < c_n; c++)
            m[ind[r * c_n + c]] = d[r * c_n + c];
    ```
  - `visit(const dense&)` — RDD case: Replace `vs::chunk(c_n) | vs::for_each(vs::take(c_n-1))` + `vs::zip` → nested loop skipping last column:
    ```cpp
    for (integer r = 0; r < r_n; r++)
        for (integer c = 0; c < c_n - 1; c++)
            m[ind[r * c_n + c]] = d[r * c_n + c];
    ```
  - `visit(const dense&)` — default case: Replace `vs::zip(ind, d)` → simple index loop:
    ```cpp
    for (integer i = 0; i < (integer)ind.size(); i++)
        m[ind[i]] = d[i];
    ```
  - `visit(const circulant&)`: Replace `vs::zip(mapped_span, mat.data())` → index loop over both spans.
  - `visit(const csr&)`: Replace `vs::zip(mapped_span, column_coefficients)` → index loop over both spans.
  - Test: `ctest --test-dir build -R t-coefficient_visitor`

- [ ] **2.8** Migrate `unit_stride_visitor.hpp` and `unit_stride_visitor.cpp`: Replace `rs::size` and `rs::begin`/`rs::end` with std equivalents.
  - Files: `src/matrices/unit_stride_visitor.hpp`, `src/matrices/unit_stride_visitor.cpp`
  - `.hpp`: Add include: `<ranges>` (if not already transitively included)
  - `.hpp`: Replace `rs::size(rx)`, `rs::size(ry)`, `rs::size(rz)` → `std::ranges::size(rx)` etc. (3 occurrences in constructor, lines 49-50)
  - `.hpp`: Replace `rs::begin(rx)`, `rs::end(rx)`, `rs::begin(ry)`, `rs::end(ry)`, `rs::begin(rz)`, `rs::end(rz)` → `std::ranges::begin(...)`, `std::ranges::end(...)` (6 occurrences, lines 57-59)
  - `.cpp`: Replace `rs::size(row_skip)`, `rs::size(col_skip)` → `std::ranges::size(...)` or `.size()` (2 occurrences, lines 122-123)
  - Test: `ctest --test-dir build -R t-unit_stride_visitor`

### Test Migration

Common range-v3 → std/C++20 replacement patterns used across test files:
- `rs::to<T>()` → `T(std::ranges::begin(view), std::ranges::end(view))` (works for common ranges)
- `vs::iota(a, b)` → `std::views::iota(a, b)`
- `vs::transform(f)` → `std::views::transform(f)`
- `vs::take(n)` → `std::views::take(n)`
- `vs::drop(n)` → `std::views::drop(n)`
- `vs::stride(n)` → `ccs::stride(rng, n)` (free function from `fields/lazy_views.hpp`; not pipeable)
- `vs::generate_n(f, n) | rs::to<T>()` → `T v(n); std::generate_n(v.begin(), n, f);`
- `rs::equal(a, b)` → `std::ranges::equal(a, b)`
- `ranges::shuffle(pts)` → `std::ranges::shuffle(pts, urbg)` (requires explicit URBG engine)
- `vs::concat(a, b, ...)` → eager `std::vector` concatenation via `insert`/`push_back`
- `vs::single(v)` / `vs::repeat_n(v, n)` → `std::vector{v}` / `std::vector(n, v)` (eager, fine for tests)
- `vs::repeat(v)` → `std::vector<real>(size, v)` of appropriate size (used in visitor tests for dummy data)

- [ ] **2.9a** Migrate visitor test files: `unit_stride_visitor.t.cpp` + `coefficient_visitor.t.cpp`.
  - Files: `src/matrices/unit_stride_visitor.t.cpp`, `src/matrices/coefficient_visitor.t.cpp`
  - `unit_stride_visitor.t.cpp`:
    - Remove `<range/v3/all.hpp>`. Add `<algorithm>`, `<ranges>`, `<vector>`.
    - Replace `vs::repeat(0.0)` (infinite range used in dense constructors) → `std::vector<real>(rows * cols, 0.0)` of appropriate size per call. The dense constructor only reads `rows * columns` elements.
    - Replace `rs::equal(a, b)` → `std::ranges::equal(a, b)` (5 occurrences).
  - `coefficient_visitor.t.cpp`:
    - Remove `<range/v3/all.hpp>`. Add `<algorithm>`, `<ranges>`, `<numeric>`.
    - Replace `vs::iota(25, 50) | rs::to<T>()` → `T` from `std::views::iota`: e.g. `auto r = std::views::iota(25, 50); T imat(std::ranges::begin(r), std::ranges::end(r));`
    - Replace `rs::equal(a, b)` → `std::ranges::equal(a, b)` (3 occurrences).
  - Test: `ctest --test-dir build -R "t-unit_stride_visitor|t-coefficient_visitor"`
  - Must come after: 2.7, 2.8

- [ ] **2.9b** Migrate dense + circulant test files: `dense.t.cpp` + `circulant.t.cpp`.
  - Files: `src/matrices/dense.t.cpp`, `src/matrices/circulant.t.cpp`
  - Both files use similar patterns: `iota`, `transform`, `drop`, `stride`, `take`, `generate_n`, `to<T>`.
  - Remove all `<range/v3/...>` includes. Add `<algorithm>`, `<ranges>`, `<numeric>`, `"fields/lazy_views.hpp"` (for `ccs::stride`).
  - Key transformations:
    - `vs::iota(0, 25)` → `std::views::iota(0, 25)`
    - `rng | vs::transform(f) | rs::to<T>()` → use iterator-pair constructor or loop
    - `vs::generate_n(g, n) | rs::to<T>()` → `T v(n); std::generate_n(v.begin(), n, g);`
    - `x | vs::drop(offset) | vs::stride(3) | vs::take(n) | rs::to<T>()` → `ccs::stride(x | std::views::drop(offset), 3) | std::views::take(n)` then collect to vector
  - Test: `ctest --test-dir build -R "t-dense|t-circulant"`
  - Must come after: 2.2, 2.3, 2.4

- [ ] **2.9c** Migrate CSR test file: `csr.t.cpp`.
  - Files: `src/matrices/csr.t.cpp`
  - Remove all `<range/v3/...>` includes. Add `<algorithm>`, `<random>`.
  - Replace `ranges::shuffle(pts)` → `std::ranges::shuffle(pts, urbg)` with a local `std::mt19937` engine (seeded from `std::random_device` or fixed seed for reproducibility).
  - Replace `vs::generate_n(g, n) | rs::to<T>()` → manual vector + `std::generate_n`.
  - Test: `ctest --test-dir build -R t-csr`
  - Must come after: 2.5, 2.6

- [ ] **2.9d** Migrate composite matrix test files: `inner_block.t.cpp` + `block.t.cpp`.
  - Files: `src/matrices/inner_block.t.cpp`, `src/matrices/block.t.cpp`
  - `inner_block.t.cpp`: Same patterns as 2.9b (iota, transform, drop, stride, take, generate_n, to<T>).
  - `block.t.cpp` (most complex test file):
    - Replace `vs::concat(vs::single(0.0), rhs_, vs::repeat_n(0.0, 4), rhs_) | rs::to<T>()` → eager vector concatenation:
      ```cpp
      T x{0.0};
      x.insert(x.end(), rhs_.begin(), rhs_.end());
      x.insert(x.end(), 4, 0.0);
      x.insert(x.end(), rhs_.begin(), rhs_.end());
      ```
    - Replace `rs::equal` → `std::ranges::equal`.
    - Replace `vs::iota | rs::to<T>()`, `vs::generate_n | rs::to<T>()`, `vs::transform | rs::to<T>()` etc.
  - Remove all `<range/v3/...>` includes. Add `<algorithm>`, `<ranges>`, `<numeric>`, `"fields/lazy_views.hpp"`.
  - Test: `ctest --test-dir build -R "t-inner_block|t-block"`
  - Must come after: 2.2, 2.3, 2.4

### Verification

- [ ] **2.10** Final verification: no range-v3 remains in `src/matrices/`.
  - Verify: `grep -r "range/v3" src/matrices/` returns no results.
  - No CMakeLists.txt changes expected (`shoccs-matrices` links `fields` which transitively provides range-v3; this transitive dependency will be removed when fields migration completes).
  - Test: `cmake --build build && ctest --test-dir build -L matrices`

---

## Ordering Summary

```
2.1 (D6 decision)
 ├── 2.3 (dense.hpp)  ──┐
 ├── 2.2 (dense.cpp)  ──┤── 2.9b (dense + circulant tests) ──┐
 ├── 2.4 (circulant.cpp)┤── 2.9d (inner_block + block tests) ├── 2.10 (verification)
 ├── 2.6 (csr.hpp) ── 2.5 (csr.cpp) ── 2.9c (csr tests) ────┤
 ├── 2.7 (coefficient_visitor.cpp) ──┐── 2.9a (visitor tests) ┘
 └── 2.8 (unit_stride_visitor.hpp/.cpp)┘
```

Items 2.2–2.8 have no inter-dependencies (except 2.5 depends on 2.6) and can be done in parallel after 2.1.

---

## Completion Criteria

- All 7 matrix test files pass.
- No `#include <range/v3/...>` remains in `src/matrices/`.
- Decision D6 is recorded in `meta.md`.
- `shoccs-matrices` library no longer links range-v3 directly (may still get it transitively until fields is done).

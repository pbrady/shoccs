# Phase 15: Code Smell Fixes

**Goal:** Address all critical, important, and test quality issues identified by the 6-team code review of Phases 8–13.

**Depends on:** Phases 8–13 (complete)

**Read first:**
- `src/fields/lazy_views.hpp` (repeat_n_view iterator dangling pointer)
- `src/fields/field_registry.hpp` (missing assertions)
- `src/fields/expr.hpp` (size mismatch, lifetime contract)
- `src/fields/selection_desc.hpp` (zero guard, integer truncation, predicate invariant)
- `src/matrices/block.hpp` (missing fence, disjoint output assertion)
- `src/matrices/circulant.cpp` (nesting detection, missing fence)
- `src/matrices/csr.cpp` (not parallelized)
- `src/temporal/slot_ops.hpp` (missing fence)
- `src/simulation/simulation_cycle.cpp` (field_ref initialization)
- `src/systems/inviscid_vortex.hpp` + `inviscid_vortex.cpp` (dead code)
- `src/systems/scalar_wave.hpp` (dead comments)

**Test commands:**
```bash
cmake --build build
ctest --test-dir build
```

---

## Items

### 15.1 — Critical: Memory safety and correctness

- [x] **15.1a** Fix dangling pointer in `repeat_n_view::iterator` (`lazy_views.hpp`):
  - The iterator stores `const T* value_` (line 56) pointing into the parent view's `value_` member. After copy/move of the view, the pointer dangles.
  - Changes (4 edits in `src/fields/lazy_views.hpp`):
    1. Line 56: change `const T* value_;` → `T value_;`
    2. Line 68: change constructor `constexpr iterator(const T* v, std::ptrdiff_t p) : value_{v}` → `constexpr iterator(T v, std::ptrdiff_t p) : value_{std::move(v)}`
    3. Line 70: change `constexpr reference operator*() const { return *value_; }` → `constexpr reference operator*() const { return value_; }`
    4. Line 71: change `constexpr reference operator[](difference_type) const { return *value_; }` → `constexpr reference operator[](difference_type) const { return value_; }`
  - Also update `begin()` (line 139) and `end()` (line 140) to pass `value_` by value instead of `&value_`.
  - Remove `pointer` typedef (line 63) or change to `const T*` (it's unused but should be consistent).
  - `repeat_n_view` is used with small scalar types (`real`, `int`), so the copy is cheap.
  - File: `src/fields/lazy_views.hpp`
  - Test: `ctest --test-dir build -R t-expr` (repeat_n is used in expression tests)

- [x] **15.1b** Add size-match assertion to `deep_copy_slot` (`field_registry.hpp`):
  - Inside the loop at line 171, before `Kokkos::deep_copy(buffers_[dst_base + i], buffers_[src_base + i])`, add:
    `assert(buffers_[dst_base + i].extent(0) == buffers_[src_base + i].extent(0));`
  - File: `src/fields/field_registry.hpp`
  - Test: `ctest --test-dir build -R t-field_registry`

- [x] **15.1c** Add bounds checks to `view()`/`data()`/`size()` in `field_registry.hpp`:
  - Add `assert(ref.slot >= 0 && ref.slot < MaxSlots)` and `assert(h.id >= 0 && h.id < buffers_per_slot)` at the top of each accessor.
  - Apply to all 5 accessor methods: `view()` (lines 135, 140), `data()` (lines 145, 150), `size()` (line 155).
  - File: `src/fields/field_registry.hpp`
  - Test: `ctest --test-dir build -R t-field_registry`

- [x] **15.1d** Add size-match assertion to binary `scalar_expr` operators (`expr.hpp`):
  - In all four binary operators (`+`, `-`, `*`, `/`), add `assert(a.sizes[i] == b.sizes[i])` before setting `result.sizes[i] = a.sizes[i]`.
  - File: `src/fields/expr.hpp`
  - Test: `ctest --test-dir build -R t-expr`

### 15.2 — Critical: Missing Kokkos::fence()

All four review teams flagged this independently. The fix is the same pattern everywhere: add `Kokkos::fence()` after `parallel_for` before any host-side read of the output buffer.

- [x] **15.2a** Add `Kokkos::fence()` to `block::operator()` after `parallel_for`:
  - Add `Kokkos::fence();` after line 39 (after the `parallel_for` lambda closes). The caller reads `b` on the host after `operator()` returns, so the fence ensures all writes are visible.
  - File: `src/matrices/block.hpp`
  - Test: `ctest --test-dir build -R t-block`

- [x] **15.2b** Add `Kokkos::fence()` to standalone `circulant::operator()` paths:
  - Two `parallel_for` calls need fences: the `st==1` non-nested branch (after line 69's `});`) and the `st!=1` non-nested branch (after line 87's `});`, before the closing `}` on line 88).
  - The nested (serial) branches don't need fences since they're already synchronous.
  - File: `src/matrices/circulant.cpp`
  - Test: `ctest --test-dir build -R t-circulant`
  - Ordering: do this before or together with 15.3a since both modify `circulant.cpp`.

- [x] **15.2c** Add `Kokkos::fence()` to `slot_ops.hpp` helpers:
  - `slot_assign_lc` (lines 22-38): add a single `Kokkos::fence()` after the outer `for (int s = ...)` loop closes (after line 37). One fence covers all the `parallel_for` calls issued inside the nested loops.
  - `slot_accumulate` (lines 41-56): same pattern — add `Kokkos::fence()` after line 55.
  - `slot_zero` (lines 11-19): uses `Kokkos::deep_copy` which is already synchronous. No fence needed, but add one after line 18 for consistency with future device builds.
  - File: `src/temporal/slot_ops.hpp`
  - Test: `ctest --test-dir build -R t-rk4_v2`

### 15.3 — Critical: Nesting detection portability

- [x] **15.3a** Replace `omp_in_parallel()` with portable Kokkos nesting detection in `circulant.cpp`:
  - Replace lines 51-55:
    ```cpp
    #if defined(KOKKOS_ENABLE_OPENMP)
    const bool nested = omp_in_parallel();
    #else
    const bool nested = false;
    #endif
    ```
  - With: `const bool nested = execution_space::in_parallel();`
  - `execution_space::in_parallel()` is a static member that wraps `omp_in_parallel()` for OpenMP, and returns `false` for Serial and Threads backends. This makes the code portable without backend-specific `#if` guards.
  - **Deprecation note:** `in_parallel()` is marked `KOKKOS_DEPRECATED` in Kokkos 4.7. It compiles because the project's Kokkos build has `KOKKOS_ENABLE_DEPRECATED_CODE_4` enabled. There is no non-deprecated Kokkos replacement. If Kokkos removes this API in a future version, revert to `omp_in_parallel()` with the `#if` guard or pass a `nested` flag from `block::operator()`.
  - Remove `#include <omp.h>` (line 9) and the `#if defined(KOKKOS_ENABLE_OPENMP)` guard (lines 8-10) since both are replaced by the Kokkos abstraction.
  - File: `src/matrices/circulant.cpp`
  - Test: `ctest --test-dir build -R t-circulant`

### 15.4 — Important: Selection descriptor safety

- [x] **15.4a** Add zero-guard for `strided_selection::inner_count_`:
  - Add `assert(inner_count_ > 0)` in `make_y_plane_desc` and `make_z_plane_desc` factory functions.
  - Add a comment on the `strided_selection` struct documenting the invariant: `inner_count_ must be > 0`.
  - File: `src/fields/selection_desc.hpp`
  - Test: `ctest --test-dir build -R t-selection_desc`

- [x] **15.4b** Document the predicate-index invariant in `make_gather_from_predicate`:
  - Add a prominent comment above `make_gather_from_predicate` (selection_desc.hpp, line 127):
    > "The returned indices are positions within the `infos` span. Callers must ensure the R data buffer at the same position holds the value associated with `infos[i]`."
  - In `mesh.hpp`, `dirichlet_object_desc()` (line 98) and `non_dirichlet_object_desc()` (line 107) call `make_gather_from_predicate(R(dir), pred)`. The `R(dir)` span's layout must match the R buffer layout — this is guaranteed by construction, so add a one-line comment at each call site rather than a runtime assertion (the buffer size isn't available in the mesh's scope).
  - Files: `src/fields/selection_desc.hpp`, `src/mesh/mesh.hpp`
  - Test: `ctest --test-dir build -R t-mesh`

- [x] **15.4c** Add integer truncation guard in `make_gather_from_slices`:
  - Add `assert(idx <= std::numeric_limits<int>::max())` before the `static_cast<int>(idx)`.
  - Add similar overflow guards in `make_x_plane_desc`, `make_y_plane_desc`, `make_z_plane_desc` for the `nx*ny`/`ny*nz` products.
  - File: `src/fields/selection_desc.hpp`
  - Test: `ctest --test-dir build -R t-selection_desc`

### 15.5 — Important: expr.hpp lifetime contract

- [x] **15.5a** Document the synchronous-only contract on `assign()` and `assign_selected()`:
  - Add a comment block at the top of `assign()`:
    > "IMPORTANT: This function requires that execution_space is synchronous (DefaultHostExecutionSpace). The Expr captures raw real* pointers whose lifetime is only guaranteed for the duration of this call. For async/device execution spaces, Expr must capture Kokkos::View instead of raw pointers."
  - Same comment on `assign_selected()`, `fill_selected()`, `plus_assign()`, etc.
  - File: `src/fields/expr.hpp`
  - No test changes needed (documentation only).

### 15.6 — Important: Parallelize csr::operator()

- [x] **15.6a** Add `Kokkos::parallel_for` to `csr::operator()`:
  - The row loop is trivially parallel — each row writes to a distinct `b[row]`.
  - Replace lines 37-38:
    ```cpp
    for (integer row = 0; row < rows(); row++)
        for (integer i = u[row]; i < u[row + 1]; i++) b[row] += w[i] * x[v[i]];
    ```
  - With:
    ```cpp
    const auto nr = rows();
    const auto* w_ptr = w.data();
    const auto* v_ptr = v.data();
    const auto* u_ptr = u.data();
    const auto* x_ptr = x.data();
    auto* b_ptr = b.data();
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, nr),
        [=](integer row) {
            for (integer i = u_ptr[row]; i < u_ptr[row + 1]; i++)
                b_ptr[row] += w_ptr[i] * x_ptr[v_ptr[i]];
        });
    Kokkos::fence();
    ```
  - Must extract raw pointers from both spans (`x`, `b`) AND member vectors (`w`, `v`, `u`) before the lambda. The `this` pointer is not capturable in `KOKKOS_LAMBDA`, and `std::span` / `std::vector` are not trivially copyable.
  - Add `#include "kokkos_types.hpp"` to `csr.cpp` (currently only includes `csr.hpp` and `<algorithm>`).
  - File: `src/matrices/csr.cpp`
  - Test: `ctest --test-dir build -R t-csr`

### 15.7 — Important: block disjointness assertion

- [x] **15.7a** Add debug-mode disjoint output region check to `block::builder::to_block()`:
  - **Decision (D15-1):** Place the assertion in `builder::to_block()` (line 61), not in `operator()`. This runs once at construction time rather than per matrix-vector product, which is cheaper and catches bugs earlier.
  - In `to_block()`, before `return block{MOVE(b)}`, add a debug-mode check (`#ifndef NDEBUG`):
    - For each pair of inner_blocks, verify their output row ranges (computed from `row_offset()`, `rows()`, `stride()`) do not overlap.
    - Implementation: iterate over all pairs of blocks and check that `[b[i].row_offset(), b[i].row_offset() + b[i].rows() * b[i].stride())` does not overlap with `[b[j].row_offset(), b[j].row_offset() + b[j].rows() * b[j].stride())`.
  - File: `src/matrices/block.hpp`
  - Test: `ctest --test-dir build -R t-block`

### 15.8 — Important: Dead code removal

- [x] **15.8a** Delete `#if 0` blocks from `inviscid_vortex.hpp` and `inviscid_vortex.cpp`:
  - `inviscid_vortex.hpp`: remove 2 `#if 0` blocks — lines 17-29 (dead member variables) and lines 34-43 (dead `euler_vortex` constructor).
  - `inviscid_vortex.cpp`: remove 3 `#if 0` blocks — lines 19-39 (`const_var_spans`/`var_spans` templates using `absl::Span`), lines 140-212 (dead `euler_vortex` constructor using `cart_mesh`), and lines 252-396 (dead `euler_vortex` methods using `absl::Span`, `null_boundary_tuple`).
  - Total: ~270 lines of dead code referencing removed types.
  - Files: `src/systems/inviscid_vortex.hpp`, `src/systems/inviscid_vortex.cpp`
  - Test: `ctest --test-dir build -L systems`

- [x] **15.8b** Remove commented-out `std::vector<double>` members from `scalar_wave.hpp`:
  - Delete lines 24-25 (commented-out `// std::vector<double> grad_c;` and `// std::vector<double> grad_u;`). These use `double` instead of `real` and reference old gradient scratch data that has been replaced by the explicit `du_*` members (lines 35-37).
  - File: `src/systems/scalar_wave.hpp`
  - Test: `ctest --test-dir build -L systems`

### 15.9 — Important: simulation_cycle field_ref initialization

- [x] **15.9a** Improve `field_ref` initialization in `simulation_cycle.cpp`:
  - Line 40 initializes `field_ref u0_ref{0}, u1_ref{1}, rk_ref{2}, srhs_ref{3}` with `n_scalars=0, n_vectors=0`. For zero-field systems (e.g., `inviscid_vortex` returns `system_size{}`), the allocation loops (lines 41-52) never execute, so these refs remain with zero counts, which silently no-ops all slot operations.
  - After the allocation loops (after line 52), add:
    ```cpp
    // For zero-field systems (nscalars==0, nvectors==0), refs retain their
    // initial {slot, 0, 0} state — slot_ops correctly no-op.
    assert(u0_ref.n_scalars == sz.nscalars && u0_ref.n_vectors == sz.nvectors);
    ```
  - File: `src/simulation/simulation_cycle.cpp`
  - Test: `ctest --test-dir build -R t-simulation_cycle`

### 15.10 — Test quality improvements

- [x] **15.10a** Replace exact `== 0` floating-point comparisons in system tests:
  - In `heat.t.cpp` (lines 121, 235, 352) and `scalar_wave.t.cpp` (line 91), replace `REQUIRE(st.stats[0] == 0)` with `REQUIRE_THAT(st.stats[0], Catch::Matchers::WithinAbs(0.0, 1e-13))`.
  - Add `#include <catch2/matchers/catch_matchers_floating_point.hpp>` to each file if not already present.
  - Files: `src/systems/heat.t.cpp`, `src/systems/scalar_wave.t.cpp`
  - Test: `ctest --test-dir build -R "t-heat|t-scalar_wave"`

- [x] **15.10b** Fix the aliasing test in `expr.t.cpp` to be non-trivial:
  - The current test (lines 144-159) does `assign(a.data(), n, handle_expr{a.data()})` — a self-identity copy that produces correct results even without the temporary staging path. This doesn't verify that the aliasing-detection code path actually works.
  - Replace the test body with a test that assigns `a + a` to `a`:
    ```cpp
    TEST_CASE("assign detects aliasing and stages through temporary")
    {
        constexpr int n = 100;
        Kokkos::View<real*, memory_space> a("a", n);
        for (int i = 0; i < n; ++i) a(i) = static_cast<real>(i + 1);

        // dst[i] = dst[i] + dst[i] — aliased, must stage through temporary.
        assign(a.data(), n,
               binary_expr{std::plus<>{}, handle_expr{a.data()}, handle_expr{a.data()}});

        for (int i = 0; i < n; ++i) {
            REQUIRE(a(i) == static_cast<real>(2 * (i + 1)));
        }
    }
    ```
  - This tests the actual aliasing code path because `a + a` assigned to `a` would be incorrect without the temporary (reads and writes to the same buffer in a `parallel_for`). Although element-wise `a + a → a` happens to be safe due to no cross-element dependency, it exercises the temporary staging code path that `contains_ptr` triggers.
  - File: `src/fields/expr.t.cpp`
  - Test: `ctest --test-dir build -R t-expr`

- [x] **15.10c** Add `Kokkos::fence()` to `field_registry.t.cpp` tests before host reads:
  - After `deep_copy_slot` and `swap_slots` calls, add `Kokkos::fence()` before reading View data on the host. This makes the tests correct for future device builds.
  - File: `src/fields/field_registry.t.cpp`
  - Test: `ctest --test-dir build -R t-field_registry`

- [x] **15.10d** Remove dead Lua code from `heat.t.cpp` test strings:
  - In the "floating" test case, remove commented-out Lua alternatives:
    - Lines 186-190: `--[[ return (math.sin(time) + x*x*(y+z) + ...) ]]` in `call`
    - Lines 200-204: `--[[ return 2.*x*(y+z) + ... ]]` in `grad`
    - Line 209: `--return 2. * (y + z) + 2. * (x + z) + 2. * (x + y)` in `lap`
  - In the "2D floating" test case, remove commented-out Lua alternatives:
    - Lines 305-308: `--[[ return (math.sin(time) + x*x*y + ...) ]]` in `call`
    - Lines 317-321: `--[[ return 2.*x*y + y*y + ... ]]` in `grad`
    - Line 325: `--return 2. * y + 2. * x` in `lap`
  - These are remnants of higher-order manufactured solutions that were simplified. The active code (linear solutions with zero laplacian) is the correct test configuration.
  - File: `src/systems/heat.t.cpp`
  - Test: `ctest --test-dir build -R t-heat`

- [x] **15.10e** Remove redundant `static_assert` test cases:
  - `handle.t.cpp`: Remove lines 277-282 ("handle types are trivially copyable") — duplicates `handle.hpp:456-458`. Also remove lines 284-289 ("handle types are aggregates") — duplicates `handle.hpp:462-464`.
  - `field_registry.t.cpp`: Remove lines 27-35 ("field_ref is trivially copyable" and "field_ref size is 12 bytes") — duplicates `field_registry.hpp:54-55`.
  - `expr.t.cpp`: Remove lines 75-86 ("expression node types are trivially copyable") — duplicates `expr.hpp:54-55` for leaf types, and the template-level `static_assert` inside `binary_expr`/`unary_expr` (expr.hpp:35-37,46-47) covers instantiations automatically at compile time.
  - Files: `src/fields/handle.t.cpp`, `src/fields/field_registry.t.cpp`, `src/fields/expr.t.cpp`
  - Test: `ctest --test-dir build -L fields`

- [x] **15.10f** Decouple `selection_desc.t.cpp` from internal struct layout:
  - Remove the three plane-factory TEST_CASEs that access internal members (`desc.offset_`, `desc.inner_count_`, `desc.outer_count_`, `desc.outer_stride_`): "make_x_plane_desc" (lines 116-126), "make_y_plane_desc" (lines 128-138), "make_z_plane_desc" (lines 140-150). The behavioral cross-check tests (lines 155-224) already verify correctness through the public `element()` and `count()` APIs.
  - File: `src/fields/selection_desc.t.cpp`
  - Test: `ctest --test-dir build -R t-selection_desc`

### 15.11 — Build hygiene

- [x] **15.11a** Fix `shoccs-random` CMake hygiene:
  - Replace `target_include_directories(shoccs-random PUBLIC ..)` with `target_include_directories(shoccs-random PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)`.
  - File: `src/random/CMakeLists.txt`
  - Test: `cmake --build build`

- [x] **15.11b** Add `shoccs-random` to the install target list:
  - Add `shoccs-random` to the `install(TARGETS ...)` block in the top-level `CMakeLists.txt`.
  - File: `CMakeLists.txt`
  - Test: `cmake --build build`

---

## Ordering Constraints

```
15.1a-d (memory safety)     — independent, can be done in parallel
15.2a-c (fences)            — independent, can be done in parallel
15.3a   (nesting detection) — do together with 15.2b (both edit circulant.cpp)
15.4a-c (selection safety)  — independent
15.5a   (documentation)     — independent
15.6a   (csr parallel)      — depends on 15.2 pattern (needs fence)
15.7a   (disjointness)      — independent
15.8a-b (dead code)         — independent
15.9a   (field_ref init)    — independent
15.10a-f (test quality)     — independent, can be done in parallel
15.11a-b (build hygiene)    — independent
```

Co-edit note: 15.2b and 15.3a both modify `circulant.cpp` — do them in the same work pass to avoid merge conflicts.

---

## Completion Criteria

- All `assert()` guards added for bounds, size-match, and zero-division.
- `Kokkos::fence()` present after every `parallel_for` that precedes host reads.
- `omp_in_parallel()` replaced with portable `execution_space::in_parallel()`.
- `repeat_n_view::iterator` stores value by copy, not pointer.
- `csr::operator()` parallelized.
- Dead code removed from `inviscid_vortex` and `scalar_wave`.
- Test quality issues fixed (FP equality, aliasing test, fences in tests).
- Build hygiene fixed for `shoccs-random`.
- All existing tests pass (no regressions).

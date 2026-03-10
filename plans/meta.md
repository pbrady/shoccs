# SHOCCS Kokkos Migration — Meta Plan

**Purpose:** Cross-cutting decisions and phase ordering for the range-v3 to Kokkos migration.

---

## Phase Ordering

Plans must be executed in this order due to dependency chains:

| Phase | Plan File | Depends On |
|-------|-----------|------------|
| 0 | `00-foundation.md` | Nothing |
| 1 | `01-fields.md` | Phase 0 |
| 2 | `02-matrices.md` | Phases 0, 1 |
| 3 | `03-stencils.md` | Phase 0 |
| 4 | `04-operators.md` | Phases 0–3 |
| 5 | `05-systems.md` | Phases 0–4 |
| 6 | `06-temporal.md` | Phases 0, 1 |
| 7 | `07-simulation-io-mms.md` | Phases 0–6 |

---

## Decision Log

Record cross-cutting architectural decisions here. Each decision should be referenced by plan items that depend on it.

### D1: Host vs. Device Execution Strategy
**Decision:** **(a) Start with `Kokkos::DefaultHostExecutionSpace` everywhere, port to GPU later.**
The codebase is entirely CPU-only today. Starting host-only minimizes risk and allows validating correctness before adding device portability. GPU migration can be a separate future effort.
**Options:**
- **(a) Start with `Kokkos::DefaultHostExecutionSpace` everywhere, port to GPU later** ← CHOSEN
- (b) Use `Kokkos::DualView` from the start
- (c) Use `Kokkos::View` with explicit `Kokkos::DefaultExecutionSpace` and mirror views for I/O
**Considerations:** The code is CPU-only today. Starting host-only minimizes risk and lets us validate correctness before adding device portability.

### D2: range-v3 Removal Strategy
**Decision:** **(b) Keep `range-v3` as a project dependency; remove usage incrementally phase by phase.**
93 source files use range-v3 (1590 total occurrences of `rs::`/`vs::`). Many uses (`vs::cartesian_product`, `vs::take_exactly`, `vs::stride`, `rs::inner_product`, `rs::to`) have no direct C++20 `std::ranges` equivalents. Each subsequent phase will remove range-v3 from its subsystem. The top-level `find_package(range-v3)` and the `rs`/`vs` namespace aliases in `types.hpp` remain until all phases complete. Phase 0 only removes range-v3 from `index_view.t.cpp`.
**Options:**
- (a) Remove `range-v3` dependency entirely, replace all uses with Kokkos + std
- **(b) Keep `range-v3` for host-side-only convenience code (tests, init), remove from hot paths** ← CHOSEN
- (c) Replace with C++20/23 `std::ranges` where possible, Kokkos for parallel ops
**Considerations:** Option (c) is pragmatic — `std::ranges` covers many simple view compositions (transform, filter, take, drop). Kokkos is needed for parallel execution patterns. Tests can continue using range-v3 or std::ranges.

### D3: cppcoro Generator Replacement
**Decision:** **(b) Replace with plain functions returning `std::vector`.**
Generators are only used in `index_view.hpp` (2 overloads yielding `int3`) and `mesh_view.hpp` (2 overloads yielding `real3`). Both are simple triple-nested loops. Callers either iterate with range-for or collect to `std::vector` immediately. Returning `std::vector` is the simplest replacement with identical caller semantics. The allocation overhead is negligible for these host-only iteration utilities.
**Options:**
- (a) Replace with `Kokkos::MDRangePolicy<Rank<3>>` for all iteration
- **(b) Replace with plain nested loops (host-only code paths)** ← CHOSEN
- (c) Keep coroutines for host-side iteration, add Kokkos parallel paths alongside
**Considerations:** Generators are only used in `index_view.hpp` and `mesh_view.hpp`. Both are iteration utilities that map directly to MDRangePolicy.

### D4: Field Storage Migration
**Decision:** **(c) Keep `std::vector` for host-only data; defer `Kokkos::View` for field data to GPU phase.**
Phase 1 removes range-v3 from the fields subsystem; it does not introduce Kokkos Views for storage. The `field` class stores `std::vector<scalar_real>` and `std::vector<vector_real>`, where the underlying element storage is `std::vector<real>` inside nested `tuple` wrappers. These remain `std::vector` throughout Phase 1. Range-v3 concepts (`rs::range`, `rs::sized_range`, `rs::output_range`, etc.) are replaced with `std::ranges` equivalents, but the storage types are unchanged. Migration to `Kokkos::View<real*>` will happen in a future GPU-enablement phase after correctness is validated with the host-only range removal.
**Options:**
- (a) Replace `std::vector<real>` with `Kokkos::View<real*>` everywhere
- (b) Use `Kokkos::DualView<real*>` for fields that need host+device access
- **(c) Keep `std::vector` for host-only data (mesh coords, stencil coefficients), use `Kokkos::View` for field data** ← CHOSEN
**Considerations:** Option (c) avoids unnecessary changes to setup-time-only data while migrating the hot-path field storage. Phase 1 only addresses range-v3 removal, so storage stays as `std::vector`.

### D5: Selector/View Adaptor Replacement
**Decision:** **(d) Replace `rs::view_adaptor` with standalone custom views using `std::ranges::view_interface`; defer index arrays to GPU phase.**
The 4 custom `rs::view_adaptor` classes (`plane_view<1>`, `multi_slice_view`, `optional_view`, `predicate_view`) and the 2 composition-based views (`plane_view<0>`, `plane_view<2>`) are rewritten as standalone C++20 views:
- `plane_view<0>` (x-plane): Replace `vs::drop_exactly | vs::take_exactly` with `std::views::drop | std::views::take` or `std::ranges::subrange`.
- `plane_view<1>` (y-plane): Replace `rs::view_adaptor` inheritance with `std::ranges::view_interface`. Preserve the existing strided-iterator logic but implement iterator/sentinel directly (no `adaptor_base`).
- `plane_view<2>` (z-plane): Replace `vs::drop_exactly | vs::stride` with a custom strided view (C++20 has no `std::views::stride`; that's C++23).
- `multi_slice_view`: Replace `rs::view_adaptor` with `std::ranges::view_interface`. Preserve slice-navigation iterator logic.
- `optional_view`: Replace with a simple conditional `std::ranges::subrange` (empty or full).
- `predicate_view`: Replace `rs::view_adaptor` with `std::ranges::view_interface`. Preserve filter-style iteration.
Range-v3 internal utilities (`rs::semiregular_box_t`, `rs::make_view_closure`, `rs::bind_back`, `rs::compose`) are replaced with project-local equivalents in `src/fields/ccs_range_utils.hpp` (see D8).
Pre-computed index arrays (option a) are deferred to the GPU migration phase.
**Options:**
- (a) Pre-computed index arrays (`Kokkos::View<int*>`) for all selections
- (b) Boolean masks (`Kokkos::View<bool*>`) for all selections
- (c) Mixed: index arrays for sparse selections (Dirichlet BCs, fluid), masks for dense selections
- **(d) Custom `std::ranges::view_interface` classes preserving existing iterator logic** ← CHOSEN
**Considerations:** Option (d) minimizes behavioral risk in Phase 1 by preserving identical iterator semantics. The custom views can later be replaced with Kokkos-based index arrays when GPU support is added.

### D6: Matrix-Vector Product Migration
**Decision:** **(b) Write all custom explicit loops (no KokkosKernels dependency).**
The matrices are small per-line operators (dense: 2–5 rows, circulant: stencil width 3–7, CSR: sparse boundary coupling), not global sparse systems. KokkosKernels (option a) is overkill for these sizes. Per D1 (host-only execution) and D4 (keep `std::vector` storage), there is no device memory to manage (option c is N/A). Replace range-v3 view pipelines (`vs::zip_with`, `vs::chunk`, `vs::repeat_n`, `vs::sliding`, `vs::stride`, `vs::zip`) with explicit `for` loops using `std::inner_product` for dot products. Phase 1 utilities (`ccs::stride`, `ccs::zip_transform`, `ccs::repeat_n` from `fields/lazy_views.hpp`) are available but explicit loops are simpler and clearer for these small fixed-size operations. Kokkos parallel kernels can be introduced in a future GPU-enablement phase.
**Options:**
- (a) Use KokkosSparse::CrsMatrix + KokkosSparse::spmv for CSR; custom kernels for dense/circulant
- **(b) Write all custom explicit loops (no KokkosKernels dependency)** ← CHOSEN
- (c) Keep matrix assembly on host, copy to device, apply via Kokkos kernels
**Considerations:** The matrices are small per-line operators (not global sparse systems), so KokkosKernels may be overkill. Custom kernels give more control over the composite inner_block structure.

### D7: Kokkos Include Propagation Strategy
**Decision:** **(b) Create a separate `kokkos_types.hpp` header; add Kokkos to each CMake target incrementally as its phase migrates.**
`types.hpp` is included by 35+ headers across all subsystems. Adding Kokkos there in Phase 0 forces every library target to link Kokkos before any of them actually use it. Option (b) is less disruptive: Phase 0 creates the header and links Kokkos only to `indexing` and `shoccs-exe`. Phase 1 (fields) adds Kokkos to the `fields` INTERFACE library (which most targets already depend on), at which point `kokkos_types.hpp` can optionally be merged into `types.hpp`.
**Options:**
- (a) Add `Kokkos_Core.hpp` to `types.hpp` immediately; update all ~17 CMake library targets to link `Kokkos::kokkos`
- **(b) Create a separate `kokkos_types.hpp` header; add Kokkos to each CMake target incrementally as its phase migrates** ← CHOSEN
**Considerations:** `types.hpp` is included by 35+ headers across all subsystems. Adding Kokkos there in Phase 0 forces every library target to link Kokkos before any of them actually use it.

### D8: C++20 Range Utility Replacements
**Decision:** **(a) Create project-local utilities in `src/fields/ccs_range_utils.hpp` for range-v3 internal APIs with no C++20 `std::ranges` equivalent.**
The project uses C++20 (`CMAKE_CXX_STANDARD 20`). Several range-v3 features used in the fields subsystem have no direct C++20 equivalents:
- `vs::view_closure<Fn>` → project-local `ccs::view_closure<Fn>` (pipeable callable wrapper)
- `rs::make_view_closure(fn)` → project-local `ccs::make_view_closure(fn)`
- `rs::bind_back(fn, args...)` → project-local `ccs::bind_back(fn, args...)` (C++23 has `std::bind_back`)
- `rs::compose(f, g)` → project-local `ccs::compose(f, g)`
- `rs::semiregular_box_t<Fn>` → project-local `ccs::semiregular_box<Fn>` using `std::optional`
- `vs::zip_with(f, rngs...)` → project-local `ccs::zip_transform_view` (C++23 has `std::views::zip_transform`)
- `vs::zip(rngs...)` → in-place operators use index-based iteration; `field_utils.hpp` uses project-local zip
- `vs::repeat_n(v, n)` → `std::views::transform` with constant lambda where possible; project-local `ccs::repeat_n_view` where lazy view is needed
- `vs::stride(n)` → project-local `ccs::stride_view` (C++23 has `std::views::stride`)
- `vs::concat(rngs...)` → test-only; replace with eager `std::vector` construction
- `vs::take_exactly(n)` → `std::views::take(n)` (C++20; slightly different: checks bounds)
- `vs::drop_exactly(n)` → `std::views::drop(n)` (C++20; slightly different: checks bounds)
- `rs::to<T>()` → test-only; replace with `T(std::ranges::begin(r), std::ranges::end(r))`
- `rs::common_tuple` → remove specialization (only produced by range-v3 zip; no longer needed)

All project-local utilities go in `src/fields/ccs_range_utils.hpp`, are header-only, and are minimal implementations sufficient for the fields subsystem usage patterns.
**Options:**
- **(a) Project-local minimal utilities in `ccs_range_utils.hpp`** ← CHOSEN
- (b) Upgrade to C++23 and use `std::views::zip_transform`, `std::views::stride`, `std::bind_back`, etc.
- (c) Replace all lazy views with eager `std::vector`-based computation
**Considerations:** Option (b) would be simpler but requires verifying compiler/Kokkos C++23 support. Option (c) changes performance characteristics of lazy expression trees in `tuple_math`. Option (a) is safest for Phase 1.

### D9: `vs::cartesian_product` Replacement Strategy
**Decision:** **(a) Add a project-local `ccs::cartesian_product_view` to `src/fields/lazy_views.hpp`.**
`vs::cartesian_product(x, y, z)` is used in `cartesian.hpp` (`domain()`) and `selections.hpp` (`location()`). C++23 has `std::views::cartesian_product` but C++20 does not. The project targets C++20 (per `CMAKE_CXX_STANDARD 20`).
The new `ccs::cartesian_product_view` lazily yields `std::tuple<T1&, T2&, T3&>` in triple-nested-loop order (first range slowest, third range fastest). It models `std::ranges::view_interface` with at least forward iteration. This matches the existing range-v3 `vs::cartesian_product` behavior and the mesh layout convention (x-slowest, z-fastest).
Eager alternatives (returning `std::vector<real3>`) were considered but rejected because `domain()` is stored as a member type in `mesh.hpp` using `decltype`, and changing it to a vector would alter the type and semantics of the fields framework's location tuples.
**Options:**
- **(a) Project-local `ccs::cartesian_product_view` in `lazy_views.hpp`** ← CHOSEN
- (b) Upgrade to C++23 for `std::views::cartesian_product`
- (c) Change `domain()` API to return `std::vector<std::tuple<real,real,real>>`
**Considerations:** Option (a) is consistent with D8 (project-local utilities for missing C++20 features). Option (c) would change the mesh.hpp member types and downstream field DSL usage.

---

## Build & Test Commands

```bash
# Configure (from repo root)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON

# Build all
cmake --build build

# Run all tests
ctest --test-dir build

# Run tests by label
ctest --test-dir build -L fields
ctest --test-dir build -L matrices
ctest --test-dir build -L operators
ctest --test-dir build -L stencils
ctest --test-dir build -L mesh
ctest --test-dir build -L systems
ctest --test-dir build -L temporal
ctest --test-dir build -L simulation

# Run a specific test
ctest --test-dir build -R t-dense
```

---

## Files Excluded from Migration Scope

- `src/systems/cc_elliptic.hpp/cpp` — Legacy code using old inheritance-based API; references missing `pudding_limits.hpp`. Dead code.
- `src/operators/directional.hpp/cpp` — Commented out of CMakeLists. Uses older `geometry`/`domain_boundaries` API. Can be migrated later if re-enabled.
- `src/operators/discrete_operator.hpp` — Stub, unused.
- `src/operators/divergence.hpp` — Stub, unused.
- `src/sentinels.hpp` — Empty file.
- `src/fields/view_tuple_seg.cpp` — Debugging standalone, not production code.

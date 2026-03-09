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
**Decision:** TBD (to be resolved in Phase 0 planning)
**Options:**
- (a) Start with `Kokkos::DefaultHostExecutionSpace` everywhere, port to GPU later
- (b) Use `Kokkos::DualView` from the start
- (c) Use `Kokkos::View` with explicit `Kokkos::DefaultExecutionSpace` and mirror views for I/O
**Considerations:** The code is CPU-only today. Starting host-only minimizes risk and lets us validate correctness before adding device portability.

### D2: range-v3 Removal Strategy
**Decision:** TBD (to be resolved in Phase 0 planning)
**Options:**
- (a) Remove `range-v3` dependency entirely, replace all uses with Kokkos + std
- (b) Keep `range-v3` for host-side-only convenience code (tests, init), remove from hot paths
- (c) Replace with C++20/23 `std::ranges` where possible, Kokkos for parallel ops
**Considerations:** Option (c) is pragmatic — `std::ranges` covers many simple view compositions (transform, filter, take, drop). Kokkos is needed for parallel execution patterns. Tests can continue using range-v3 or std::ranges.

### D3: cppcoro Generator Replacement
**Decision:** TBD (to be resolved in Phase 0)
**Options:**
- (a) Replace with `Kokkos::MDRangePolicy<Rank<3>>` for all iteration
- (b) Replace with plain nested loops (host-only code paths)
- (c) Keep coroutines for host-side iteration, add Kokkos parallel paths alongside
**Considerations:** Generators are only used in `index_view.hpp` and `mesh_view.hpp`. Both are iteration utilities that map directly to MDRangePolicy.

### D4: Field Storage Migration
**Decision:** TBD (to be resolved in Phase 1 planning)
**Options:**
- (a) Replace `std::vector<real>` with `Kokkos::View<real*>` everywhere
- (b) Use `Kokkos::DualView<real*>` for fields that need host+device access
- (c) Keep `std::vector` for host-only data (mesh coords, stencil coefficients), use `Kokkos::View` for field data
**Considerations:** Option (c) avoids unnecessary changes to setup-time-only data while migrating the hot-path field storage.

### D5: Selector/View Adaptor Replacement
**Decision:** TBD (to be resolved in Phase 1 planning)
**Options:**
- (a) Pre-computed index arrays (`Kokkos::View<int*>`) for all selections
- (b) Boolean masks (`Kokkos::View<bool*>`) for all selections
- (c) Mixed: index arrays for sparse selections (Dirichlet BCs, fluid), masks for dense selections
**Considerations:** Index arrays are better for GPU divergence. The current `plane_view`, `multi_slice_view`, `optional_view`, `predicate_view` custom adaptors all need replacement.

### D6: Matrix-Vector Product Migration
**Decision:** TBD (to be resolved in Phase 2 planning)
**Options:**
- (a) Use KokkosSparse::CrsMatrix + KokkosSparse::spmv for CSR; custom kernels for dense/circulant
- (b) Write all custom Kokkos kernels (no KokkosKernels dependency)
- (c) Keep matrix assembly on host, copy to device, apply via Kokkos kernels
**Considerations:** The matrices are small per-line operators (not global sparse systems), so KokkosKernels may be overkill. Custom kernels give more control over the composite inner_block structure.

### D7: Kokkos Include Propagation Strategy
**Decision:** TBD (to be resolved in Phase 0, item 0.5)
**Options:**
- (a) Add `Kokkos_Core.hpp` to `types.hpp` immediately; update all ~17 CMake library targets to link `Kokkos::kokkos`
- (b) Create a separate `kokkos_types.hpp` header; add Kokkos to each CMake target incrementally as its phase migrates
**Considerations:** `types.hpp` is included by 35+ headers across all subsystems. Adding Kokkos there in Phase 0 forces every library target to link Kokkos before any of them actually use it. Option (b) is less disruptive: Phase 0 creates the header and links Kokkos only to `indexing` and `shoccs-exe`. Phase 1 (fields) adds Kokkos to the `fields` INTERFACE library (which most targets already depend on), at which point `kokkos_types.hpp` can optionally be merged into `types.hpp`. Option (b) is recommended.

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

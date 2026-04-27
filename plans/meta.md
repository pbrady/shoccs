# SHOCCS Kokkos Migration — Meta Plan

**Purpose:** Cross-cutting decisions and phase ordering for the range-v3 → Kokkos migration and DSL restructuring.

---

## Phase Ordering

### Completed: range-v3 Removal (Phases 0–7)

| Phase | Plan File | Status |
|-------|-----------|--------|
| 0 | `00-foundation.md` | DONE |
| 1 | `01-fields.md` | DONE |
| 2 | `02-matrices.md` | DONE |
| 3 | `03-stencils.md` | DONE |
| 4 | `04-operators.md` | DONE |
| 5 | `05-systems.md` | DONE |
| 6 | `06-temporal.md` | DONE |
| 7 | `07-simulation-io-mms.md` | DONE |

### Active: DSL Restructuring and Kokkos Execution (Phases 8–14)

| Phase | Plan File | Depends On | Goal |
|-------|-----------|------------|------|
| 8 | `08-registry-and-handles.md` | Phases 0–7 | Field registry + handle types + span bridge |
| 9 | `09-field-lifecycle.md` | Phase 8 | Migrate system/integrator/sim_cycle to registry + field_ref |
| 10 | `10-expression-templates.md` | Phases 8, 9 | Expression templates + Kokkos parallel_for dispatch |
| 11 | `11-selector-migration.md` | Phases 8, 9 | Selection descriptors replace iterator-based views |
| 12 | `12-legacy-removal.md` | Phases 8–11 | Delete old tuple infrastructure, rewrite tests |
| 13 | `13-kokkos-parallel.md` | Phases 8, 10 | Parallelize matrix-vector products (block, circulant) |
| 14 | `14-kokkos-gpu.md` | Phases 8–13 | GPU execution: device memory, host mirrors, KOKKOS_LAMBDA |

### Tooling: Stencil Derivation (Phases 20–21)

| Phase | Plan File | Depends On | Goal |
|-------|-----------|------------|------|
| 20 | `20-stencil-derivation-pipeline.md` | None (standalone) | SymPy pipeline for deriving stencil coefficients and generating C++ |
| 21 | `21-e4-cut-cell-stencils.md` | Phase 20 | E4_1 cut-cell stencil generation via generalized TEMO pipeline |

### Analysis: Group Velocity (Phases 34–36)

| Phase | Plan File | Depends On | Goal |
|-------|-----------|------------|------|
| 34 | `34-group-velocity-analysis.md` | Phase 33 | Core group velocity module, interior + boundary analysis |
| 35 | `35-group-velocity-cut-cell.md` | Phase 34 | Cut-cell psi-dependent group velocity, psi sweeps |
| 36 | `36-group-velocity-2d.md` | Phase 35 | 2D/3D extension, varying coefficients, scaling comparison |

### Maintenance: Test Suite (Phase 37)

| Phase | Plan File | Depends On | Goal |
|-------|-----------|------------|------|
| 37 | `37-test-suite-refactoring.md` | None (standalone) | Separate research sweeps from regression tests, reduce default suite to <15s |
| 38 | `38-sweep-extraction.md` | Phase 37 | Extract research sweeps to standalone scripts, remove from pytest |

### Design Documents

| Document | Purpose |
|----------|---------|
| `kokkos-view-migration-impact.md` | Impact analysis of Kokkos::View on current storage/DSL |
| `dsl-restructuring-proposal.md` | Full design proposal with review findings |

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

### D10: Dead Code Cleanup for `directional.cpp`/`directional.t.cpp`
**Decision:** **(a) Delete these files entirely.**
`directional.cpp`, `directional.t.cpp`, and `directional.hpp` are commented out of `src/operators/CMakeLists.txt` (line 21). They use an older API (`geometry` class, `domain_boundaries`, `mesh` constructor with positional args) that no longer exists in the current codebase, and have heavy range-v3 + cppcoro dependencies (5+ range-v3 includes in the .cpp, 7 includes in the test, cppcoro generator). Migrating these files would require both range-v3 removal AND API updates for a code path that is not compiled or tested. The `directional` operator was superseded by the `derivative` operator. Similarly, `src/io/format_test.cpp` is a standalone demo (not in the build) with range-v3 usage — delete it too.
**Options:**
- **(a) Delete the files entirely** ← CHOSEN
- (b) Fully migrate range-v3 usage and update to current API
- (c) Keep files as-is (dead code with range-v3 dependencies)
**Considerations:** Option (b) would require updating to APIs that no longer exist (`geometry`, `domain_boundaries`). The functionality can be reimplemented from scratch if needed in the future. Option (a) is the pragmatic choice for achieving zero range-v3 references.

### D11: E2_2.t.cpp `#if 0` Blocks
**Decision:** **(a) Delete the `#if 0` blocks entirely.**
`src/stencils/E2_2.t.cpp` has two `#if 0` blocks (lines 209–284, 287–453) containing ~25 range-v3 call sites in permanently disabled test cases. The first block tests wall interpolation edge cases; the second tests a quadratic interpolant with 3-point and 4-point stencils that differ from the active test configurations. These were disabled during development and the active tests (lines 1–208) provide sufficient coverage.
**Options:**
- **(a) Delete the `#if 0` blocks** ← CHOSEN
- (b) Migrate all ~25 call sites using the same patterns as active code
**Considerations:** Migrating dead test code adds ~50 lines of diff with no test coverage benefit. The patterns are identical to active code, so if these tests are needed in the future they can be rewritten using the migrated active code as a template.

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

## DSL Restructuring Decisions (Phases 8–14)

### D-R1: Centralized Registry — Storage Never Moves
**Decision:** All field buffer storage lives in a `field_registry` singleton (or simulation-scoped local). The registry is never copied, moved, or passed by value. Only `field_ref` handles (trivially-copyable integer structs) flow through the system.
**Why:** Eliminates the `container_tuple`/`view_tuple` re-anchoring machinery (~310 lines), prevents the 3× heap allocation in `system::rhs`, and makes `swap(u0, u1)` a 2-integer swap.

### D-R2: Max Capacity at Compile Time, Runtime Allocation
**Decision:** `field_registry<MaxSlots, MaxScalars, MaxVectors>` uses `std::array<Kokkos::View<real*>, N>` with compile-time max capacity. Runtime `n_scalars`/`n_vectors` tracks active slots. Unused slots are default-constructed Views (24 bytes, no allocation).
**Why:** Preserves a single non-templated `field_ref` type for the system/integrator interface (no variant dispatch, no template cascading). All current systems use `(ns,nv) ≤ (1,0)`.

### D-R3: field_ref Replaces field_span and field_view
**Decision:** `field_ref { int slot; int n_scalars; int n_vectors; }` (12 bytes, trivially copyable) replaces `field_span` and `field_view` as the type passed through the system/integrator chain. Const-correctness moves to the function signature (`const field_registry&` vs `field_registry&`).
**Why:** Eliminates heap allocation on every by-value pass of `field_span`/`field_view`.

### D-R4: Expression Nodes Carry Raw Pointers, Not Registry References
**Decision:** `handle_expr { real* ptr; }` pre-extracts pointers from the registry on the host before `KOKKOS_LAMBDA` capture. The registry reference never enters a kernel.
**Why:** `field_registry` contains `std::vector`/`std::array` (not trivially copyable). `KOKKOS_LAMBDA` requires trivially copyable captures for GPU compatibility.

### D-R5: Handles Store Only Buffer Index, Not Length
**Decision:** `buf_handle { int id; }` stores only the buffer index. Lengths are queried from the registry at kernel-launch time (`registry.size(handle)`).
**Why:** Cached lengths become stale if Views are resized. A single query at launch time is negligible cost.

### D-R6: Coexistence — New System Introduced Alongside Old
**Decision:** Phases 8–11 introduce the registry/handle/expression/selector system as new code alongside the existing tuple infrastructure. Phase 12 deletes the old code. No existing tests break until Phase 12 rewrites them.
**Why:** Allows incremental validation. Each phase can be tested independently.

### D-R7: Selection Descriptors Replace Iterator-Based Views
**Decision:** Pre-computed `contiguous_selection`, `strided_selection`, and `gather_selection` descriptors replace `plane_view<0/1/2>`, `multi_slice_view`, and `predicate_view`. Index arrays for gather selections are built once at mesh construction time and cached.
**Why:** Iterator-based views hold host pointers and `std::optional` state — incompatible with `KOKKOS_LAMBDA`. Descriptors are trivially copyable.

### D-R8: Aliasing Detection in Expression Assignment
**Decision:** `assign()` checks at runtime (pointer comparison) that the destination buffer does not appear in the source expression. If aliased, evaluation stages through a temporary buffer. Mutating operators (`+=`, `-=`) are always safe (element-wise, no neighbor dependencies).
**Why:** `Kokkos::parallel_for` has no ordering guarantees between iterations.

### D-R10: Simulation Registry Concrete Type
**Decision:** A single concrete type alias `sim_registry = field_registry<8, 8, 4>` is used throughout the simulation chain (system, integrator, simulation_cycle). MaxSlots=8 (rk4 uses 4 slots: u0, u1, rk_rhs, system_rhs; 8 provides headroom). MaxS=8, MaxV=4 matches `general_layout` and accommodates all current systems (heat: 1,0; scalar_wave: 1,0; eigenvalues: 0,0).
**Why:** The `system` and `integrator` variant wrappers are non-template classes. Their new registry-based methods need a concrete registry type in the signature. Templating the wrappers would cascade template parameters through the entire simulation chain. Using a fixed alias avoids this while providing sufficient capacity for all systems.

### D-R11: Integrator Scratch Slot Ownership
**Decision:** Scratch slots (rk_rhs, system_rhs) are allocated by `simulation_cycle::run()` and passed as `field_ref` tokens to the integrator. The integrator does not own or allocate scratch storage.
**Why:** Centralizing allocation in `simulation_cycle::run()` keeps the registry as a single-owner local. It also eliminates the lazy `ensure_size()` pattern where integrators resize their owned `field` members on first call.

### D-R12: System Adapter Delegation vs Standalone
**Decision:** Registry-based adapter methods in concrete systems follow two patterns:
- **Delegating**: Methods whose existing signatures already use `field_view`/`field_span` (non-owning types): `rhs`, `update_boundary`, `write`. The adapter constructs a temporary `field_view`/`field_span` from extracted `scalar_view`/`scalar_span` and calls the existing method.
- **Standalone**: Methods whose existing signatures use `const field&`/`field&` (owning types): `stats`, `timestep_size`, `operator()`/`initialize`. The adapter implements the logic directly using extracted `scalar_view`/`scalar_span`, since `field_view` is not assignment-compatible with `const field&` (they are different template instantiations of `detail::field`; conversion would require deep-copying all `std::vector<real>` buffers). The DSL operators (`|`, `sel::D`, mesh selections, `abs`, `max`, etc.) work generically on any `Scalar` type, so the adapter body is identical to the existing method body operating on `scalar_view`/`scalar_span` instead of `scalar_real`.
**Why:** `field = detail::field<std::vector<scalar_real>, std::vector<vector_real>>` is an owning type (backed by `std::vector<real>`). `field_view = detail::field<std::vector<scalar_view>, std::vector<vector_view>>` uses `std::span<const real>`. These are separate template instantiations — passing a `field_view` as `const field&` would require constructing a temporary `field` that copies all data, defeating the zero-copy goal. The standalone adapter avoids this by working directly on spans.

### D-R9: Kokkos Test Initialization Pattern
**Decision:** Tests that allocate `Kokkos::View` must provide a custom `main()` using `Kokkos::ScopeGuard` + Catch2 v3's `Catch::Session`. These tests link `Catch2::Catch2` (not `Catch2::Catch2WithMain`) and define:
```cpp
#include <Kokkos_Core.hpp>
#include <catch2/catch_session.hpp>
int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    return Catch::Session().run(argc, argv);
}
```
Tests that don't allocate Views (e.g., handle arithmetic tests) continue using `add_unit_test()` with `Catch2::Catch2WithMain`.
**Why:** `Kokkos::View` allocation requires `Kokkos::initialize()` to have been called. The existing `add_unit_test()` CMake function links `Catch2::Catch2WithMain`, which provides its own `main()` — incompatible with a custom main. Manually defining the test executable avoids this conflict.

### D-R13: Slot-Level Element-Wise Helper Functions
**Decision:** Extract three free functions in `src/temporal/slot_ops.hpp` for element-wise slot arithmetic: `slot_zero`, `slot_assign_lc` (dst = src + c * rhs), and `slot_accumulate` (dst += c * src). Both `rk4` and `euler` include this header. The helpers iterate over allocated buffers using `field_ref.n_scalars` (not `buffers_per_slot`) so unallocated buffer indices are never touched.
**Why:** rk4 uses `slot_assign_lc` (4 stages) + `slot_accumulate` (4 stages) + `slot_zero` (2 calls). euler uses `slot_assign_lc` (1 call) + `slot_zero` (1 call). Duplicating the triple-nested loop in both files would be ~40 lines of identical code. A shared header avoids DRY violations and ensures consistent iteration patterns.

### D-R14: handle_expr Const-Correctness for Expression Templates
**Decision:** TBD — either (a) use a single `handle_expr { real* ptr; }` and `const_cast` for const-registry reads, or (b) make `handle_expr` a template `handle_expr<T>` where `T` is `real*` or `const real*`, or (c) add a separate `const_handle_expr { const real* ptr; }`.
**Why:** `bind_scalar` from a `const field_registry&` yields `const real*` pointers, but expression nodes must be trivially copyable and capturable by `KOKKOS_LAMBDA`. The choice affects how `operator+` etc. compose expressions from const and mutable sources. For Phase 10 (host-only, mutable-registry only), option (a) suffices. The decision should be revisited in Phase 14 (GPU) if const-correctness matters for device memory.
**Status:** For Phase 10, use mutable `bind_scalar` only (single `handle_expr { real* ptr; }`). Const variant deferred.

### D-R15: Unified strided_selection for y-plane and z-plane
**Decision:** A single `strided_selection{offset, inner_count, outer_count, outer_stride}` struct handles both y-plane and z-plane access patterns. The z-plane is the degenerate case where `inner_count = 1`.
**Why:** y-plane selects `nx` groups of `nz` contiguous elements with `ny*nz` stride between groups. z-plane selects `nx*ny` single elements with `nz` stride. Both follow the formula `element(i) = offset + (i / inner_count) * outer_stride + (i % inner_count)`. Unifying avoids a separate type while keeping both patterns efficient.

### D-R16: assign_selected Uses Absolute Indices
**Decision:** `assign_selected(dst, desc, expr)` evaluates `dst[desc.element(i)] = expr(desc.element(i))` — the expression is called with the **absolute** flat index from `desc.element(i)`, not the relative selection index `i`.
**Why:** In BC application, source data is typically a full-domain buffer (e.g., manufactured solution evaluated at all grid points). Using absolute indices means `handle_expr{sol_ptr}` naturally reads `sol_ptr[flat_index]` without needing a separate index mapping. Source values are pre-evaluated into scratch buffers using the existing tuple DSL (D-R6 coexistence).

### D-R17: lazy_views.hpp Retention Scope (Phase 12)
**Decision:** After Phase 12, keep `stride_view`/`stride`, `repeat_n_view`/`repeat_n`, `cartesian_product_view`/`cartesian_product`, `linear_distribute`, and the C++20 `std::basic_common_reference` backport in `lazy_views.hpp`. Delete `zip_transform_view`/`zip_transform` (~250 lines) and remove `#include "ccs_range_utils.hpp"`.
**Why:** Codebase analysis confirms ongoing production usage: `stride` (matrix tests), `repeat_n` (`xdmf.cpp`), `cartesian_product` (`cartesian.hpp` `domain()`), `linear_distribute` (`cartesian.cpp`, stencil tests). Only `zip_transform_view` is replaced by expression templates.

### D-R18: system_size Redesign (Phase 12)
**Decision:** Replace `scalar<integer> scalar_size` member in `system_size` with 4 plain integer fields: `integer d_size, rx_size, ry_size, rz_size`. Move `system_size` from `field_fwd.hpp` to `field_registry.hpp`, removing its dependency on `scalar.hpp` → `tuple.hpp`. (`field_registry.hpp` chosen over a new `system_size.hpp` because it already defines the related `field_ref` type and is already `#include`d by all system files that use `system_size`.)
**Why:** `system_size` is used by all system `.hpp/.cpp` files. Its `scalar<integer>` member creates a transitive dependency on the entire tuple infrastructure being deleted. Plain integer fields eliminate this dependency while preserving the same information.

### D-R19: scalar_span/scalar_view Simplification (Phase 12)
**Decision:** Replace the tuple-based `scalar_span = scalar<std::span<real>>` and `scalar_view = scalar<std::span<const real>>` with simple structs holding 4 named span members (`D`, `Rx`, `Ry`, `Rz`). The span bridge functions (`extract_scalar_span`/`extract_scalar_view` in `field_registry.hpp`) are updated to construct these structs directly.
**Why:** The span bridge is used by `heat.cpp`, `scalar_wave.cpp`, and test files. The types must persist for Phase 12 coexistence, but can be trivially decoupled from the tuple hierarchy. Named members (`s.D`) are clearer than tuple-indexed access (`get<0>(get<0>(s))`).

### D15-1: Block Disjointness Assertion Placement
**Decision:** **(b) Assert in `block::builder::to_block()` at construction time, not in `operator()` per call.**
**Why:** Construction-time checking is cheaper (runs once per block), catches bugs earlier (at setup rather than first matvec), and avoids debug-mode overhead on the hot path. Inner blocks are immutable after construction, so the invariant can't be violated after the check.
**Options:**
- (a) Per-call check in `block::operator()` before `parallel_for`
- **(b) Construction-time check in `builder::to_block()`** ← CHOSEN

### D-R18: Stencil Derivation Tooling
**Decision:** **(a) Build a SymPy-based derivation pipeline in `scripts/stencil_gen/`.**
The existing Mathematica notebooks are too slow for iterating on new stencil schemes. A Python/SymPy pipeline gives version-controlled, reproducible derivations integrated into the repo. The pipeline derives coefficients symbolically and generates C++ code matching the existing `src/stencils/` patterns via `sympy.cse()`.
**Options:**
- **(a) SymPy pipeline in the repo** ← CHOSEN
- (b) Continue with Mathematica notebooks
- (c) Use a different CAS (Maple, SageMath)
**Considerations:** SymPy 1.14 is already available in the container. The `QQ(psi)` fraction field provides 24,000x speedup over naive symbolic substitution for TEMO cut-cell derivations. `uv` is available for dependency management.

### D-R22: TEMO Cut-Cell Column Convention and Variant Mapping
**Decision:** The C++ code and plan 20.5 use column ordering `[wall, x_0, x_1, ..., x_{t-1}]` (wall = column 0). The math reference uses `[f_0, f_delta, f_1, ...]` (wall = column 1). All formulas in plan 20.5 use the C++ convention.
**Variant mapping (floating/Dirichlet):**
- 1st derivative (nu=1): math reference B^{d,2} — wall gets weight for ALL rows, x_0 zeroed for ALL rows.
- 2nd derivative (nu=2): math reference B^{d,1} — row 0: wall zeroed, x_0 gets weight; rows >= 1: wall gets weight, x_0 zeroed.
- 2nd derivative Neumann: uses different variant B^{d,0} — see D-R24.
**Why:** The C++ stencil code (`E2_1.cpp`, `E2_2.cpp`) stores coefficients in `[wall, x_0, ...]` order. Matching this convention avoids column-swapping bugs. The variant mapping was verified numerically against `E2_2.cpp` at `psi=0`: row 0 = `[0, 1, -2, 1]` (wall zeroed) and row 1 = `[1, 0, -2, 1]` (x_0 zeroed), confirming B^{d,1}.

### D-R23: Neumann Cut-Cell Derivation via Augmented Taylor System
**Decision:** The Neumann cut-cell stencil is derived using the same TEMO procedure as the floating stencil, but with an augmented Vandermonde matrix that includes a virtual column for `h * f'(x_wall)`. The virtual column entries are `V[k, T] = delta_wall^{k-1} / (k-1)!` for k >= 1, `V[0, T] = 0`, where `delta_wall = -(psi + i)`. The `eta_i` coupling coefficient is an additional unknown solved from the Taylor system — it is not independently prescribed. The same Category A/B/C + conservation machinery handles both floating and Neumann, differing only in (a) the variant choice (D-R24) and (b) the Vandermonde width (T vs T+1 columns).
**Why:** The augmented approach reuses the entire TEMO solve infrastructure (`build_temo_vandermonde`, `solve_temo_row`, conservation) with minimal changes. The alternative (substitution of f(x_wall) via Taylor expansion from the floating stencil) requires higher-order truncation matching and produces less clean code. Verified by reconstructing all E2_2 Neumann coefficients from the augmented system.

### D-R24: Neumann Variant Selection (B^{d,0} for nu=2)
**Decision:** The Category A zeroed-column variant for 2nd-derivative Neumann stencils is B^{d,0} (row 0: x_0 zeroed, wall gets `alpha^{uN}`; rows >= 1: wall zeroed, x_0 gets `alpha^{uN}`). This is the **opposite** of the floating/Dirichlet variant B^{d,1} (D-R22).
**Why:** Verified against `E2_2.cpp` `nbs_neumann` at psi=0: row 0 = `[-2, 0, 2, 0]` (x_0 zeroed, wall = -2 from uniform Neumann) and row 1 = `[0, -2, 2, 0]` (wall zeroed, x_0 = -2). The math reference Step 6 ("all schemes use alpha^d_{i,0} = 0 for i >= 1") applies to the floating/Dirichlet stencils but not the Neumann stencil, because the Neumann stencil operates on the uniform Neumann base `B^{uN}_l` (which has different alpha^u values). For E2_2, the uniform Neumann is `[-2, 2, 0], eta^u = -2`.

### D-R25: E4 Dimension Formula (REVERTED in 27621e3)
**Decision:** **(e) REVERTED. Restore the original Eq. 11b formula: `r = q + 1 + nextra`.** The D-R25(d) correction `r = p + 1 + nextra` was wrong — it was based on matching Table 1's r+1=4 for E4_1, but Table 1 dimensions were ALSO wrong (the paper's errata or our reading of Table 1 was the root cause). The Eq. 11 formulas from the paper are correct as written; the issue was that D-R25(d) was trying to match incorrect target dimensions.
**Evidence for reversion:**
- With `r = q + 1 + nextra = 4` for E4_1, the cut-cell stencil has R=5, T=7.
- R=5 gives 4 weight unknowns for conservation (6 equations, 2 excess) — feasible.
- With the old `r = p + 1 + nextra = 3` (R=4), conservation had 3 unknowns for 6 equations (3 excess) — proven infeasible by Phase 22 investigations.
- The Eq. 11 sizing formula applies to 1st derivatives (nu=1). For 2nd derivatives (nu=2), `r_eff = r - 1` handles the overlap.
**Impact:** `compute_dimensions` was restored to `r = q + 1 + nextra` (commit 27621e3). Phase 23 addresses the downstream effects.
**Previous (wrong):**
- **(d) Fix r formula to `r = p + 1 + nextra`** ← WAS CHOSEN, NOW REVERTED

### D23-near-interior: Near-Interior Row Overflow Handling
**Decision:** **(a) Limit conservation-fixed columns in the near-interior row to avoid overdetermining the Taylor system.**
When the interior stencil extends beyond the T-frame (i.e., `r + p + 1 > t`), the `build_degenerate_stencil` and `solve_uniform_limit` functions must not fix more columns via conservation than the Taylor system can accommodate.
**Rule:** `n_conservation_cols = T - n_zeroed - n_eqs`, selecting the rightmost eligible conservation columns. Eligible columns are those where the interior contribution IC(j) is computable (always true for T-frame columns).
**Why:** With r=4, p=2, T=7, n_eqs=4: the old code fixed 3-4 conservation columns, leaving 3 unknowns for 4 Taylor equations (inconsistent). The fix limits conservation to 2 columns (degenerate) or 3 columns (uniform limit), keeping exactly `n_eqs` unknowns.
**Backward compatible:** For E2_1 (r=3, p=1, T=5, n_eqs=2), the formula gives the same column selection as the current code.
**Status:** TBD (Phase 23 item 23.2a/23.2b)

### D23-1: E4_1 Conservation Infeasible with Constant Weights at R=5
**Decision:** Conservation (SBP property) for the E4_1 cut-cell stencil is **infeasible with constant (ψ-independent) norm weights** at R=5. This was the expected-feasible case after the dimension correction from D-R25.
**Evidence:**
- Theta-linearization of the 6 conservation equations (21 scalar ψ-coefficients, 14 linearized unknowns) gives rank(M)=8, rank([M|b])=9 → rank gap = 1 (Rouché-Capelli inconsistency).
- Direct symbolic solve confirms: the weight solutions w_1..w_3 are rational functions of ψ, not constants. The system IS solvable for each specific ψ value (6 eqs in 9 unknowns, 3 free parameters), but the solutions vary with ψ.
- The standard SBP norm requires H = diag(ψ, w_1, w_2, w_3, w_4, 1, 1, ...) with constant w_i. This is structurally incompatible with E4_1's TEMO-derived stencil.
**Impact:** Items 23.3b–23.3d (conservation enforcement) and 23.4a (C++ regeneration with conservation) are blocked. The current non-conservative E4_1 stencil can still be used for accuracy (Taylor order maintained), but not for provable stability.
**Potential alternatives (not yet investigated):**
- (a) ψ-dependent norm: allow w_i = f_i(ψ) (rational in ψ). Non-standard but physically motivated (mesh varies with ψ). Requires positivity analysis for ψ ∈ (0,1].
- (b) Larger stencil: increase nextra to add more DOF (more rows/alphas). Increases the operator bandwidth.
- (c) Different boundary derivation: use a formulation that builds conservation in from the start (rather than post-hoc enforcement on the TEMO stencil).
**Test:** `test_e4_1_conservation_constant_weights_infeasible_r5` in `test_e4_cut_cell.py`

### DD22-4: Conservation Wall Column Target by Derivative Order
**Decision:** **(a) Column 0 target depends on `nu`: −1 for 1st derivative, 0 for 2nd derivative.**
For the 1st derivative (nu=1), the SBP property `Q + Qᵀ = B` where `B = diag(-1, 0, ..., 0, 1)` requires the norm-weighted column 0 sum to equal −1. For the 2nd derivative (nu=2), constant annihilation `HD₂·𝟏 = 0` requires ALL column sums to equal 0. The `build_cut_cell_conservation_system` function (Phase 22, item 22.2a) parameterizes the target by `nu`. The existing `conservation.py` module (`build_conservation_system`) hardcodes the nu=1 convention and is currently only used for nu=1 cases; updating it for nu=2 is out of scope for Phase 22.
**Options:**
- **(a) Target varies by nu: −1 for nu=1, 0 for nu=2** ← CHOSEN
- (b) Use −1 for all nu (incorrect for nu=2)
- (c) Parameterize by full boundary matrix B

### D-Opt-1: Multi-Objective Pareto Optimization Architecture (Plan 45)
**Decision:** Four cross-cutting choices for the NSGA-II-based multi-objective driver shipped in plan 45:

**(a) NSGA-II via pymoo over a pure-numpy implementation.**
pymoo 0.6 is the standard, maintained implementation of NSGA-II/NSGA-III with a stable `ElementwiseProblem`/`Callback`/`HV` surface that composes cleanly with our existing scalar `make_objective` pattern. Writing a pure-numpy NSGA-II (fast non-dominated sort + crowding distance + tournament selection + SBX/polynomial operators) would be ~300 lines of optimizer code plus hypervolume-indicator code, all needing long-term maintenance. pymoo has no PyTorch/TensorFlow dependency (unlike BoTorch/Trieste), so the devcontainer footprint stays small. The dependency is opt-in at call sites — the existing scalar `sweeps optimize` path does not import pymoo.
**Options:**
- **(a) pymoo NSGA-II** ← CHOSEN
- (b) Pure-numpy implementation
- (c) BoTorch / Trieste (Bayesian multi-objective — deferred to plan 46)

**(b) Per-run JSON persistence under `sweeps/pareto_fronts/` over a `known_values.json` key.**
A full Pareto front is 15–40 members, each with `(x, params, objectives, report)` — hundreds of KB per run. Storing this under `known_values.json["brady2d_pareto"]` would bloat that file, create concurrency pitfalls (two concurrent runs racing on a single JSON), and produce noisy git diffs where a single member change rewrites the whole file. One file per `(scheme, kernel, objective_mangle)` gives O(1) updates, trivial parallel-safe concurrency, and clean diffs. Filenames are mangled via `{scheme}_{kernel}_{field1_dots_to_underscores}__{field2...}.json`. The directory is committed (via `.gitkeep`) and tracked (not in `.gitignore`), matching `output/` convention.
**Options:**
- **(a) Per-run JSON files under `sweeps/pareto_fronts/`** ← CHOSEN
- (b) New key in `sweeps/known_values.json`
- (c) SQLite / DuckDB

**(c) Finite sentinel `1e12` over `+inf` for gate-fail / exception returns in `make_multi_objective`.**
pymoo's hypervolume indicator (`pymoo.indicators.hv.HV`) rejects `+inf` objective values, and NSGA-II's `ftol` termination criterion propagates NaN when any column has `+inf`. The scalar `make_objective` returns `+inf` on gate-fail without issue because scipy/nlopt scalar drivers tolerate it; multi-objective must diverge from this convention. Sentinel `1e12` is large enough to dominate any realistic objective value (all measured SHOCCS stability metrics fall below ~100) but finite, keeping HV and non-dominated sort well-defined. Sentinel rows are filtered out of the final `ParetoResult.front`; the count is recorded in `extras["n_sentinel_filtered"]`.
**Options:**
- **(a) Finite sentinel `1e12`** ← CHOSEN
- (b) `+inf` (pymoo HV/ftol break)
- (c) `np.nan` (NSGA-II non-dominated sort breaks — NaN is incomparable)

**(d) `gate_layer` auto-infer as `max(max_layer - 1, 0)` applies to both scalar `make_objective` and multi-objective `make_multi_objective`.**
The old hardcoded `gate_layer=3` in `make_objective` made L6/L7 objectives never gate (waste — no early exit on L2 failure) and L3r objectives gate themselves (`+inf` trap documented in plan 44's handoff, reproduced as the 2634-eval DE run). The fix is to infer `gate_layer` from the objective field's native layer: `gate_layer = max(max_layer - 1, 0)`. The multi-objective factory (`make_multi_objective`) uses the max across its objective fields. Both entry points compute the value identically — one concept, two call sites. The `max(..., 0)` floor handles the degenerate `max_layer=1` case (no gate; objective is the only layer).
**Options:**
- **(a) `gate_layer = max(max_layer - 1, 0)` auto-infer in both scalar + multi-objective factories** ← CHOSEN
- (b) Hardcoded `gate_layer=3` (old behavior — broken for L3r, L6, L7)
- (c) Auto-infer only in `make_multi_objective`, leave `make_objective` hardcoded (inconsistent)

**Why these are cross-cutting:** choices (a)–(c) pin the optimizer / persistence / sentinel surface that plans 47 (multi-fidelity Bayesian) and 48 (Brady-Livescu 1D Euler) will build on or contrast against. Choice (d) ties together `stencil_gen/optimizer.py` and `stencil_gen/pareto.py` — any future scalar driver (plan 47's Bayesian surrogate scalarization) must observe the same auto-infer contract so users can swap `make_objective` ↔ `make_multi_objective` without re-specifying `gate_layer`.
**Implementing plan items:** 45.0b (auto-infer in `make_objective`), 45.0d (CLI wiring), 45.1b (`make_multi_objective`), 45.2a (`run_nsga2`), 45.4a (`_pareto_io.py`).

---

## Files Excluded from Migration Scope

- `src/systems/cc_elliptic.hpp/cpp` — Legacy code using old inheritance-based API; references missing `pudding_limits.hpp`. Dead code.
- `src/operators/directional.hpp/cpp/t.cpp` — Commented out of CMakeLists. Uses older `geometry`/`domain_boundaries` API. **Decision D10: delete in Phase 7 (item 7.24a).**
- `src/operators/discrete_operator.hpp` — Stub, unused.
- `src/operators/divergence.hpp` — Stub, unused.
- `src/sentinels.hpp` — Empty file.
- `src/fields/view_tuple_seg.cpp` — Debugging standalone, not production code.

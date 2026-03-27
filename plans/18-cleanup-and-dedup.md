# Phase 18: Cleanup, Deduplication, and Dead Code Removal

**Goal:** Address all findings from the 6-team code review: extract duplicated system logic into shared utilities, fix bugs, remove ~800 lines of dead code, and clean up architectural smells.

**Depends on:** Phase 17 complete

**Read first:**
- `src/systems/heat.cpp` (duplicated methods: eval_at_locations, stats, initialize, write)
- `src/systems/scalar_wave.cpp` (same duplicated methods)
- `src/fields/handle.hpp` (~300 lines of dead infrastructure)
- `src/fields/expr.hpp` (~170 lines of unused scalar_expr operators and reductions)
- `src/fields/scalar.hpp` (tuple-based converting constructors)
- `src/matrices/block.hpp` (duplicated operator()/graph_node lambda)
- `src/matrices/inner_block.hpp` + `inner_block.cpp` (dead after TeamPolicy)
- `src/operators/derivative.hpp` + `derivative.cpp` (dead `interior_c` member, double fence)
- `src/operators/laplacian.hpp` (dead `logger` member)
- `src/temporal/euler.cpp` (bypasses Graph API)

**Test commands:**
```bash
cmake --build build
ctest --test-dir build
```

---

## Items

### 18.1 — Bugs

- [x] **18.1a** Fix `euler.cpp` to use `submit_rhs_graph()` instead of `sys.rhs()`:
  - Changed `sys.rhs(...)` to `sys.submit_rhs_graph(...)` in `euler.cpp`.
  - Updated `euler_v2.t.cpp` to call `sys.build_rhs_graph()` before the euler step (graph must be pre-built, using `u0_ref` as input since euler reads from u0 directly).
  - File: `src/temporal/euler.cpp`, `src/temporal/euler_v2.t.cpp`
  - Test: `ctest --test-dir build -L temporal` — all 3 passed

- [x] **18.1a-fixup** Fix graph slot mismatch for euler in the production path:
  - Applied Option 3: Restructured euler to use RK4's slot convention.
  - `euler.cpp`: Added `deep_copy_slot(output, u0)` before submit so the graph (bound to output slot) reads correct data; changed `submit_rhs_graph` to pass `output` instead of `u0`.
  - `integrator.cpp`: Changed euler dispatch to pass `scratch2` (=srhs_ref) instead of `scratch1` (=rk_ref), matching the graph's output binding.
  - `euler_v2.t.cpp`: Updated `build_rhs_graph` to use `(u1_ref, srhs_ref)` matching production convention.
  - `simulation_cycle.t.cpp`: Added "cycle - 2D euler" test exercising the full `simulation_cycle::run()` path with euler integrator.
  - Files: `src/temporal/euler.cpp`, `src/temporal/integrator.cpp`, `src/temporal/euler_v2.t.cpp`, `src/simulation/simulation_cycle.t.cpp`
  - Test: `ctest --test-dir build -R "t-euler|t-simulation_cycle"` — all passed

- [x] **18.1b** Guard `MinMax` reduction result when `fd.count() == 0` in `stats()`:
  - In both `heat.cpp` and `scalar_wave.cpp`, wrapped the MinMax result with a guard:
    ```cpp
    real u_min = fd.count() > 0 ? minmax_result.min_val : 0.0;
    real u_max = fd.count() > 0 ? minmax_result.max_val : 0.0;
    ```
  - Without this, empty fluid domains produce `+inf`/`-inf` stats.
  - Files: `src/systems/heat.cpp`, `src/systems/scalar_wave.cpp`
  - Test: `ctest --test-dir build -R "t-heat|t-scalar_wave"` — both passed

- [x] **18.1c** Remove double fence in Neumann derivative overload:
  - Extracted kernel-submission logic into private `apply_kernels()` helper (no fence).
  - Non-Neumann `operator()`: calls `apply_kernels()` then fences once.
  - Neumann `operator()`: calls `apply_kernels()`, then `N(...)`, then fences once.
  - Removed stale `// This is ugly` comment from the switch.
  - File: `src/operators/derivative.hpp`, `src/operators/derivative.cpp`
  - Test: `ctest --test-dir build -R t-derivative` — passed

### 18.2 — System duplication: extract shared utilities

- [x] **18.2a** Create `src/systems/detail/scalar_system_utils.hpp` with shared `eval_at_locations`:
  - Created `src/systems/detail/scalar_system_utils.hpp` with the heat.cpp version (has `bool parallel` parameter for Lua MMS safety) in `ccs::systems::detail` namespace.
  - Both `heat.cpp` and `scalar_wave.cpp` now `#include "detail/scalar_system_utils.hpp"` and use `using detail::eval_at_locations;` instead of local copies.
  - scalar_wave.cpp callers (which never passed a `parallel` arg) get the default `parallel=true` — behavior unchanged.
  - File: `src/systems/detail/scalar_system_utils.hpp` (new)
  - Files modified: `src/systems/heat.cpp`, `src/systems/scalar_wave.cpp`
  - Test: `ctest --test-dir build -R "t-heat|t-scalar_wave|t-simulation_cycle"` — all 3 passed

- [x] **18.2b** Extract shared `compute_scalar_stats()` into the utils header:
  - Added `compute_scalar_stats(const mesh&, const bcs::Object&, scalar_view u, scalar_view sol)` to `scalar_system_utils.hpp`.
  - Contains the full stats body: MinMax reduction on fluid D, MaxLoc error on D, per-component R MinMax/MaxLoc loop, with `fd.count() == 0` guard.
  - Both `heat::stats()` and `scalar_wave::stats()` now evaluate `sol`, then delegate to `detail::compute_scalar_stats()`.
  - Reduced ~90 duplicated lines per caller.
  - File: `src/systems/detail/scalar_system_utils.hpp`
  - Files modified: `src/systems/heat.cpp`, `src/systems/scalar_wave.cpp`
  - Test: `ctest --test-dir build -R "t-heat|t-scalar_wave|t-simulation_cycle"` — all 3 passed

- [x] **18.2c** Extract shared `initialize_scalar_field()` into the utils header:
  - Added `initialize_scalar_field(const mesh&, scalar_span u, scalar_span sol)` to `scalar_system_utils.hpp`.
  - Contains the common body: zero D with parallel_for, assign_selected at fluid indices, copy R buffers via parallel_for, fence.
  - `heat::initialize()` keeps its `if (!m_sol) return;` guard, evaluates sol, then delegates.
  - `scalar_wave::initialize()` evaluates sol, then delegates.
  - Added `#include "fields/expr.hpp"` and `#include "fields/selection_desc.hpp"` to the utils header (needed for `handle_expr` and `assign_selected`).
  - File: `src/systems/detail/scalar_system_utils.hpp`
  - Files modified: `src/systems/heat.cpp`, `src/systems/scalar_wave.cpp`
  - Test: `ctest --test-dir build -R "t-heat|t-scalar_wave|t-simulation_cycle"` — all 3 passed

- [x] **18.2d** Extract shared `write_scalar_error()` into the utils header:
  - Added `write_scalar_error(const mesh&, const bcs::Object&, const bcs::Grid&, scalar_view u, scalar_view sol, scalar_span error, field_io&, io_names, step_controller&, dt)` to `scalar_system_utils.hpp`.
  - Contains the full write body: zero error buffers via parallel_for, compute |u - sol| at fluid D indices and non-dirichlet R indices, fence, zero Dirichlet grid/object entries, build io_scalars, call io.write.
  - Both `heat::write()` and `scalar_wave::write()` now evaluate sol, then delegate to `detail::write_scalar_error()`.
  - Added `#include "io/field_io.hpp"` and `#include "temporal/step_controller.hpp"` to the utils header.
  - Reduced ~50 duplicated lines per caller.
  - File: `src/systems/detail/scalar_system_utils.hpp`
  - Files modified: `src/systems/heat.cpp`, `src/systems/scalar_wave.cpp`
  - Test: `ctest --test-dir build -R "t-heat|t-scalar_wave|t-simulation_cycle"` — all 3 passed

- [x] **18.2e** Fix heat `update_boundary` Neumann section to use `eval_at_locations`:
  - Replaced the serial `cartesian_product` loop filling `grad_d` with `eval_at_locations(m, lambda, grad, m_sol.is_thread_safe())`.
  - The lambda evaluates `m_sol.gradient(time, loc)[dir]` at each location.
  - This parallelizes the evaluation (when thread-safe) and respects Lua MMS thread-safety via the `parallel` parameter.
  - File: `src/systems/heat.cpp`
  - Test: `ctest --test-dir build -R "t-heat|t-simulation_cycle"` — both passed

### 18.3 — Dead code in fields infrastructure

- [x] **18.3a** Remove dead code from `handle.hpp`:
  - Deleted: `handle_sel` namespace (11 selector structs + instances, ~130 lines), `handle_for_each` (4 overloads, ~35 lines), 7 free-function index helpers (~40 lines), `make_scalar_handle_unchecked`/`make_vector_handle_unchecked` (~20 lines), 3 layout type aliases (~8 lines), `scalar_accessor`/`vector_accessor` templates (~15 lines), entire `detail::` verification block (~90 lines).
  - Kept: `field_layout`, `buf_handle`, `scalar_handle`, `vector_handle`, `make_scalar_handle`, `make_vector_handle`, structural type static_asserts.
  - Updated `handle.t.cpp`: removed tests for `unchecked` factories, `handle_sel` dispatch, `handle_for_each`, and paired iteration.
  - `handle.hpp` reduced from 613 lines to 195 lines (~68% reduction).
  - File: `src/fields/handle.hpp`, `src/fields/handle.t.cpp`
  - Test: `ctest --test-dir build -R t-handle` — passed

- [x] **18.3b** Remove dead code from `expr.hpp`:
  - Deleted: `scalar_expr` struct, `bind_scalar`, 12 `scalar_expr` operator overloads (+,-,*,/ for expr-expr, scalar-right, scalar-left), `assign_scalar`, `plus_assign_scalar`, all 5 reduction functions (`reduce_max`/`reduce_min`/`reduce_sum` for pointer + `reduce_max`/`reduce_sum` for expr).
  - Kept: `handle_expr`, `scalar_literal_expr`, `binary_expr`, `unary_expr`, `contains_ptr`, `assign`, `plus_assign`, `minus_assign`, `times_assign`, `divide_assign`, `times_assign_scalar` (used by heat.cpp).
  - Removed unused includes: `<array>`, `<cassert>`, `<functional>`, `<limits>` from `expr.hpp`.
  - Updated `expr.t.cpp`: removed tests for all deleted code, removed `<limits>` include.
  - `expr.hpp` reduced from 447 to 172 lines (~62% reduction, ~275 lines removed).
  - `expr.t.cpp` reduced from 582 to 268 lines (~314 lines removed).
  - File: `src/fields/expr.hpp`, `src/fields/expr.t.cpp`
  - Test: `ctest --test-dir build -R "t-expr|t-heat|t-scalar_wave|t-simulation_cycle"` — all 4 passed

- [x] **18.3c** Remove tuple-based converting constructors from `scalar.hpp`:
  - Deleted the template converting constructors in `scalar_span` and `scalar_view` that used `get<0>(get<0>(s))` — they referenced deleted `tuple.hpp` infrastructure. No callers existed.
  - `scalar.hpp` reduced from 89 to 70 lines (~19 lines removed).
  - File: `src/fields/scalar.hpp`
  - Test: `ctest --test-dir build -L fields` — all 5 passed

### 18.4 — Dead code: files and stubs

- [x] **18.4a** Delete dead files:
  - `src/systems/cc_elliptic.hpp` + `src/systems/cc_elliptic.cpp` (wrong namespace, won't compile, range-v3)
  - `src/sentinels.hpp` (empty, never included)
  - `src/utils/extents.hpp` + `src/utils/extents.cpp` (stub, never compiled, typo namespace)
  - `src/operators/discrete_operator.hpp` (stub, never used)
  - `src/operators/divergence.hpp` (stub, `div` member never called)
  - `src/mesh/mesh_view.hpp` + `src/mesh/mesh_view.t.cpp` (no production callers)
  - Remove `#include "discrete_operator.hpp"` from `heat.cpp`, `scalar_wave.cpp`, `hyperbolic_eigenvalues.cpp`.
  - Remove `#include "divergence.hpp"` and `divergence div` member from `inviscid_vortex.hpp`.
  - Remove `mesh_view` test from `src/mesh/CMakeLists.txt`.
  - Deleted 11 files total, removed 4 stale includes and 1 dead member.
  - Files deleted: `cc_elliptic.{hpp,cpp}`, `sentinels.hpp`, `extents.{hpp,cpp}`, `discrete_operator.hpp`, `divergence.hpp`, `mesh_view.{hpp,t.cpp}`
  - Files modified: `heat.cpp`, `scalar_wave.cpp`, `hyperbolic_eigenvalues.cpp`, `inviscid_vortex.hpp`, `src/mesh/CMakeLists.txt`
  - Test: `ctest --test-dir build` — all tests pass (4 pre-existing failures unrelated to this change)

- [x] **18.4b** Remove commented-out code blocks:
  - `src/simulation/simulation_cycle.t.cpp` — removed 75-line commented-out 3D test case.
  - `src/mms/lua_mms.cpp` — removed 22-line old struct/factory comments.
  - `src/stencils/E6u_1.cpp`, `E2_2.cpp`, `E2_1.cpp` — removed old stencil coefficient comments.
  - `src/mesh/mesh.cpp` — removed dead `offset` lambda, `off()` calls, `return std::nullopt`.
  - `src/mesh/mesh.hpp` — removed commented-out `location()` method.
  - `src/mesh/cartesian.hpp` — removed commented-out `int3 n_`.
  - `src/matrices/csr.hpp` — removed `// using CSR_Builder` and `// std::cout` debug line.
  - `src/operators/derivative.cpp` — removed commented-out assert and cp_shift.
  - `src/operators/laplacian.cpp` — removed stale misleading comment about block accumulation.
  - 11 files modified, ~120 lines of commented-out code removed.
  - Test: `ctest --test-dir build` — all pass (4 pre-existing failures unrelated)

### 18.5 — Architecture cleanup

- [x] **18.5a** Extract shared kernel functor from `block::operator()` and `block::graph_node`:
  - Extracted the duplicated ~50-line KOKKOS_LAMBDA body into a named `matvec_functor<Op>` template struct inside `block`.
  - The functor stores `meta`, `coeffs`, `x_ptr`, `b_ptr`, `op` as members with a `KOKKOS_INLINE_FUNCTION operator()`.
  - Both `operator()` and `graph_node` now construct and pass the shared functor instead of duplicating the lambda.
  - Reduced `block.hpp` by ~45 lines of duplicated kernel code.
  - File: `src/matrices/block.hpp`
  - Test: `ctest --test-dir build -R "t-block|t-inner_block|t-simulation_cycle"` — all 3 passed

- [x] **18.5b** Make `derivative::interior_c` a local variable:
  - Removed `std::vector<real> interior_c;` member from `derivative.hpp`.
  - Changed to `auto interior_c = std::vector<real>(2 * p + 1);` local in the constructor body in `derivative.cpp`.
  - File: `src/operators/derivative.hpp`, `src/operators/derivative.cpp`
  - Test: `ctest --test-dir build -R t-derivative` — passed

- [x] **18.5c** Remove dead `laplacian::logger` member:
  - Removed `std::shared_ptr<spdlog::logger> logger;` from `laplacian.hpp`. Never assigned or read.
  - File: `src/operators/laplacian.hpp`
  - Test: `ctest --test-dir build -R t-laplacian` — pre-existing failure (unrelated floating-point test)

- [x] **18.5d** Remove dead `enum class scalars` from `heat.cpp`:
  - Deleted `enum class scalars : int { u };`. Never referenced.
  - File: `src/systems/heat.cpp`
  - Test: `ctest --test-dir build -R t-heat` — passed

- [x] **18.5e** Remove debug `spdlog::debug` from `scalar_wave.cpp` constructor:
  - Deleted `spdlog::debug("-grad_G {}\n", gG_xrx[0]);`.
  - File: `src/systems/scalar_wave.cpp`
  - Test: `ctest --test-dir build -R t-scalar_wave` — passed

- [x] **18.5f** Remove stale includes:
  - `src/io/field_io.hpp`: removed `<iostream>` and `<tuple>`.
  - `src/io/field_io.cpp`: removed `<fstream>` and `<iomanip>`.
  - `src/io/xdmf.cpp`: removed `<iomanip>`.
  - `src/systems/scalar_wave.cpp`: removed `<limits>` and `<iterator>`.
  - Deleted `src/temporal/empty_integrator.cpp` (empty compilation unit — `empty` is header-only).
  - Updated `src/temporal/CMakeLists.txt` to remove `empty_integrator.cpp` from source list.
  - Files modified: `field_io.hpp`, `field_io.cpp`, `xdmf.cpp`, `scalar_wave.cpp`, `CMakeLists.txt`
  - File deleted: `empty_integrator.cpp`
  - Test: `ctest --test-dir build` — all pass (4 pre-existing failures unrelated)

### 18.6 — Final verification

- [x] **18.6a** Full build and test:
  - `cmake --build build` — clean build, no errors.
  - `ctest --test-dir build` — 40/44 passed, 4 pre-existing failures (t-object_geometry, t-csr, t-E2_1, t-laplacian) — all unrelated to phase 18 changes.
  - Net line reduction: ~1825 lines removed in phase 18 (42 files changed, 501 insertions, 2326 deletions). Exceeds ~800 target.
  - No remaining commented-out code blocks > 3 lines found.
  - Phase 18 complete.

---

## Ordering

```
18.1a-fixup (graph slot mismatch — do before other work that touches euler or simulation_cycle)
18.1b-c (remaining bugs) — independent, do first
18.2a (shared eval_at_locations) → 18.2b-d (shared stats/init/write) → 18.2e (neumann fix)
18.3a-c (fields dead code) — independent of 18.2
18.4a-b (file deletion, commented-out code) — independent
18.5a-f (architecture) — independent
18.6a (verify) — last
```

18.2a must come before 18.2b-d since the shared header must exist first. All other groups are independent.

---

## Completion Criteria

- Zero duplicated method bodies between heat.cpp and scalar_wave.cpp.
- `MinMax` reduction guarded for empty fluid domains.
- Euler uses the Graph API path.
- ~800 lines of dead code removed.
- `handle.hpp` reduced from ~613 to ~270 lines.
- `expr.hpp` reduced by ~210 lines.
- Dead files deleted (cc_elliptic, sentinels, extents, discrete_operator, divergence, mesh_view).
- Block kernel functor shared between `operator()` and `graph_node`.
- All tests pass.

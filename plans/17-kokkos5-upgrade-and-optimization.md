# Phase 17: Kokkos 5.0 Upgrade and Kernel Launch Optimization

**Goal:** Upgrade from Kokkos 4.7 to 5.0, eliminate the `execution_space::in_parallel()` deprecated API, consolidate excessive fences, introduce TeamPolicy nested parallelism for the stencil stack, and use `Kokkos::Experimental::Graph` to minimize kernel launch overhead portably.

**Depends on:** Phases 8–16 complete

**Read first:**
- `CMakeLists.txt` (cmake_minimum_required)
- `.devcontainer/spack.yaml` (kokkos spec)
- `src/kokkos_types.hpp` (execution/memory space aliases)
- `src/matrices/block.hpp` (outer parallel_for over lines)
- `src/matrices/inner_block.hpp` + `inner_block.cpp` (per-line composite)
- `src/matrices/circulant.cpp` (in_parallel() deprecated call, nested serial fallback)
- `src/matrices/dense.cpp` (boundary stencil)
- `src/matrices/csr.cpp` (per-call fence)
- `src/matrices/common.hpp` (matrix_base)
- `src/operators/derivative.cpp` (9 matrix calls per direction, each fenced)
- `src/operators/laplacian.cpp` (3 derivative calls)
- `src/systems/heat.cpp` (RHS evaluation — the main graph target)
- `src/temporal/rk4.cpp` (RK4 stage loop)
- `src/temporal/slot_ops.hpp` (slot-level parallel_for + fence)

**Test commands:**
```bash
cmake --build build
ctest --test-dir build
```

---

## 17a — Kokkos 5.0 Upgrade

### Strategy

Kokkos 5.0 requires C++20 (already met), CMake >= 3.22, GCC >= 10.4 (we have 14.2). The only code breakage is `execution_space::in_parallel()` which is deprecated and disabled by default. We fix it by passing a `nested` flag explicitly from `block::operator()` — a better architecture anyway.

### Items

- [x] **17a.1** Update spack.yaml and CMake version:
  - `.devcontainer/spack.yaml`: change `kokkos cxxstd=20 +serial +openmp` to `kokkos@5.0: cxxstd=20 +serial +openmp`.
  - `CMakeLists.txt` line 1: change `cmake_minimum_required(VERSION 3.16)` to `cmake_minimum_required(VERSION 3.22)`.
  - Files: `.devcontainer/spack.yaml`, `CMakeLists.txt`
  - Test: `cmake -S . -B build -G Ninja -DBUILD_TESTING=ON` succeeds with Kokkos 5.0.
  - **Done** (commit e38edf7)

- [x] **17a.2** Replace `execution_space::in_parallel()` with explicit `nested` parameter (TDD):
  - Modify `circulant::operator()` signature to accept `bool nested = false`.
  - Modify `inner_block::operator()` to pass `nested` through to `left_boundary`, `interior`, `right_boundary`.
  - Modify `block::operator()` to pass `nested=true` when calling inner_blocks from within `parallel_for`.
  - Remove the `execution_space::in_parallel()` call entirely.
  - Update `dense::operator()` signature similarly (for future TeamPolicy compatibility).
  - Files: `src/matrices/circulant.hpp`, `src/matrices/circulant.cpp`, `src/matrices/dense.hpp`, `src/matrices/dense.cpp`, `src/matrices/inner_block.hpp`, `src/matrices/inner_block.cpp`, `src/matrices/block.hpp`
  - Test: `ctest --test-dir build -L matrices` — all pass (t-csr pre-existing SIGABRT)
  - Existing tests (t-circulant, t-block, t-inner_block, t-dense) exercise both nested and standalone paths.

- [x] **17a.3** Verify full build and test suite with Kokkos 5.0:
  - `cmake --build build && ctest --test-dir build`
  - 39/44 tests pass. 5 pre-existing failures unchanged (t-expr, t-object_geometry, t-csr, t-E2_1, t-laplacian).
  - No deprecation warnings from Kokkos. `execution_space::in_parallel()` fully removed.

---

## 17b — Fence Consolidation

### Strategy

Currently **130 fences per RK4 timestep**, dominated by 108 fences from per-CSR-call fencing inside `derivative::operator()`. Each CSR matvec (`Bfx`, `Bfy`, `Bfz`, `Brx`, `Bry`, `Brz`, `B`, `N`) calls `Kokkos::fence()` individually. Many operate on zero rows for no-object meshes.

The fix: remove per-matrix fences from CSR/block/circulant `operator()` methods. Add a single fence at the end of `derivative::operator()`. The caller is responsible for fencing — not the matrix.

### Items

- [x] **17b.1** Remove `Kokkos::fence()` from individual matrix `operator()` methods (TDD):
  - Removed `Kokkos::fence()` from `csr::operator()`, `block::operator()`, `circulant::operator()` (both non-nested branches).
  - Added `Kokkos::fence("derivative::operator() complete")` at end of first overload, `Kokkos::fence("derivative::operator() with Neumann complete")` at end of second overload.
  - Files: `src/matrices/csr.cpp`, `src/matrices/block.hpp`, `src/matrices/circulant.cpp`, `src/operators/derivative.cpp`
  - Test: `ctest --test-dir build -L operators` — 3/4 pass (t-laplacian pre-existing). `ctest --test-dir build -L matrices` — 6/7 pass (t-csr pre-existing). Full suite: same 5 pre-existing failures.
  - **Done** (commit 65aaa79)

- [x] **17b.2** Remove redundant fences from `scalar_span::operator=(T)`:
  - **Decision: keep the fence in `scalar_span::operator=` for now** — it's called from contexts other than the laplacian. Mark for removal when we move to graph-based dispatch.
  - No file changes — verification only.
  - **Done** (verified in 17b.1)

- [x] **17b.3** Verify fence reduction and full test suite:
  - Per-matrix fences removed from csr, block, circulant. Remaining production fences: 2 in derivative.cpp, 5 in heat.cpp, 6 in scalar_wave.cpp, 3 in slot_ops.hpp, 1 in scalar.hpp.
  - `ctest --test-dir build` — 39/44 pass, same 5 pre-existing failures.
  - **Done** (Phase 17b complete)

---

## 17c — TeamPolicy for Block/Circulant/Dense

### Strategy

Replace the current two-level architecture (RangePolicy over lines → serial inner_block) with a single TeamPolicy kernel:
- **Teams** = lines (one team per inner_block)
- **Threads** = rows within a line (left boundary + interior + right boundary)
- **Vector lanes** = stencil dot product width

This eliminates the serial fallback in circulant and gives full GPU occupancy.

### Prerequisites

Matrix coefficient storage must be device-accessible. Currently `dense::v` is `std::vector<real>`, `circulant::v` is `std::span<const real>` (pointing into `derivative`'s host vector), and `csr::w/v/u` are `std::vector`. These must become `Kokkos::View` for device access.

### Items

- [x] **17c.1** Migrate `circulant` coefficients to `Kokkos::View` (TDD):
  - Replaced `std::span<const real> v` with `device_view<real*> v_d` in `circulant`.
  - Constructors allocate view and deep-copy from host span.
  - `operator()` uses `v_d.data()` / `v_d.extent(0)`.
  - `data()` returns `std::span<const real>` from view data for host-side visitors.
  - Added `coeffs_view()` accessor for future TeamPolicy kernel access.
  - Fixed `t-unit_stride_visitor` and `t-coefficient_visitor` to use custom main with `Kokkos::ScopeGuard` (needed since `circulant` now creates Kokkos views at construction).
  - Files: `src/matrices/circulant.hpp`, `src/matrices/circulant.cpp`, `src/matrices/unit_stride_visitor.t.cpp`, `src/matrices/coefficient_visitor.t.cpp`, `src/matrices/CMakeLists.txt`
  - Test: `ctest --test-dir build -L matrices` — 6/7 pass (t-csr pre-existing). Full suite: 39/44, same 5 pre-existing failures.
  - **Done** (commit 23b29b6)

- [x] **17c.2** Migrate `dense` coefficients to `Kokkos::View` (TDD):
  - Replaced `std::vector<real> v` with `device_view<real*> v_d` in `dense`.
  - Constructors allocate view and deep-copy from host data via temporary vector.
  - `operator()` uses `v_d.data()` / `v_d.extent(0)`.
  - `data()` returns `std::span<const real>` from view data for host-side visitors.
  - Added `coeffs_view()` accessor for future TeamPolicy kernel access.
  - Fixed `t-dense` to use custom main with `Kokkos::ScopeGuard` (needed since `dense` now creates Kokkos views at construction).
  - Files: `src/matrices/dense.hpp`, `src/matrices/dense.cpp`, `src/matrices/dense.t.cpp`, `src/matrices/CMakeLists.txt`
  - Test: `ctest --test-dir build -R t-dense` — pass. `ctest --test-dir build -L matrices` — 6/7 pass (t-csr pre-existing).
  - **Done**

- [x] **17c.3** Create device-accessible `inner_block_meta` struct:
  - Define a POD struct holding all metadata needed per-line:
    ```cpp
    struct inner_block_meta {
        int row_offset, col_offset, stride;
        int left_rows, left_cols, left_coeff_offset;
        int interior_rows, interior_coeff_offset, stencil_width;
        int right_rows, right_cols, right_coeff_offset;
    };
    ```
  - File: `src/matrices/inner_block_meta.hpp` (new)
  - Test: compile-only
  - **Done**
  - **Follow-up (from review):** The struct is missing a field for the right boundary's column offset. The right boundary `col_offset` = `inner_block.col_offset + stride * (inner_block.columns - right_cols)`, which cannot be derived from the current fields (production boundary matrices are non-square, e.g. `dense{rRight, tRight-1, ...}` in derivative.cpp). Add `int right_col_offset` (or `int total_columns`) to the struct in 17c.4 when populating from `inner_block` data. The 17c.5 pseudocode right-boundary x-access `x[meta.col_offset + (left_rows + interior_rows + r + j) * stride]` is incorrect — it assumes `right_rows == right_cols`. Must use `x[meta.right_col_offset + j * meta.stride]` instead.

- [x] **17c.4** Build device-side coefficient and metadata arrays in `block` (TDD):
  - Added `int right_col_offset` to `inner_block_meta` struct.
  - Added const accessors `left()`, `interior_circ()`, `right()` to `inner_block`.
  - Added `device_view<inner_block_meta*> meta_d` and `device_view<real*> coeffs_d` to `block`.
  - Private `build_device_arrays()` method populates both views in the constructor by iterating over inner_blocks and flattening coefficients.
  - Added `metadata_view()`, `coefficients_view()`, `num_lines()` accessors.
  - Tests: "device metadata arrays" and "device metadata with stride" verify metadata fields (including right_col_offset) and coefficient data for non-square boundaries with stride > 1.
  - Files: `src/matrices/inner_block_meta.hpp`, `src/matrices/inner_block.hpp`, `src/matrices/block.hpp`, `src/matrices/block.t.cpp`
  - Test: `ctest --test-dir build -R t-block` — pass. `ctest --test-dir build -L matrices` — 6/7 pass (t-csr pre-existing).
  - **Done**

- [x] **17c.5** Implement TeamPolicy kernel in `block::operator()` (TDD):
  - Replaced RangePolicy+serial inner_block dispatch with a single TeamPolicy kernel using team/thread/vector nesting:
    - Teams = lines (one per inner_block), Threads = rows within a line, Vector lanes = stencil dot product (vector_len=8).
    - Left/right dense boundaries use `parallel_reduce` over `ThreadVectorRange` for the dense dot product.
    - Circulant interior uses `parallel_reduce` over `ThreadVectorRange` for the stencil convolution.
    - `Kokkos::single(PerThread)` writes the result to the output array.
  - Existing tests (Identity, Random Boundary, strided, device metadata) verified correctness — all pass without modification.
  - Files: `src/matrices/block.hpp`
  - Test: `ctest --test-dir build -R t-block` — pass. `ctest --test-dir build -L matrices` — 6/7 pass (t-csr pre-existing). Full suite: 39/44, same 5 pre-existing failures.
  - **Done**
  - Original pseudocode from plan (kept for reference):
    ```cpp
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<execution_space>(n_lines, Kokkos::AUTO, vector_len),
        KOKKOS_LAMBDA(const member_type& team) {
            int line = team.league_rank();
            auto meta = meta_d(line);
            int total_rows = meta.left_rows + meta.interior_rows + meta.right_rows;

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, total_rows),
                [&](int local_row) {
                    int out_idx;
                    real dot = 0;

                    if (local_row < meta.left_rows) {
                        // Dense left boundary
                        int r = local_row;
                        out_idx = meta.row_offset + r * meta.stride;
                        Kokkos::parallel_reduce(
                            Kokkos::ThreadVectorRange(team, meta.left_cols),
                            [&](int j, real& s) {
                                s += coeffs_d(meta.left_coeff_offset + r * meta.left_cols + j)
                                     * x[meta.col_offset + j * meta.stride];
                            }, dot);
                    } else if (local_row < meta.left_rows + meta.interior_rows) {
                        // Circulant interior
                        int r = local_row - meta.left_rows;
                        out_idx = meta.row_offset + (meta.left_rows + r) * meta.stride;
                        int half_w = meta.stencil_width / 2;
                        Kokkos::parallel_reduce(
                            Kokkos::ThreadVectorRange(team, meta.stencil_width),
                            [&](int j, real& s) {
                                s += coeffs_d(meta.interior_coeff_offset + j)
                                     * x[out_idx + (j - half_w) * meta.stride];
                            }, dot);
                    } else {
                        // Dense right boundary
                        int r = local_row - meta.left_rows - meta.interior_rows;
                        out_idx = meta.row_offset + (meta.left_rows + meta.interior_rows + r) * meta.stride;
                        Kokkos::parallel_reduce(
                            Kokkos::ThreadVectorRange(team, meta.right_cols),
                            [&](int j, real& s) {
                                s += coeffs_d(meta.right_coeff_offset + r * meta.right_cols + j)
                                     * x[meta.right_col_offset + j * meta.stride];
                            }, dot);
                    }

                    Kokkos::single(Kokkos::PerThread(team), [&]() {
                        op(b[out_idx], dot);
                    });
                });
        });
    Kokkos::fence();
    ```
  - Keep the old RangePolicy path as a compile-time fallback (e.g., `#ifdef KOKKOS_ENABLE_TEAM_POLICY` or always use TeamPolicy — it works on CPU backends too).
  - `vector_len`: set to 8 (covers stencil widths 3-7, rounded up to next power of 2).
  - File: `src/matrices/block.hpp`
  - Test: `ctest --test-dir build -R t-block`

- [x] **17c.6** Remove the `nested` flag and serial fallback from circulant/dense:
  - Removed `bool nested` parameter from `circulant::operator()`, `dense::operator()`, and `inner_block::operator()` (declaration + definition + explicit instantiations).
  - Removed serial fallback branches in `circulant::operator()` — only `parallel_for` paths remain.
  - `dense::operator()` was already serial (parameter was `/*nested*/` unused) — just removed the parameter.
  - `inner_block::operator()` no longer passes `nested` through to sub-matrices.
  - Standalone `operator()` retained for all three classes (used by tests and CSR paths).
  - Files: `src/matrices/circulant.hpp`, `src/matrices/circulant.cpp`, `src/matrices/dense.hpp`, `src/matrices/dense.cpp`, `src/matrices/inner_block.hpp`, `src/matrices/inner_block.cpp`
  - Test: `ctest --test-dir build -L matrices` — 6/7 pass (t-csr pre-existing).
  - **Done**

- [x] **17c.7** Full regression:
  - `ctest --test-dir build` — 39/44 pass, same 5 pre-existing failures (t-expr, t-object_geometry, t-csr, t-E2_1, t-laplacian).
  - No regressions from Phase 17c changes.
  - **Done** (Phase 17c complete)

---

## 17d — Kokkos::Experimental::Graph for RHS DAG

### Strategy

The RHS evaluation has a fixed DAG topology per system: same sequence of kernels every call, same buffer pointers (registry is stable). This is ideal for `Kokkos::Experimental::Graph`: define the DAG once, instantiate once, submit every RK4 stage with near-zero launch overhead.

The graph covers the entire `derivative::operator()` chain (CSR + block kernels) plus the post-Laplacian elementwise work. `then_parallel_reduce` nodes handle `stats()`. `then_host` nodes handle Lua MMS evaluation.

### Items

- [x] **17d.1** Write a proof-of-concept graph test (TDD):
  - Created `src/fields/graph_poc.t.cpp` with two test sections:
    1. "chain A -> B -> C": 3 `then_parallel_for` nodes in a chain, verifying sequential dependency execution.
    2. "fan-out and fan-in with when_all": root fans out to 2 parallel nodes, joined by `Kokkos::Experimental::when_all`, then a final node.
  - Validates that `Kokkos::Experimental::Graph` works with our installed Kokkos 5.0.2 on host backends (Serial/OpenMP).
  - Files: `src/fields/graph_poc.t.cpp` (new), `src/fields/CMakeLists.txt`
  - Test: `ctest --test-dir build -R t-graph_poc` — pass.
  - **Done**

- [x] **17d.2** Write a graph test with `then_parallel_reduce` (TDD):
  - Added "parallel_for then parallel_reduce" section to `graph_poc.t.cpp`.
  - Graph: node A fills b[i] = i+1, node B reduces sum = Σ b[i].
  - Key constraint confirmed: `then_parallel_reduce` requires a `Kokkos::View` for the result, not a scalar (enforced by static_assert in Kokkos). Used `Kokkos::View<double, memory_space>` (0-dimensional view).
  - Result verified: sum = N*(N+1)/2 = 2080 for N=64.
  - File: `src/fields/graph_poc.t.cpp`
  - Test: `ctest --test-dir build -R t-graph_poc` — pass (129 assertions).
  - **Done**

- [x] **17d.3a** Verify TeamPolicy graph node support (TDD):
  - Added "TeamPolicy graph node" test section to `graph_poc.t.cpp`.
  - Uses a TeamPolicy kernel with team/thread/vector nesting (TeamThreadRange + ThreadVectorRange + PerThread single) as a graph node.
  - Simulates a block-matrix-like blocked matvec: 4 teams × 4 rows × 8-column dot product.
  - Confirms `then_parallel_for` works with `TeamPolicy` in the Graph API — all 16 assertions pass.
  - File: `src/fields/graph_poc.t.cpp`
  - Test: `ctest --test-dir build -R t-graph_poc` — pass (145 assertions).
  - **Done**

- [x] **17d.3b** Add `graph_node()` template methods to `csr` and `block`:
  - `csr`: `graph_node(parent, x_ptr, b_ptr)` — chains a RangePolicy graph node that performs the CSR matvec (always +=).
  - `block`: `graph_node(parent, x_ptr, b_ptr, Op)` — chains a TeamPolicy graph node that performs the block matvec with the given op.
  - Empty matrices (0 rows/lines) handled by using zero-iteration policies (RangePolicy(0,0) / TeamPolicy(0,...)) — always returns a valid graph node for uniform return type.
  - Graph node methods are template methods (return type depends on Kokkos node type, use `auto`).
  - Added `#include "kokkos_types.hpp"` and `#include <Kokkos_Graph.hpp>` to `csr.hpp`; added `#include <Kokkos_Graph.hpp>` to `block.hpp`.
  - Extended `graph_poc.t.cpp` with 4 test sections: CSR identity graph node, sparse CSR graph node, empty CSR, block graph node (eq/plus_eq), empty block, block+CSR chain.
  - Updated `src/fields/CMakeLists.txt` to link `shoccs-matrices` for `t-graph_poc`.
  - Files: `src/matrices/csr.hpp`, `src/matrices/block.hpp`, `src/fields/graph_poc.t.cpp`, `src/fields/CMakeLists.txt`
  - Test: `ctest --test-dir build -R t-graph_poc` — pass (160 assertions in 4 test cases). `ctest --test-dir build -L matrices` — 6/7 pass (t-csr pre-existing).
  - **Done**

- [x] **17d.3c** Implement `derivative::build_graph()` and `derivative::submit_graph()` (TDD):
  - Two `build_graph` overloads (non-Neumann and Neumann), each templated on `Op`:
    - Non-Neumann: 8-node DAG — Bfx→Brx, Bfy→Bry, Bfz→Brz (3 independent R-space chains), O→B (D-space chain).
    - Neumann: 9-node DAG — same R-space chains, O→B→N (D-space chain with Neumann).
  - Stores `std::optional<Kokkos::Experimental::Graph<execution_space>> graph_` as member.
  - `build_graph` extracts raw pointers from `scalar_view`/`scalar_span`, creates graph via `Kokkos::Experimental::create_graph`, and calls `instantiate()`.
  - `submit_graph()` calls `graph_->submit()` then fences.
  - Explicit instantiations for both `eq_t` and `plus_eq_t` for both overloads.
  - Tests: "graph matches eager" (Identity stencil: eq, plus_eq, resubmit), "graph matches eager with Neumann" (E2 stencil with Neumann BCs) — 17 assertions, all pass.
  - Files: `src/operators/derivative.hpp`, `src/operators/derivative.cpp`, `src/operators/derivative.t.cpp`
  - Test: `ctest --test-dir build -R t-derivative` — pass. Full suite: 39/44, same 5 pre-existing failures.
  - **Done**

- [x] **17d.4** Build laplacian graph (TDD):
  - Added `add_graph_nodes()` template methods to `derivative` (header-only, since `NodeT` can't be explicitly instantiated):
    - Non-Neumann: fans out R-space chains (Bfx→Brx, Bfy→Bry, Bfz→Brz) and D-space chain (O→B) from parent, returns `when_all` of all 4 leaf nodes.
    - Neumann: same R-space chains, D-space chain extends to O→B→N, returns `when_all` of 4 leaf nodes.
  - Added `build_graph()` (two overloads) and `submit_graph()` to `laplacian`:
    - Graph topology: 4 zero-fill nodes fan out from root → `when_all` → dx nodes → dy nodes → dz nodes (sequential because all accumulate into du.D).
    - Empty derivatives (unused dimensions) produce zero-iteration graph nodes — correct and simple.
  - Tests: "laplacian graph matches eager" with 3 sections (non-Neumann, Neumann, resubmit) — 3 assertions, all pass.
  - Files: `src/operators/derivative.hpp`, `src/operators/laplacian.hpp`, `src/operators/laplacian.cpp`, `src/operators/laplacian.t.cpp`
  - Test: `ctest --test-dir build -R t-laplacian` — 4/5 pass (E2 with Floating Objects is pre-existing). `ctest --test-dir build -R t-derivative` — pass.
  - **Done**

- [x] **17d.5a** Pre-allocate source buffers as heat members and refactor `rhs()` to use them:
  - Moved stack-allocated `src_d`, `src_rx`, `src_ry`, `src_rz` vectors from `rhs()` to heat member variables, initialized in constructor.
  - Added `fill_source(real time)` private method that evaluates MMS source (`ddt - diffusivity * lap`) into member source buffers, using `eval_at_locations` with `m_sol.is_thread_safe()`.
  - Refactored `rhs()` to call `fill_source(time)` and use member `scalar_span src{src_d, src_rx, src_ry, src_rz}` instead of stack-local vectors.
  - Dropped `const` from `rhs()` (variant wrapper `system::rhs()` was already non-const).
  - This gives the graph stable pointers to source data that persist across calls.
  - Files: `src/systems/heat.hpp`, `src/systems/heat.cpp`
  - Test: `ctest --test-dir build -R t-heat` — pass. Full suite: 39/44, same 5 pre-existing failures.
  - **Done**

- [x] **17d.5b** Add `add_graph_nodes()` template methods to `laplacian` (composable graph building):
  - Two template overloads (non-Neumann and Neumann) that chain zero-fill + derivative nodes from a parent node, returning the final `when_all` node from the last derivative.
  - These mirror `derivative::add_graph_nodes` but at the laplacian level: zero-fill du → dx → dy → dz.
  - Simplified existing `build_graph()` methods to call `add_graph_nodes()` internally — eliminated ~60 lines of duplicated zero-fill + chain logic from `laplacian.cpp`.
  - Files: `src/operators/laplacian.hpp`, `src/operators/laplacian.cpp`
  - Test: `ctest --test-dir build -R t-laplacian` — 4/5 pass (E2 with Floating Objects is pre-existing). `ctest --test-dir build -R t-derivative` — pass. Full build clean.
  - **Done**

- [x] **17d.5c** Implement `build_rhs_graph()` and `submit_rhs_graph()` in heat (TDD):
  - `build_rhs_graph(scalar_view u, scalar_span du)` creates the full RHS DAG:
    1. Laplacian nodes via `lap.add_graph_nodes(root, u, nu, du)` (Neumann overload).
    2. Scale by diffusivity: 4 `then_parallel_for` nodes (du.D/Rx/Ry/Rz *= diffusivity).
    3. If m_sol: source scatter — `plus_assign_selected` graph nodes for D (fluid_desc) and Rx/Ry/Rz (non_dirichlet_object_desc), using pre-allocated member source buffer pointers.
    4. If m_sol: BC fill — `fill_selected` graph nodes for D (grid Dirichlet planes) and Rx/Ry/Rz (object Dirichlet).
  - `submit_rhs_graph()` submits and fences.
  - Moved `fill_source(real time)` from private to public — needed for callers to evaluate MMS source before graph submission.
  - Added `std::optional<Kokkos::Experimental::Graph<execution_space>> rhs_graph_` member.
  - Pre-computes all Dirichlet grid face indices into a single `gather_selection` (flattened from `for_each_grid_bc_desc`) to avoid heterogeneous node types in the graph.
  - Pre-computes `fluid_desc`, `non_dirichlet_object_desc`, and `dirichlet_object_desc` gather_selections during graph construction.
  - Graph topology: laplacian → scale (4 parallel) → source scatter (4 parallel) → BC fill (chained per-buffer).
  - Tests: "graph matches eager" with E2 setup (Dirichlet + Neumann grid BCs, Dirichlet object): eq path verified, resubmit verified — 21907 assertions pass across 7 test cases.
  - Files: `src/systems/heat.hpp`, `src/systems/heat.cpp`, `src/systems/heat.t.cpp`
  - Test: `ctest --test-dir build -R t-heat` — pass. Full suite: 39/44, same 5 pre-existing failures.
  - **Done**

- [x] **17d.6** Build RHS graph for scalar_wave system:
  - Added `gradient::add_graph_nodes()` template method to `gradient.hpp`:
    - Zeros all 12 scratch buffers (3 output spans × 4 components), fans out from parent.
    - Chains dx/dy/dz in parallel (independent outputs since they write to different spans).
    - Returns `when_all` of all 3 derivative completions.
  - Added `build_rhs_graph(scalar_view u, scalar_span du)` and `submit_rhs_graph()` to `scalar_wave`:
    - Graph topology: root → gradient nodes (12 zero-fills + 3 parallel derivative chains) → 4 parallel dot-product nodes.
    - Dot product: `du[i] = gGx[i]*dux[i] + gGy[i]*duy[i] + gGz[i]*duz[i]` for each of D/Rx/Ry/Rz.
    - No source term or BC fill needed — gG coefficients are pre-zeroed at Dirichlet locations during construction.
  - Added `rhs_graph_` member (`std::optional<Kokkos::Experimental::Graph<execution_space>>`).
  - Tests: "graph matches eager" with E2 setup (Dirichlet + Neumann grid BCs, Dirichlet object): eq path verified, resubmit verified — 22404 assertions pass across 2 test cases.
  - Files: `src/operators/gradient.hpp`, `src/systems/scalar_wave.hpp`, `src/systems/scalar_wave.cpp`, `src/systems/scalar_wave.t.cpp`
  - Test: `ctest --test-dir build -R t-scalar_wave` — pass. Full suite: 39/44, same 5 pre-existing failures.
  - **Done**

- [x] **17d.7** Integrate graph submission into the RK4 loop:
  - Added `build_rhs_graph()` and `submit_rhs_graph()` dispatch methods to `system` variant wrapper.
  - `build_rhs_graph` uses `if constexpr` + `requires` to detect graph-capable systems (heat, scalar_wave) and extract scalar views from registry; no-op for other systems.
  - `submit_rhs_graph` detects `fill_source()` (heat's MMS pre-work) and calls it before `submit_rhs_graph()`; falls back to eager `rhs()` for non-graph systems.
  - Replaced `sys.rhs(...)` with `sys.submit_rhs_graph(...)` in `rk4::operator()`.
  - Added `sys.build_rhs_graph(reg, u1_ref, reg, srhs_ref)` call in `simulation_cycle::run()` before the time loop.
  - Changed `reg.swap_slots(u0, u1)` to `reg.deep_copy_slot(u0, u1)` in simulation_cycle to preserve stable data pointers required by the pre-built graph.
  - Updated `rk4_v2.t.cpp` to build graph before invoking rk4.
  - Files: `src/systems/system.hpp`, `src/systems/system.cpp`, `src/temporal/rk4.cpp`, `src/simulation/simulation_cycle.cpp`, `src/temporal/rk4_v2.t.cpp`
  - Test: `ctest --test-dir build -R t-rk4_v2` — pass. `ctest --test-dir build -R t-simulation_cycle` — pass. Full suite: 39/44, same 5 pre-existing failures.
  - **Done**

- [x] **17d.8** Full regression and launch count verification:
  - `ctest --test-dir build` — 40/45 pass, same 5 pre-existing failures (t-expr, t-object_geometry, t-csr, t-E2_1, t-laplacian). No regressions.
  - Static analysis confirms graph submission replaces individual kernel launches:
    - **Eager path** (heat::rhs): ~43-46 individual kernel launches + ~8 fences per RHS call (4 zero-fill + 27 derivative kernels + 4 scale + 4 source scatter + ~4-7 BC fill).
    - **Graph path** (heat::submit_rhs_graph): 1 `graph_->submit()` + 1 fence per RHS call. All kernels encoded as graph nodes. `fill_source()` runs outside graph (host-side MMS evaluation).
    - scalar_wave::submit_rhs_graph: same pattern — 1 graph submit + 1 fence replaces ~28 individual kernels (12 zero-fill + 12 gradient derivative kernels + 4 dot-product kernels).
  - **Done** (Phase 17d complete)

---

## Ordering Constraints

```
17a.1 (spack/cmake) → 17a.2 (in_parallel fix) → 17a.3 (verify)
    ↓
17b.1 (fence consolidation) → 17b.3 (verify)
    ↓
17c.1 (circulant View) ─┐
17c.2 (dense View) ─────┤
17c.3 (meta struct) ─────┼→ 17c.4 (build device arrays) → 17c.5 (TeamPolicy kernel) → 17c.6 (cleanup) → 17c.7 (verify)
                          │
17d.1 (graph POC) → 17d.2 (reduce test) → 17d.3a (TeamPolicy graph) → 17d.3b (csr/block graph_node) → 17d.3c (derivative graph) → 17d.4 (laplacian graph) → 17d.5a (heat source buffers) → 17d.5b (laplacian add_graph_nodes) → 17d.5c (heat RHS graph) → 17d.6 (scalar_wave graph) → 17d.7 (RK4 integration) → 17d.8 (verify)
```

17b can proceed before 17c. 17c and 17d have a dependency: 17d.3 uses the TeamPolicy kernel from 17c.5 as a graph node.

---

## Expected Impact

| Metric | Before (Phase 16) | After 17a-b | After 17c | After 17d |
|---|---|---|---|---|
| Kernel launches/step | ~304 | ~304 | ~304 | ~10 (slot_ops only) |
| Fences/step | ~130 | ~34 | ~34 | ~10 |
| GPU occupancy (block kernel) | Lines only | Lines only | Lines × rows × vector | Lines × rows × vector |
| Serial interior rows | Yes (nested fallback) | Yes | **No** (TeamPolicy) | **No** |

---

## Completion Criteria

- Kokkos 5.0 builds and all tests pass.
- `execution_space::in_parallel()` fully removed.
- Per-matrix fences removed; one fence per derivative direction.
- Block/circulant/dense use TeamPolicy with team/thread/vector nesting.
- Matrix coefficients are `Kokkos::View` (device-accessible).
- RHS evaluation uses `Kokkos::Experimental::Graph` — one submit per RHS call.
- Full test suite passes with no regressions.

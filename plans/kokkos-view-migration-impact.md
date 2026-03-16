# Kokkos::View Storage Migration — Impact Analysis

**Purpose:** Comprehensive analysis of the impact of migrating field data storage from `std::vector<real>` to `Kokkos::View` and the implications for the DSL, selectors, matrix products, and memory layout.

---

## 1. Current Storage Architecture

The field data lives in a deeply nested tuple structure with `std::vector<real>` at every leaf:

```
field
  s: std::vector<scalar_real>              // runtime-sized list of scalar fields
       scalar_real = tuple<
           tuple<std::vector<real>>,        // D  (domain interior)
           tuple<std::vector<real>,         // Rx (x-ray object intersections)
                 std::vector<real>,         // Ry
                 std::vector<real>>>        // Rz
  v: std::vector<vector_real>              // runtime-sized list of vector fields
       vector_real = tuple<scalar_real,     // x-component (4 vectors)
                           scalar_real,     // y-component (4 vectors)
                           scalar_real>     // z-component (4 vectors)
```

A single scalar field = **4 `std::vector<real>`**. A single vector field = **12 `std::vector<real>`**. The layout is pure **SoA** — components are never interleaved.

The mesh is logically 3D with **row-major (C-order) linearization**: `index = i*ny*nz + j*nz + k` (z fastest). This matches **`Kokkos::LayoutRight`** exactly.

---

## 2. Hard Incompatibilities (Structural Breaks)

These are places where `Kokkos::View` fundamentally differs from `std::vector` and no thin wrapper can bridge the gap:

| Breaking Point | File | Issue |
|---|---|---|
| **`emplace_back` / growth** | `field.hpp:27-42` | `field(system_size)` builds its scalar/vector collections by calling `s.emplace_back(...)` in a loop. `Kokkos::View` is fixed-size after construction. The `requires` guard silently removes this constructor, leaving **no way to build a field from a size spec**. |
| **Iterator-pair construction** | `container_tuple.hpp:27` | `Args{std::ranges::begin(r), std::ranges::end(r)}` — the fundamental construction path for leaf containers. `Kokkos::View` has no `(iterator, iterator)` constructor. |
| **`.resize(n)`** | `tuple_utils.hpp:278-283` | `resize_and_copy` (the central assignment engine) calls `container.resize(n)`. The fallback when resize is unavailable silently truncates data — **unsafe** if the source is larger than a fixed-extent Kokkos view. |
| **`ConstructibleFromRange` concept** | `tuple_fwd.hpp:326-342` | Gates many template overloads. `Kokkos::View` fails this concept entirely, disabling large portions of the type machinery. |

---

## 3. The DSL Pipe Syntax Cannot Survive on Device

The expression `u | m.dirichlet(grid_bcs, object_bcs) = sol` currently works through a chain:

1. **`|` operator** (`tuple_pipe.hpp`) — produces a tuple of non-owning range views into `u`'s backing storage
2. **`= sol`** — calls `resize_and_copy` which does a serial `std::ranges::copy`

This breaks on GPU because:

- **`predicate_view`** uses `std::optional<base_iter_t>` for begin-caching — mutable host state
- **`multi_slice_view`** holds `std::span<const index_slice>` — host pointers, not device-accessible
- **`semiregular_box<Fn>`** wraps lambdas in `std::optional` — not trivially copyable for `KOKKOS_LAMBDA`
- All iterator advancement is **sequential** (serial `operator++` with branching)
- `std::ranges::copy` in `resize_and_copy` is a **serial CPU loop**

The pipe syntax can survive **on the host** with parallel dispatch replacing the inner loops, but on device it needs a fundamentally different execution mechanism.

---

## 4. The Four Selection Patterns and Their Kokkos Equivalents

| Current Pattern | Used For | GPU Replacement |
|---|---|---|
| `plane_view<0>` (x-plane) | Domain x-boundary | `Kokkos::subview(field, i, ALL, ALL)` — contiguous 2D view |
| `plane_view<1>` (y-plane) | Domain y-boundary | `Kokkos::subview(field, ALL, j, ALL)` — strided 2D view |
| `plane_view<2>` (z-plane) | Domain z-boundary | `Kokkos::subview(field, ALL, ALL, k)` — strided 2D view |
| `multi_slice_view` | Fluid interior | Pre-computed `Kokkos::View<int*>` index array (one-time host pass) |
| `predicate_view` | Object BCs (Dirichlet/Neumann) | `Kokkos::parallel_scan` prefix-sum to compact index array |
| `optional_view` | Conditional BC face | `Kokkos::RangePolicy(0, 0)` when inactive |
| `stride_view` | z-plane subselection | `Kokkos::subview` with stride |
| `zip_transform_view` | Element-wise field arithmetic | `Kokkos::parallel_for` with multi-field accessor |
| `cartesian_product_view` | Domain coordinate enumeration | `Kokkos::MDRangePolicy<Rank<3>>` |

---

## 5. Matrix-Vector Products: The Main Parallelism Target

The matrix product stack is already well-structured for Kokkos:

```
block::operator()                    ← serial loop over Ny×Nz lines (PARALLEL TARGET)
  └─ inner_block::operator()         ← per-line: left + interior + right
       ├─ dense::operator()          ← tiny NBS boundary (3×4 rows), uses std::inner_product
       ├─ circulant::operator()      ← interior stencil (N rows), uses std::inner_product
       └─ dense::operator()          ← tiny NBS boundary
```

Key findings:

- **`block::operator()`** iterates over independent lines — each writes to disjoint output rows. This is an embarrassingly parallel outer loop: `Kokkos::parallel_for(blocks.size(), ...)`
- **`circulant::operator()`** is the dominant cost (O(N) rows per line). Natural fit for `TeamPolicy` where each team handles one line
- **Dense NBS matrices** are 3×4 or 4×5 — too small for individual kernels but perfect for batched `KokkosBatlas::Gemv`
- All matrices currently receive `std::span` — the **span-passing API is the migration boundary**. Changing to `Kokkos::View<real*, LayoutStride>` would eliminate the runtime `st != 1` branching entirely (stride becomes part of the view type)

---

## 6. Lazy Evaluation vs. Kokkos Kernels: The Core Design Tension

Currently:

- **Non-mutating ops** (`a + b`, `a * c`) are **lazy** — return `zip_transform_view` expression trees, zero allocation
- **Mutating ops** (`u += v`) are **eager** — serial for-loops in `tuple_math.hpp`
- **Assignment** (`lhs = rhs`) materializes lazy expressions via serial `std::ranges::copy`

For Kokkos, the lazy model has a fundamental problem: expression trees composed of C++ range views cannot be passed into `KOKKOS_LAMBDA` captures (they hold host iterators, `std::optional`-wrapped callables, etc.).

### Design Options

| Option | Description | Disruption | Performance |
|---|---|---|---|
| **A: Host `parallel_for` only** | Replace serial loops in `resize_and_copy` and `tuple_math` with `Kokkos::parallel_for` on `DefaultHostExecutionSpace`. DSL surface unchanged. | Low | CPU-parallel only |
| **B: Expression templates** | Replace `zip_transform_view` with index-callable expression nodes. Assignment `=` emits `parallel_for(N, KOKKOS_LAMBDA(i) { dst(i) = expr(i); })`. | High | Full GPU |
| **C: Dual-mode dispatch** | Thread an execution policy through assignment: `(u \| sel::D).assign(exec, expr)`. Serial for host, `parallel_for` for device. | Medium-High | Flexible |
| **D: Keep ranges + `materialize(exec)`** | Keep lazy views for composition. Add `kokkos_assign(dst, src, exec)` that extracts `.data()` pointers and launches a kernel. Works for contiguous selections; index arrays for non-contiguous. | Medium | Contiguous selections only without index arrays |

---

## 7. The `std::span` Bridge

There's a natural migration seam already in the codebase. The operator layer already consumes `scalar_span` / `scalar_view` (backed by `std::span<real>` / `std::span<const real>`). Since `Kokkos::View<real*, HostSpace>::data()` returns a `real*`:

```cpp
std::span<real>(kokkos_view.data(), kokkos_view.extent(0))
```

This means the **operator/matrix layer can work unchanged** during an initial migration — just bridge `Kokkos::View` to `std::span` at the field extraction point. The span bridge works for host-space views and lets you migrate storage without touching the entire operator stack.

---

## 8. Memory Layout Recommendation

| Option | Matches layout | Selector compat | Kernel readiness | Friction |
|---|---|---|---|---|
| **`View<real*>` per buffer** | Exact 1:1 | Full (wrap in span) | Flat `parallel_for` | **Minimal** |
| `View<real***, LayoutRight>` | Exact (same formula) | Plane = `subview`, replaces hand-rolled iterators | `MDRangePolicy<Rank<3>>` | Medium |
| `View<real***, LayoutLeft>` | Index mismatch | All selectors break | Column-major GPU kernels | High |
| `View<real**>` per vector (3×flat) | Compatible | Requires rewrite | Fused component loops | High |

**Recommendation**: Start with `View<real*>` (1D flat) per leaf buffer. This preserves all existing index arithmetic, lets spans bridge to the operator layer, and avoids touching the tuple/selector machinery. The 3D `View<real***, LayoutRight>` is a second-phase optimization that eliminates `index_extents` and enables natural `subview` slicing — but it requires rewriting the selector views.

---

## 9. Suggested Migration Phases

### Phase A (Low risk): Storage wrapper

Replace leaf `std::vector<real>` with a `field_storage` wrapper that owns a `Kokkos::View<real*, HostSpace>` but exposes `data()`, `size()`, `begin()`, `end()`, and `operator[]`. Add a `resize()` that does `Kokkos::resize()`. Update `container_tuple` construction and `resize_and_copy` to use the new type. Keep all DSL/selector/operator code unchanged via span bridges.

### Phase B (Medium risk): Host-parallel dispatch

Replace serial loops in `tuple_math.hpp` mutating operators and `resize_and_copy` with `Kokkos::parallel_for` on host execution space. Pre-compute index arrays for `multi_slice_view` and `predicate_view` selections at mesh construction time.

### Phase C (High impact): DSL redesign

Migrate to expression templates or index-callable field accessors. Replace `zip_transform_view` composition with a pattern that can emit `Kokkos::parallel_for` kernels. This is where the DSL redesign happens.

### Phase D (GPU): Device execution

Switch `execution_space` to `Kokkos::DefaultExecutionSpace` (CUDA/HIP). Move field data to device memory. Replace `std::span` bridges with device-accessible `Kokkos::View` throughout the operator stack. Replace `block::operator()` serial loop with `TeamPolicy`.

---

## 10. Critical Files by Migration Impact

| Impact | Files |
|---|---|
| **Must change first** | `scalar.hpp`, `vector.hpp` (leaf type aliases), `field.hpp` (construction), `container_tuple.hpp` (range construction), `tuple_utils.hpp` (`resize_and_copy`) |
| **Must adapt** | `tuple_math.hpp` (serial loops → `parallel_for`), `tuple_fwd.hpp` (concepts), `view_tuple.hpp` (`std::views::all` usage) |
| **Second wave** | `selector.hpp` (view types → index arrays), `lazy_views.hpp` (`zip_transform_view` → expression templates), `algorithms.hpp` (`minmax`/`max` → `parallel_reduce`) |
| **Operator layer** | `dense.cpp`, `circulant.cpp`, `csr.cpp` (span → View), `block.hpp` (serial → `parallel_for`), `derivative.cpp` (field extraction) |
| **Unchanged** | `tuple_pipe.hpp`, `selector_fwd.hpp`, `ccs_range_utils.hpp`, `field_math.hpp` (structural plumbing) |

---

## 11. Detailed API Compatibility Matrix

### `std::vector<real>` API usage vs `Kokkos::View<real*>` equivalents

#### Construction

| Pattern | Location | Kokkos Equivalent |
|---|---|---|
| `std::vector<real>(n)` | `lazy_views.hpp:827` | `Kokkos::View<real*>("label", n)` |
| `Vec(begin(r), end(r))` | `container_tuple.hpp:27`, `field.hpp:51-52`, `tuple_utils.hpp:337,340` | **None** — hard incompatibility |
| Copy construction | `tuple.hpp:56`, `view_tuple.hpp:276-279` | `Kokkos::View` copy is shallow (reference-counted) |
| Move construction | `tuple.hpp:64`, `view_tuple.hpp:281-284` | Same as copy (shallow) |

#### Sizing

| Method | Location | Kokkos Equivalent |
|---|---|---|
| `.resize(n)` | `tuple_utils.hpp:278,282` | `Kokkos::resize(view, n)` (free function, not member) |
| `std::ranges::size(c)` | `tuple_utils.hpp`, `field.hpp:90,92`, `lazy_views.hpp` | `view.extent(0)` or `view.size()` |
| `.reserve(n)` | `field.hpp:34-38` | **None** — but already guarded by `requires` |
| `.empty()` | Indirect via ranges | `view.size() == 0` |

#### Element Access

| Method | Location | Kokkos Equivalent |
|---|---|---|
| `operator[](i)` | `field.hpp:106,123` | `view(i)` — uses `()` not `[]` |
| `.data()` | matrices, operators | `view.data()` — exists, same semantics |

#### Iterators

| Method | Location | Kokkos Equivalent |
|---|---|---|
| `std::ranges::begin/end` | Pervasive | Host-only: available. Device: **not available** |
| `.rbegin()` / `.rend()` | Not used in fields | **None** |

#### Modifiers

| Method | Location | Kokkos Equivalent |
|---|---|---|
| `.emplace_back(x)` | `field.hpp:27-28,41-42` | **None** — fixed-size after construction |
| `.push_back()` | Tests only | **None** |

#### Range Concepts

| Concept | Used In | `Kokkos::View` Status |
|---|---|---|
| `std::ranges::input_range` | `tuple_fwd.hpp:164,256` | Host-only: Yes |
| `std::ranges::output_range` | `tuple_fwd.hpp:270-272` | Host-only: Yes (mutable views) |
| `std::ranges::sized_range` | `field.hpp`, `tuple_utils.hpp`, `lazy_views.hpp` | Host-only: Yes |
| `std::ranges::random_access_range` | `field.hpp:103,120`, `selector.hpp:235` | Host-only: Yes |
| `ConstructibleFromRange` | `tuple_fwd.hpp:322-335` | **Fails entirely** |

---

## 12. Boundary Condition Application and Kokkos

### Current BC execution model

BCs are applied as scatter operations to non-contiguous indices:

- Domain face planes in y and z directions (`plane_view<1>` and `plane_view<2>`) are strided/non-contiguous
- Object boundary assignments go through `predicate_view` which skips elements based on a runtime predicate
- The `multi_slice_view` for `fluid` hops between disjoint contiguous segments

### BC fraction of time step

For large 3D problems, BC application is O(N^(2/3)) for domain faces and O(N_intersections) for embedded objects, versus O(N) for the interior Laplacian/gradient. The interior dominates for large meshes.

### Kokkos BC strategy

Replace each selection pattern with a **pre-materialized flat index array** (`Kokkos::View<integer*>`) built once at mesh construction time. The `mesh::dirichlet()` / `mesh::fluid_all()` API would return an opaque index-array descriptor rather than a range view. Assignment becomes `Kokkos::parallel_for` over that index array. This preserves the declarative call-site syntax while replacing the execution mechanism.

`Kokkos::ScatterView` is **not** needed — each target index is written exactly once (no races). A plain `parallel_for` over pre-computed indices suffices.

---

## 13. Complete Time Step Trace (Heat Equation)

For reference, here is how a heat equation time step flows through all the layers:

### `update_boundary(f, time)` — 4 scatter operations

```cpp
u | m.dirichlet(grid_bcs, object_bcs) = l | m_sol(time);  // 9-component scatter
neumann_u | m.neumann<0>(grid_bcs) = l | m_sol.gradient(0, time);
neumann_u | m.neumann<1>(grid_bcs) = l | m_sol.gradient(1, time);
neumann_u | m.neumann<2>(grid_bcs) = l | m_sol.gradient(2, time);
```

### `rhs(f, time, rhs)` — operator application + BC masking

```cpp
u_rhs = lap(u, neumann_u);                            // full Laplacian (matrix products)
u_rhs *= diffusivity;                                  // scalar multiply
u_rhs | m.fluid_all(object_bcs) += src;                // multi_slice + predicate scatter-add
u_rhs | m.dirichlet(grid_bcs, object_bcs) = 0;         // zero Dirichlet points
```

The Laplacian involves the full `block::operator()` → `inner_block` → `dense`/`circulant`/`csr` chain, which is the dominant computational cost.

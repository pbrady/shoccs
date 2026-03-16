# DSL Restructuring Proposal: Flat Storage with Handle-Based Access

**Purpose:** Analyze the feasibility and impact of replacing the deeply nested tuple storage with a flat buffer array indexed by constexpr handles, separating data storage/access from DSL transformation semantics.

---

## 1. The Current Architecture and Its Costs

### 1.1 Nested Tuple Structure

The current field storage uses a deeply nested `ccs::tuple` hierarchy:

```
scalar_real = tuple< tuple<vector<real>>,              // [0] D component
                     tuple<vector<real>,               // [1] R components
                           vector<real>,
                           vector<real>> >

vector_real = tuple< scalar_real,                      // [0] X component
                     scalar_real,                      // [1] Y component
                     scalar_real >                     // [2] Z component
```

Leaf buffers per type: **4 per scalar** (D, Rx, Ry, Rz), **12 per vector** (3 scalars × 4). For a field with `ns` scalars and `nv` vectors: `N = ns×4 + nv×12`.

### 1.2 Multi-Level `get<>` Navigation

Every buffer access requires chaining compile-time `get<I>` calls through nested tuples:

| Selector | `list_index<>` | Expansion | Depth |
|---|---|---|---|
| `si::D` | `<0, 0>` | `get<0>(get<0>(s))` | 2 |
| `si::Rx` | `<1, 0>` | `get<0>(get<1>(s))` | 2 |
| `vi::Dx` | `<0, 0, 0>` | `get<0>(get<0>(get<0>(v)))` | 3 |
| `vi::xRy` | `<0, 1, 1>` | `get<1>(get<1>(get<0>(v)))` | 3 |

While these resolve to zero-cost pointer offsets at compile time, the machinery that supports them is substantial.

### 1.3 The Dual-Inheritance Tax

Every `ccs::tuple<Args...>` (owning variant) dual-inherits from:
- `container_tuple<Args...>` — holds `std::tuple<Args...>` (owns data)
- `view_tuple<Args&...>` — holds `std::tuple<ref_view<Args>...>` (tracks data)

This creates a **re-anchoring obligation**: every copy, move, and assignment must call `view::operator=(*this)` to re-point the view layer at the (potentially relocated) container layer. The codebase has:

- **8 explicit special members** in `tuple.hpp` alone, all requiring re-anchor calls
- **6 `view::operator=(*this)` call sites** across `tuple.hpp` and `view_tuple.hpp`
- **4 placement-new destroy/reconstruct pairs** inside `single_view` (formally UB territory)
- **~310 lines of code** devoted purely to managing the container/view split

A developer comment at `tuple.hpp:54` states: *"component-wise approach is not correct"* — the compiler-generated copy/move would leave `single_view` pointing at the source object. Every future constructor must manually re-anchor or introduce a dangling internal reference with no compiler warning.

### 1.4 Expression Evaluation: Lazy Views

Non-mutating arithmetic (`a + b`, `a * c`) builds `zip_transform_view` trees:

```cpp
auto result = a + b;
// Type: tuple< zip_transform_view<plus, Da, Db>,
//              zip_transform_view<plus, RxA, RxB>,
//              zip_transform_view<plus, RyA, RyB>,
//              zip_transform_view<plus, RzA, RzB> >
```

For `dot(grad_G, du)`, the tree reaches **3 levels deep** — each `operator*()` dereference involves 6+ indirect function invocations through nested iterator state. There is no kernel fusion, no SIMD vectorization, and no parallelism.

Mutating operators (`u += v`) use direct serial for-loops via `tuple_math.hpp`. Materialization of lazy trees happens through `resize_and_copy` → `std::ranges::copy`, which is also a serial CPU loop.

### 1.5 The `selection_view_fn` Dispatch

The `sel::D` object is a `view_closure<selection_view_fn<mp_list<si::D>, mp_list<vi::Dx, vi::Dy, vi::Dz>>>`. When applied via `|`, it:
1. Checks `Scalar<U>` vs `Vector<U>` at template-instantiation time
2. Uses `mp_at<Indices, selection_fn_index<U>>` to pick the right index list
3. Calls `get<list_index<...>>` to navigate the nested tuple
4. Wraps the result in a `selection<L, R, Fn>` that inherits from the extracted range and carries a `semiregular_box<Fn>` for re-targeting via `.apply()`

This works but is opaque: understanding what `sel::D` does to a `Vector` requires tracing through 4 levels of `mp_list` metaprogramming.

---

## 2. The Proposed Design: Flat Storage + Constexpr Handles

### 2.1 Core Idea

Replace the nested tuple hierarchy with:
1. A **flat array of buffers** (the storage)
2. **Constexpr handle structs** (the access mechanism)
3. **Expression nodes** that store handles, not iterator-bearing views

The separation of concerns is:
- **Storage**: A registry of contiguous buffers, indexed by integer
- **Handles**: Compile-time descriptors mapping logical names to buffer indices
- **DSL**: Expression templates that compose handles and evaluate via `parallel_for`

### 2.2 Storage Layer

```cpp
// The registry: owns all field data
struct field_storage {
    std::vector<Kokkos::View<real*>> buffers;  // or std::array<View, N> if N known

    real* data(int id) { return buffers[id].data(); }
    int   size(int id) { return buffers[id].extent(0); }
};
```

For a field with 1 scalar and 1 vector, the registry holds 16 buffers laid out as:

| Index | Semantic | Size |
|---|---|---|
| 0 | `scalar[0].D` | `nx*ny*nz` |
| 1 | `scalar[0].Rx` | `rx_size` |
| 2 | `scalar[0].Ry` | `ry_size` |
| 3 | `scalar[0].Rz` | `rz_size` |
| 4 | `vector[0].x.D` | `nx*ny*nz` |
| 5 | `vector[0].x.Rx` | `rx_size` |
| ... | ... | ... |
| 15 | `vector[0].z.Rz` | `rz_size` |

Buffers of the same size (all domain D buffers) could optionally share a single `Kokkos::View<real**>` allocation for cache coherence, following the Tpetra MultiVector pattern. Different-sized buffers (D vs Rx vs Ry vs Rz) remain separate views.

### 2.3 Handle Types

```cpp
// A single buffer reference
struct handle {
    int id;       // index into registry
    int length;   // element count (redundant with registry, but useful for kernels)

    constexpr bool operator==(const handle&) const = default;  // structural type → NTTP
};

// A scalar field: 4 handles bundled by name
struct scalar_handle {
    handle D;
    handle Rx, Ry, Rz;
};

// A vector field: 3 scalar bundles
struct vector_handle {
    scalar_handle x, y, z;
};
```

These are **plain structs**, trivially copyable, 48–144 bytes. Copy = memcpy. Move = memcpy. No re-anchoring. No `view_tuple`. No `single_view`. No placement-new.

### 2.4 Selectors Become Struct Field Access

```cpp
// sel::D on a scalar — just read the member
constexpr handle sel_D(scalar_handle h) { return h.D; }

// sel::D on a vector — return domain handles of all 3 components
constexpr auto sel_D(vector_handle h) {
    return std::array{h.x.D, h.y.D, h.z.D};
}

// sel::Rx on a scalar
constexpr handle sel_Rx(scalar_handle h) { return h.Rx; }
```

The current `list_index<0,0>` → `get<0>(get<0>(s))` chain becomes `h.D`. The current `list_index<0,1,1>` → `get<1>(get<1>(get<0>(v)))` chain becomes `h.x.Ry`. Same compile-time resolution, dramatically simpler mental model.

### 2.5 Expression Templates Replace `zip_transform_view` Trees

```cpp
template <typename Op, typename Lhs, typename Rhs>
struct binary_expr {
    Op op;
    Lhs lhs;
    Rhs rhs;

    // Index-callable: evaluate at position i
    constexpr real operator()(const field_storage& reg, int i) const {
        return op(lhs(reg, i), rhs(reg, i));
    }
};

// Leaf: a handle dereferences into the registry
struct handle_expr {
    handle h;
    constexpr real operator()(const field_storage& reg, int i) const {
        return reg.data(h.id)[i];
    }
};
```

`a + b` where `a` and `b` are `scalar_handle`s produces:

```cpp
struct scalar_expr {
    binary_expr<plus, handle_expr, handle_expr> D;
    binary_expr<plus, handle_expr, handle_expr> Rx;
    binary_expr<plus, handle_expr, handle_expr> Ry;
    binary_expr<plus, handle_expr, handle_expr> Rz;
};
```

The structure is explicit, the types are shallow (depth 1 regardless of expression complexity), and each sub-expression is independently evaluable.

### 2.6 Materialization via `parallel_for`

```cpp
template <typename Expr>
void assign(field_storage& reg, handle dst, Expr expr) {
    auto* out = reg.data(dst.id);
    Kokkos::parallel_for(dst.length, KOKKOS_LAMBDA(int i) {
        out[i] = expr(reg, i);
    });
}
```

For `dot(grad_G, du)`, which currently produces a 3-level-deep `zip_transform_view` tree:

```cpp
// Handle-based: one fused kernel
auto dot_expr = [&](const field_storage& reg, int i) {
    return gx(reg, i)*dux(reg, i) + gy(reg, i)*duy(reg, i) + gz(reg, i)*duz(reg, i);
};
Kokkos::parallel_for(N, KOKKOS_LAMBDA(int i) {
    result_D[i] = dot_expr(reg, i);
});
```

Six multiplications and two additions in one kernel body. The current system evaluates this as 3 separate view compositions materialized in sequence.

### 2.7 Pipe Syntax Preservation (Optional)

The pipe syntax `u | sel::D = rhs` could be preserved with operator overloading:

```cpp
// u | sel::D returns a "bound handle" that knows where to write
auto operator|(scalar_handle h, sel_D_tag) { return h.D; }

// Assigning to a handle dispatches a parallel_for
template <typename Expr>
void operator=(bound_handle dst, Expr expr) {
    assign(registry, dst.h, expr);
}
```

The pipe chain `u | m.dirichlet(grid_bcs, object_bcs) = sol` would decompose into:
1. `m.dirichlet(...)` returns a list of handles (plane handles, predicate handles)
2. `u | handle_list` selects the appropriate sub-handles from `u`
3. `= sol` dispatches `parallel_for` per handle with the RHS expression

---

## 3. C++20/23 Enablers

The project uses **GCC 14.2.0** with `CMAKE_CXX_STANDARD 20`. GCC 14 fully supports C++23.

### 3.1 C++20 Features (Available Now)

| Feature | Application |
|---|---|
| **Structural NTTPs** | `handle` struct as template parameter: `template<handle H> struct accessor`. Verified zero-overhead — identical assembly to raw array indexing. |
| **`consteval`** | Validate field configurations at compile time: `consteval auto make_scalar_layout(...)`. Guaranteed compile-time evaluation. |
| **`constinit`** | Safe initialization of global handle constants. |
| **`constexpr` `std::vector`** | Build field layouts in `consteval` functions using vector temporaries (freed before constexpr evaluation ends). |

### 3.2 C++23 Features (Available with `-std=c++23`)

| Feature | Application |
|---|---|
| **`static operator()`** | Selector handles use `static operator()` for zero-overhead dispatch — no `this` pointer needed. |
| **Deducing this** | Simplifies CRTP mixins. `tuple_math` and `tuple_pipe` currently use CRTP friend injection; deducing this eliminates the `derived_from` constraints for mutating operators. |
| **`if consteval`** | Single function that branches between compile-time validation and runtime execution. |
| **Multidimensional `operator[]`** | Enables `field[i, j, k]` syntax for 3D access (future GPU phase). |

### 3.3 Structural Type Requirements for NTTPs

A C++20 structural type requires: all members public, literal type, defaulted `operator==`. The `handle` and `scalar_handle` structs satisfy all three. Fixed-size arrays of integers are structural. This means `handle`, `scalar_handle`, and `vector_handle` can all be NTTPs directly.

---

## 4. What Gets Simpler

### 4.1 Copy/Move Semantics

| Aspect | Current | Proposed |
|---|---|---|
| Copy a scalar field reference | Copy `view_tuple` + re-anchor `single_view` via placement-new | Copy 4 integers (16 bytes) |
| Move a scalar field reference | Move `container_tuple` + re-anchor all views | Copy 4 integers (identical to copy) |
| Special members needed | 8 explicit, all with manual re-anchor | 0 explicit (compiler-generated) |
| Lines of re-anchoring code | ~310 | 0 |
| Pointer invalidation bugs | Possible if re-anchor forgotten | Impossible (no pointers in handles) |

### 4.2 Selector Dispatch

| Aspect | Current | Proposed |
|---|---|---|
| `sel::D` type | `view_closure<selection_view_fn<mp_list<si::D>, mp_list<vi::Dx, vi::Dy, vi::Dz>>>` | `constexpr auto sel_D(scalar_handle h) { return h.D; }` |
| Type-level dispatch | `mp_at<Indices, selection_fn_index<U>>` via boost.mp11 | Named struct field access |
| Runtime overhead | Zero (both resolve at compile time) | Zero |
| Cognitive overhead | High (4 levels of mp_list indirection) | Low (struct member access) |

### 4.3 Expression Evaluation

| Aspect | Current | Proposed |
|---|---|---|
| `a + b` result type | `tuple<zip_transform_view<plus, ...>, ...>` (grows with depth) | `scalar_expr<binary_expr<plus, ...>>` (depth always 1) |
| `dot(a, b)` evaluation | 3 separate lazy view compositions, materialized in sequence | 1 fused kernel with 6 FMA ops |
| Compile time | Heavy: nested template instantiation, mp11 metaprogramming | Light: shallow expression templates |
| Parallelism | None (serial iterator loops) | Natural: `parallel_for` at materialization |
| GPU readiness | None (host iterators, `std::optional` in views) | Handles are trivially copyable integers — `KOKKOS_LAMBDA` compatible |

### 4.4 Template Instantiation Reduction

The current system generates distinct types for every expression shape. `zip_transform_view<plus, zip_transform_view<mult, ref_view<vec>, ref_view<vec>>, ref_view<vec>>` is unique per combination of operations and operand types. With expression templates over handles, the expression node types are shallow and reuse the same `binary_expr<Op, Lhs, Rhs>` template with bounded depth.

---

## 5. What Gets Harder or Requires Redesign

### 5.1 Runtime Field Count

The current `field(system_size)` constructor calls `s.emplace_back(...)` in a loop — the number of scalars/vectors is determined at runtime. A flat `std::array<View, N>` requires `N` at compile time.

**Options:**
- **(a)** Make `field` a template: `template<int NS, int NV> class field`. Each system (`heat`, `scalar_wave`) instantiates with its specific counts. The system interface either uses templates or type-erases.
- **(b)** Keep `std::vector<View>` as the runtime container but use handles that carry a runtime offset: `handle{field_id * 4 + component}`. The handle is no longer a pure compile-time value but remains trivially copyable.
- **(c)** Fix a maximum field count and waste unused slots. Simple but wasteful.

**Recommendation:** **(a)** for the near term. Both `heat` and `scalar_wave` use `{1, 0}` (1 scalar, 0 vectors) or `{1, 1}` (1 scalar, 1 vector). The field count is always known at the system-design level. Template instantiation makes the buffer layout a compile-time constant, which enables `std::array`-based storage and `constexpr` handle arithmetic.

### 5.2 Loss of Generic Tuple Machinery

The current `for_each`, `transform`, `resize_and_copy` are generic over any `TupleLike` shape via `index_sequence` expansion. With named structs (`scalar_handle`, `vector_handle`), these become concrete per-type functions.

**Mitigation:** Define a `handle_for_each` that takes a `scalar_handle` or `vector_handle` and visits each sub-handle:

```cpp
template <typename F>
void handle_for_each(F f, scalar_handle h) { f(h.D); f(h.Rx); f(h.Ry); f(h.Rz); }

template <typename F>
void handle_for_each(F f, vector_handle h) {
    handle_for_each(f, h.x); handle_for_each(f, h.y); handle_for_each(f, h.z);
}
```

This is simpler than the current 4-overload concept-dispatched `for_each` in `tuple_utils.hpp`, but less generic. If a new field shape is introduced (e.g., a tensor), a new `handle_for_each` overload must be written.

### 5.3 The `NestedTuple` Structural Recursion

The current system's power comes from `NestedInvocableOver` — a concept that enables recursive descent through arbitrary nesting depths. A flat handle system replaces this with explicit recursion through named types (`vector_handle` → `scalar_handle` → `handle`). The recursion depth is fixed at 2 (vector → scalar → leaf) rather than unbounded.

If unbounded nesting is needed in the future (e.g., tensor fields), either the handle structs must be extended or a more general mechanism (a flat `std::array<handle, N>` with a compile-time shape descriptor) must be introduced.

### 5.4 Non-Contiguous Selections

The four runtime selection patterns translate to the handle system as follows:

| Pattern | Current | Handle System |
|---|---|---|
| **x-plane** | `views::drop(i*ny*nz) \| views::take(ny*nz)` — contiguous | `strided_handle{id, offset=i*ny*nz, stride=1, count=ny*nz}` — contiguous |
| **y-plane** | Custom `plane_view<1>` iterator with `(ny-1)*nz` jumps | `strided_handle{id, offset=j*nz, stride=ny*nz, inner_count=nz, outer_count=nx}` — 2D strided |
| **z-plane** | `views::drop(k) \| stride(nz)` | `strided_handle{id, offset=k, stride=nz, count=nx*ny}` — regular stride |
| **fluid** | `multi_slice_view` with `span<const index_slice>` | `gather_handle` with pre-computed `Kokkos::View<int*>` index array |
| **predicate** | `predicate_view` with runtime boolean scan | `gather_handle` from `parallel_scan` prefix-sum compaction |
| **optional** | Zero-size range when inactive | `handle` with `length=0` when inactive |

The y-plane is the most interesting case. Currently it requires a 70-line custom iterator with `std::div` in `operator+=`. With a 3D `Kokkos::View<real***, LayoutRight>`, it becomes `Kokkos::subview(field, ALL, j, ALL)` — a single line. Even with a 1D view, the kernel body is `data[start + (i/nz)*ny*nz + i%nz]` — the complexity moves from iterator state into inline index arithmetic, which the compiler can optimize.

### 5.5 Structured Bindings

Currently `auto&& [D, Rxyz] = scalar;` works because `tuple_size<scalar_real> = 2` with heterogeneous element types. With flat handles, `scalar_handle h` has 4 named members — structured bindings would give `auto [D, Rx, Ry, Rz] = h;` which is arguably better (more explicit, no nested `Rxyz` sub-tuple).

### 5.6 The `selection.apply()` Re-targeting Pattern

Currently, a `selection<L, R, Fn>` extracted from one field can be `.apply(other_field)` to extract the same logical region from a different field. This is used in `view_tuple_base::operator=` to re-target RHS expressions.

With handles, re-targeting is trivial: if you know `sel::D` maps to handle offset 0 within any scalar, then `other_scalar.D` gives you the re-targeted handle directly. The entire `apply` mechanism, `semiregular_box<Fn>` storage, and `selection` inheritance hierarchy dissolve.

---

## 6. Comparison With Community Patterns

| Project | Pattern | Similarity to Proposal |
|---|---|---|
| **Tpetra (Trilinos)** | Single `Kokkos::View<real**>` for multi-vector; column subviews | Matches "2D pool for same-sized buffers" option |
| **SCREAM (E3SM)** | Registry with `std::map<name, Field>`; 1D type-erased `DualView<char*>` | Matches "registry + handle" concept |
| **Cabana** | `MemberTypes<...>` with compile-time `slice<I>()` access | Matches "compile-time handle as NTTP index" |

The Kokkos community **explicitly discourages** `View<View<real*>*>` (view-of-views) due to construction, reference counting, and device access problems. The recommended patterns are: separate managed views in a host-side container, or a single 2D view for same-sized arrays.

---

## 7. Recommended Migration Path

### Phase 1: Introduce `field_storage` Registry (Low Risk)

Create a `field_storage` class that holds a flat `std::vector<Kokkos::View<real*>>`. Define `scalar_handle` and `vector_handle` structs. Add `get_buffer(handle)` accessor. Keep the existing tuple system as a compatibility layer that wraps the registry.

### Phase 2: Introduce Expression Templates (Medium Risk)

Define `binary_expr`, `handle_expr`, `scalar_expr` types. Implement `operator+/-/*/` on `scalar_handle` returning expression templates. Implement `assign(handle, expr)` using `Kokkos::parallel_for`. Test alongside existing `zip_transform_view` system.

### Phase 3: Migrate Selectors to Handle Patterns (Medium Risk)

Replace `sel::D` with struct member access. Replace `plane_view<1>` with strided handles. Pre-compute `gather_handle` index arrays for `multi_slice_view` and `predicate_view` at mesh construction time.

### Phase 4: Remove Tuple Infrastructure (High Impact)

Remove `container_tuple`, `view_tuple`, `single_view`, `tuple_pipe`, the `list_index` type system, and the `mp_list`-based selector dispatch. Replace `for_each`/`transform` with `handle_for_each`. Remove the ~310 lines of re-anchoring code.

### Phase 5: GPU Execution (Phase D from Prior Plan)

Switch `execution_space` to `DefaultExecutionSpace`. Move registry buffers to device memory. Expression template lambdas are already `KOKKOS_LAMBDA`-compatible (they capture only handles — trivially copyable integers).

---

## 8. Impact on Files

### Files That Would Be Eliminated

| File | Lines | Purpose | Replacement |
|---|---|---|---|
| `container_tuple.hpp` | ~70 | Owning `std::tuple<Args...>` wrapper | `field_storage` registry |
| `view_tuple.hpp` | ~350 | Non-owning view layer + `single_view` + re-anchoring | Handles (trivially copyable) |
| `tuple_pipe.hpp` | ~110 | 5 `operator\|` overloads for tuple dispatch | Handle-based pipe (simpler) |
| `ccs_range_utils.hpp` | ~220 | `view_closure`, `semiregular_box`, `bind_back`, `compose` | Standard lambdas (no `semiregular_box` needed) |

### Files That Would Be Heavily Rewritten

| File | Lines | Change |
|---|---|---|
| `tuple.hpp` | ~210 | Replace dual-inheritance `tuple` with `scalar_handle`/`vector_handle` |
| `tuple_math.hpp` | ~120 | Replace `zip_transform_view` construction with expression templates |
| `tuple_utils.hpp` | ~400 | Replace recursive `for_each`/`transform` with `handle_for_each`/`handle_transform` |
| `selector.hpp` | ~1100 | Replace `selection_view_fn` + custom views with handle patterns + strided/gather handles |
| `tuple_fwd.hpp` | ~650 | Replace `mp_list`-based concepts with handle-based concepts |
| `field.hpp` | ~155 | Replace `std::vector<scalar_real>` with template-parameterized `field_storage` |
| `lazy_views.hpp` | ~830 | Remove `zip_transform_view`, `stride_view`; keep `linear_distribute`, `cartesian_product` |

### Files That Would Be Lightly Modified

| File | Change |
|---|---|
| `scalar.hpp`, `vector.hpp` | Redefine as `scalar_handle`/`vector_handle` aliases |
| `selector_fwd.hpp` | Replace `list_index` types with `constexpr handle` values |
| `field_math.hpp` | Update to use handle-based `for_each` |
| `field_utils.hpp` | Replace `for_each_scalar/vector` with handle iteration |
| `algorithms.hpp` | Replace `minmax`/`max`/`dot` with handle-based reductions |
| `derivative.cpp` | Replace `get<si::D>(u)` with `u.D` member access |
| `gradient.cpp`, `laplacian.cpp` | Same pattern |
| `heat.cpp`, `scalar_wave.cpp` | Update field access patterns |

### Files Unchanged

| File | Why |
|---|---|
| All stencil files | No field DSL usage in library code |
| All matrix files | Already use `std::span`, which bridges to any backing storage |
| `kokkos_types.hpp` | Kokkos aliases remain |
| `index_extents.hpp`, `indexing.hpp` | Mesh indexing is independent of field storage |

---

## 9. Summary of Trade-offs

| Dimension | Current (Nested Tuples) | Proposed (Flat Handles) |
|---|---|---|
| **Mental model** | Deep: `mp_list` metaprogramming, `NestedTuple` recursion, dual-inheritance | Shallow: named structs, integer indices |
| **Copy/move safety** | Fragile: manual re-anchoring, placement-new, developer vigilance | Trivial: compiler-generated, no pointers |
| **Expression types** | Deep: `zip_transform_view` trees grow with expression complexity | Shallow: expression templates with bounded depth |
| **Parallelism** | None: serial iterators | Natural: `parallel_for` at materialization |
| **GPU readiness** | None: host-only iterators, `std::optional` in views | Ready: handles are trivially copyable integers |
| **Generality** | High: generic over any `TupleLike` shape | Fixed: scalar (4 buffers) and vector (12 buffers) are concrete types |
| **Compile time** | Heavy: `mp11` metaprogramming, deep template instantiation | Light: shallow templates, no metaprogramming |
| **Code volume** | ~3000 lines across tuple infrastructure | Estimated ~1200 lines for handle infrastructure |
| **Migration cost** | N/A (status quo) | High: touches 15+ files, rewrites ~2500 lines |
| **Future extensibility** | Adding new field shapes requires no new code (generic dispatch) | Adding new shapes requires new handle types and `handle_for_each` overloads |

---

## 10. Review Findings and Corrections

The proposal was reviewed by four independent agents. The findings below represent critical corrections, missing concerns, and recommended improvements.

### 10.1 Critical: `field` Templating Breaks the Entire System Polymorphism Stack

The proposal's recommendation to make `field` a template (`template<int NS, int NV> class field`) conflicts with the existing runtime dispatch architecture. The `system` class uses `std::variant<empty, scalar_wave, inviscid_vortex, heat, hyperbolic_eigenvalues>`. Every system method signature uses the untemplatized `field`, `field_view`, and `field_span` types:

```cpp
// system.hpp
system_stats stats(const field&, const field&, const step_controller&) const;
std::function<void(field_span)> rhs(field_view, real);
```

The integrators (`rk4.hpp`) store `field rk_rhs` and `field system_rhs` as owning members. The `simulation_cycle` creates fields at runtime based on Lua configuration. Templating `field` on `<NS, NV>` would require templating the entire `system`/`integrator`/`simulation_cycle` stack or introducing type-erasure.

**Correction:** Section 5.1 option (a) ("template `field`") is not viable as a near-term recommendation. Option (b) (runtime handles with runtime offsets) is the only compatible near-term path. The proposal should be revised to reflect this.

### 10.2 Critical: Expression Nodes Must Not Capture the Registry

The proposal's `handle_expr::operator()(const field_storage& reg, int i)` signature is incompatible with GPU execution. `field_storage` holds `std::vector<Kokkos::View<real*>>`, which is not trivially copyable and cannot be captured by `KOKKOS_LAMBDA`. Even on CPU, passing the registry through every expression evaluation is unnecessary overhead.

**Fix:** Expression nodes must carry pre-extracted `real*` pointers, resolved on the host before kernel launch:

```cpp
struct handle_expr {
    real* ptr;  // extracted from View::data() before parallel_for
    constexpr real operator()(int i) const { return ptr[i]; }
};
```

The `assign()` function should extract pointers from the registry, build the expression with raw pointers, then launch the kernel. The `(reg, i)` signature shown in the proposal must be replaced with `(i)`.

### 10.3 Critical: Aliasing Policy is Missing

The proposal does not address what happens with `u = u + something_derived_from_u` inside `parallel_for`. With serial iteration (the current system), this is safe because reads and writes are ordered. With parallel execution, same-buffer read+write is a data race.

**Fix:** Add a compile-time aliasing check:

```cpp
template <typename Expr>
void assign(field_storage& reg, handle dst, Expr expr) {
    static_assert(!expr_contains_handle(expr, dst),
        "destination handle appears in RHS expression — use a temporary");
    // ...
}
```

Or document the invariant: "The destination buffer must not appear in the RHS expression. Use a separate buffer for in-place updates." The current codebase already follows this convention implicitly (stencil operations always write to a separate output buffer).

### 10.4 Critical: Reduction Operations are Undesigned

The expression template framework is map-only (`binary_expr::operator()(int i) -> real`). Global reductions like `max(abs(u - sol) | m.fluid_all(...))` (used in `heat.cpp:73` and `scalar_wave.cpp:115`) require `Kokkos::parallel_reduce`, which has a fundamentally different interface.

**Fix:** Add a `reduce_expr` node type:

```cpp
template <typename ReduceOp, typename Expr>
struct reduce_expr {
    ReduceOp op;
    Expr expr;
    handle source;
    real materialize(field_storage& reg) const {
        real result;
        Kokkos::parallel_reduce(source.length,
            KOKKOS_LAMBDA(int i, real& val) { op(val, expr(i)); },
            result);
        return result;
    }
};
```

### 10.5 Important: The "Depth-1 Types" Claim is Incorrect

Section 4.3 claims expression template types have "depth always 1." This is false. `binary_expr<plus, binary_expr<mult, handle_expr, handle_expr>, handle_expr>` is depth 2. The actual benefit is that expression template nodes are **smaller per level** (no iterator state, no `semiregular_box`) and **fully inlineable** by the compiler into a single kernel body. The claim should be corrected.

### 10.6 Important: `gather_handle` Breaks Trivial Copyability

Section 5.4 mentions `gather_handle` with a `Kokkos::View<int*>` index array. `Kokkos::View` is reference-counted and not trivially copyable, contradicting the handle design principle.

**Fix:** `gather_handle` should store an integer index into a mesh-owned registry of pre-allocated device-resident index arrays, not a `View` directly.

### 10.7 Important: `handle::length` is a Stale-able Cache

Section 2.3 stores `length` in the handle for "kernel convenience." This creates an invalidation footgun — if a view is resized, cached handles become stale with no detection mechanism.

**Fix:** Remove `length` from `handle`. Query the registry at `assign()` time: `reg.buffers[dst.id].extent(0)`.

### 10.8 Important: `sel::R` Multi-Selection Has No Clean Handle Analog

`sel::R` applied to a vector selects 9 boundary buffers (`xRx, xRy, xRz, yRx, ..., zRz`) of potentially different sizes. The current system handles this via `mp_list`-based structural dispatch in `selection_view_fn`. A flat handle system needs a `handle_list` type that can hold heterogeneous-length handles and drive separate `parallel_for` calls for each.

### 10.9 Important: `dot(grad_G, du)` Fusion is Overstated

The proposal claims dot can be "one fused kernel" versus "3 separate view compositions." In reality, `grad(u)` (a sparse matrix-vector product) must be materialized into `du` before `dot(grad_G, du)` can run — the gradient cannot be expressed as a `binary_expr`. The current system also materializes gradient separately. The fusion improvement applies only to the dot product itself (element-wise multiply + component sum), not to the full RHS computation.

### 10.10 The Cheapest Fix: Replace `single_view` Placement-New

The proposal's biggest claimed safety benefit (eliminating re-anchoring bugs) can be captured without the full redesign. `single_view` uses placement-new because `ref_view` is non-reassignable. Replacing `single_view` inheritance with `semiregular_box<std::views::all_t<A>>` (which already exists in `ccs_range_utils.hpp`) eliminates the UB while leaving the rest of the architecture intact. **Estimated effort: 2-3 weeks**, with zero DSL surface changes and all 141 test cases passing without modification.

---

## 11. Revised Migration Recommendation

Based on the review findings, the migration should be resequenced:

### Phase 0 (Quick Win, 2-3 weeks): Eliminate Re-anchoring Hazard
Replace `container_tuple`/`view_tuple` dual-inheritance with a simpler single class that stores both `std::vector<real>` (or `Kokkos::View<real*>`) and exposes `std::span` accessors directly. Eliminate `single_view` placement-new. This fixes the UB risk and removes ~310 lines of complexity with zero DSL surface change.

### Phase 1 (4-6 weeks): Introduce Handle Types as Wrappers
Define `scalar_handle` and `vector_handle` as constexpr structs that encode `list_index` paths. Implement `get(handle, field)` as a thin wrapper over the existing `get<list_index>(tuple)` machinery. This provides the new API surface without changing storage.

### Phase 2 (4-6 weeks): Introduce Expression Templates
Define `binary_expr`, `handle_expr` (with pre-extracted `real*` pointers), and `reduce_expr` types. Implement `assign()` using `Kokkos::parallel_for` on `DefaultHostExecutionSpace`. Add compile-time aliasing detection. Test alongside existing `zip_transform_view` system.

### Phase 3 (6-8 weeks): Migrate Selectors
Replace runtime selector views with handle patterns:
- `plane_view<0/1/2>` → strided handles (or `Kokkos::subview` if using 3D views)
- `multi_slice_view` → pre-computed gather index arrays (built at mesh construction)
- `predicate_view` → prefix-scan compaction to gather index arrays
- `optional_view` → zero-length handle

### Phase 4 (3-4 weeks): Remove Tuple Infrastructure
Delete `container_tuple.hpp`, `view_tuple.hpp`, `tuple_pipe.hpp`, `ccs_range_utils.hpp`. Rewrite ~130 test cases.

### Phase 5 (Separate effort): GPU Execution
Switch execution space. Replace `std::span` bridges with device-accessible views. Add `Kokkos::fence()` and `deep_copy` at I/O boundaries.

**Total estimated effort: 4-6 months for Phases 0-4.** The C++23 features are nice-to-have but not essential — the entire migration is achievable in C++20.

---

## 12. Key Codebase Metrics

| Metric | Count |
|---|---|
| Direct `get<si::/vi::>` call sites | 73 across 9 files |
| `sel::` usage sites | 429 across 18 files |
| Actual `(ns, nv)` across all systems | heat: `{1,0}`, scalar_wave: `{1,0}`, others: `{0,0}` |
| Max observed `(ns, nv)` | `(1, 0)` — no system currently uses vector fields |
| `for_each`/`transform` from tuple_utils | ~30 direct call sites |
| Field test cases (`src/fields/*.t.cpp`) | 141 TEST_CASEs in 15 files |
| Test cases broken by Phase 4 | ~130 of 141 |
| Production files using `scalar_view`/`scalar_span` types | 6 (derivative, gradient, laplacian, heat, scalar_wave, field_data) |
| Lines in tuple infrastructure to be replaced | ~2800 |
| Lines of re-anchoring code eliminated by Phase 0 | ~310 |

---

## 13. Max-Capacity Design: Compile-Time Max, Runtime Allocation

### 13.1 The Problem Restated

The critical blocker from review finding 10.1 is that templating `field` on `<NS, NV>` breaks the `system` variant and `integrator` interfaces, which all use the non-templated `field` type. However, the actual field counts across all systems are tiny:

| System | nscalars | nvectors |
|---|---|---|
| `heat` | 1 | 0 |
| `scalar_wave` | 1 | 0 |
| `hyperbolic_eigenvalues` | 0 | 0 |
| `inviscid_vortex` | 0 | 0 |
| `empty` | 0 | 0 |

**Max observed: `(1, 0)`.** No system currently uses vector fields in the `field` container (scalar_wave stores its vector data as private members, not in the system `field`).

### 13.2 The Solution: `std::array<View, MaxN>` + Runtime Count

Instead of templating on the exact count, use a **fixed maximum capacity** with a runtime active count. The `field` type remains concrete and non-templated (assuming project-wide max constants):

```cpp
struct field {
    static constexpr int MaxScalars = 4;
    static constexpr int MaxVectors = 4;
    static constexpr int MaxN = MaxScalars * 4 + MaxVectors * 12;  // 64

    std::array<Kokkos::View<real*>, MaxN> buffers;
    int n_scalars = 0;
    int n_vectors = 0;
};
```

Key properties:
- **Single non-templated type.** All systems, integrators, and `simulation_cycle` use the same `field` type. The `system` variant and all `std::function<void(field_span)>` signatures remain unchanged.
- **Zero overhead for unused slots.** A default-constructed `Kokkos::View<real*>` is 24 bytes with no heap allocation, no ref-counting, `data() == nullptr`, `size() == 0`. For `MaxN=64`, the metadata waste is `(64 - n_active) × 24` bytes ≈ 1.5 KB — negligible versus field data.
- **Runtime bounds safety.** `n_scalars` and `n_vectors` enable debug-mode assertions: `assert(i < n_scalars)` before accessing scalar `i`.

### 13.3 Pattern Comparison

Six patterns were evaluated:

| Pattern | Preserves single type? | Unused slot cost | Kokkos compat | Complexity |
|---|---|---|---|---|
| **`std::array<View, MaxN>` + count** | Yes | 24 bytes/slot (no alloc) | Excellent | **Low** |
| `std::array<optional<View>, MaxN>` | Yes | 32 bytes/slot | Good | Medium (unwrap noise) |
| `std::inplace_vector<View, MaxN>` | Yes | Same as array | Good | C++26 only |
| `View<real**, LayoutRight>` 2D pool | Yes | Allocates all MaxN rows | Excellent | Heterogeneous sizes break it |
| `template<MaxS, MaxV>` everywhere | **No** (breaks variant) | Zero | Good | High (interface cascade) |
| `std::variant<field<1,0>, field<1,1>,...>` | Partially (double dispatch) | Zero | OK | High (visit × visit) |

**Recommendation: Pattern 2 (`std::array` + runtime count)** is the clear winner. It preserves the uniform `field` type, has negligible overhead, works natively with Kokkos, and requires no C++23/26 features.

### 13.4 Memory Impact with RK4

The RK4 integrator holds **4 simultaneous `field` objects**: `u0`, `u1` (in `simulation_cycle`), `rk_rhs`, `system_rhs` (in `rk4`). Euler uses 3.

With `MaxN=64` and the array pattern, each field has `64 × 24 = 1536` bytes of View metadata. Four fields = ~6 KB of metadata overhead — completely negligible versus the actual field data (which is `O(nx*ny*nz*8)` bytes per buffer).

The 4× data duplication (RK4 needs 4 copies of the field data) is the real memory cost, but this is identical to the current system — the max-capacity design adds no new data duplication.

### 13.5 The `field_view`/`field_span` Unification

The three field variants map cleanly:

```cpp
// All share the same MaxN. Buffer type varies.
template <typename BufferT>
struct flat_field {
    static constexpr int MaxN = MaxScalars * 4 + MaxVectors * 12;
    std::array<BufferT, MaxN> buffers;
    int n_scalars, n_vectors;
};

using field      = flat_field<Kokkos::View<real*>>;         // owning
using field_span = flat_field<std::span<real>>;               // mutable view
using field_view = flat_field<std::span<const real>>;         // const view
```

Converting `field` → `field_span`: wrap each `buffers[i].data()` + `buffers[i].extent(0)` into `std::span<real>`. This is a host-only operation (safe for `HostSpace` views).

Converting `field` → `field_view`: same, but with `std::span<const real>`.

If `MaxScalars` and `MaxVectors` are project-wide `inline constexpr` values, `flat_field<BufferT>` is a single concrete type per buffer type — no visible template parameters at call sites.

### 13.6 Retaining Scalar/Vector Structure

A key insight from the analysis: the operators (`derivative`, `gradient`, `laplacian`) work on individual `scalar_view`/`scalar_span` objects, not on whole fields. Rather than flattening into raw buffer arrays, we can retain the scalar/vector handle structure within the flat array:

```cpp
// Access scalar i's complete handle
scalar_handle scalar(int i) const { return {i * 4}; }

// Access vector j's complete handle
vector_handle vector(int j) const { return {MaxScalars * 4 + j * 12}; }

// Extract a scalar_span from the flat storage
scalar_span extract_scalar_span(int i) {
    int base = i * 4;
    return scalar_span_from_buffers(buffers[base], buffers[base+1],
                                    buffers[base+2], buffers[base+3]);
}
```

This preserves the `derivative(scalar_view u, scalar_span du)` interface unchanged — `scalar_view` is still a 4-buffer struct, just extracted from the flat array by index.

### 13.7 Prototype Implementation

A working handle header has been created at `src/fields/handle.hpp` with:

- **`field_layout<MaxS, MaxV>`** — encodes buffer arithmetic with compile-time max and runtime active counts
- **`buf_handle`**, **`scalar_handle`**, **`vector_handle`** — trivially copyable, structural types (C++20 NTTP-compatible)
- **`consteval` factory functions** — `make_scalar_handle()` / `make_vector_handle()` with bounds checking that fires as a compile error
- **`handle_sel::D/R/Rx/Ry/Rz/xR/yR/zR/Dx/Dy/Dz`** — selector dispatch matching current `sel::` semantics
- **`handle_for_each`** — replaces recursive `for_each` over `NestedTuple`s
- **`scalar_accessor<H>` / `vector_accessor<H>`** — NTTP-parameterized for zero-overhead dispatch
- **Comprehensive `static_assert` verification** — all arithmetic verified at compile time for a `field_layout<2,1>` with 20 buffers

All handle types are verified trivially copyable and aggregate (structural type requirements for NTTPs). The header compiles cleanly with GCC 14 in C++20 mode.

### 13.8 Layout Flexibility: Per-System vs Project-Wide

Two approaches are viable:

**(a) Project-wide max constants (simpler):**
```cpp
inline constexpr int MaxScalars = 4;
inline constexpr int MaxVectors = 4;
```
Single `field` type everywhere. Slightly wasteful for heat (uses 1/4 scalar slots) but negligible overhead.

**(b) Layout as a template parameter of `field` but with a project-wide max alias:**
```cpp
template <int MaxS, int MaxV> struct flat_field { ... };
using field = flat_field<4, 4>;  // project-wide alias
```
Same single type at the interface level, but the template structure allows compile-time specialization if needed later. System-specific layouts (`heat_layout`, `scalar_wave_layout`) are defined in the header for documentation but the actual `field` type uses the general max.

Approach (a) is recommended for simplicity — the template exists in the handle header for future flexibility but the `field` type itself should be a concrete alias.

### 13.9 The Storage Never Moves: Registry + Handle Architecture

A comprehensive trace of every `field`, `field_span`, and `field_view` usage site in the codebase reveals a critical insight: **the `std::array<View, MaxN>` should never be copied, moved, or passed around**. It should live in a centralized registry, and only lightweight handles flow through the system.

#### 13.9.1 Current Ownership Topology

Only **4 `field` objects** ever own storage simultaneously (during an RK4 step):

| Object | Location | Purpose |
|---|---|---|
| `u0` | `simulation_cycle::run()` local | Current time-step state |
| `u1` | `simulation_cycle::run()` local | Next time-step state |
| `rk_rhs` | `rk4` member | RK accumulator |
| `system_rhs` | `rk4` member | Per-stage RHS scratch |

(Euler uses 3: `u0`, `u1`, `system_rhs`.)

Every `field_span` and `field_view` in the codebase is already a **non-owning view** into one of these 4 storage owners. No `field_span` or `field_view` ever owns data. The problem is that passing them around currently requires copying a `std::vector<scalar_span>` (heap allocation + span metadata copy) at every call site.

#### 13.9.2 The Hidden Cost: Heap Allocation Per View Pass

`field_span` and `field_view` are passed **by value** everywhere — function parameters, lambda captures, `std::function` storage. Each by-value pass heap-allocates a new `std::vector<scalar_span>`. The worst offender is `system::rhs(field_view, real)`, which copies the `field_view` **3 times** (parameter, outer lambda capture, inner `std::function` capture) — 3 heap allocations per RHS evaluation, per RK4 stage, per time step.

#### 13.9.3 The Registry Design

```cpp
// The registry: a singleton or simulation-scoped object.
// It owns ALL field buffer storage. Nothing else does.
struct field_registry {
    static constexpr int MaxN = MaxScalars * 4 + MaxVectors * 12;

    // Storage: allocated once, never moved.
    std::array<Kokkos::View<real*>, MaxN> buffers;

    // Multiple "field slots" that share the same buffer array.
    // Each slot is a named group of handles (u0, u1, rk_rhs, system_rhs).
    // The slot count is small and fixed.

    // Access by handle:
    Kokkos::View<real*>& operator[](buf_handle h) { return buffers[h.id]; }
    const Kokkos::View<real*>& operator[](buf_handle h) const { return buffers[h.id]; }

    real* data(buf_handle h) { return buffers[h.id].data(); }
    int   size(buf_handle h) { return buffers[h.id].extent(0); }
};
```

**However**, the 4 field objects (`u0`, `u1`, `rk_rhs`, `system_rhs`) each need their own **independent set of buffers** — they don't share a single set of MaxN slots. With 4 field objects and MaxN=64 slots each, the registry holds `4 × 64 = 256` View slots total. But this is just `256 × 24 bytes = 6 KB` of metadata, and only `4 × n_active` buffers are actually allocated.

A cleaner model: each "field slot" is a fixed offset into the registry:

```cpp
struct field_registry {
    static constexpr int MaxN = MaxScalars * 4 + MaxVectors * 12;
    static constexpr int MaxSlots = 8;  // u0, u1, rk_rhs, system_rhs, + spares

    std::array<Kokkos::View<real*>, MaxSlots * MaxN> buffers;  // flat

    // A field_handle is just a slot index × MaxN offset.
};

// A field_ref is just a slot ID. 4 bytes. Trivially copyable.
struct field_ref {
    int slot;  // which field in the registry (0=u0, 1=u1, 2=rk_rhs, 3=system_rhs)
};
```

Now `field_ref` + `scalar_handle` together locate any buffer: `registry[slot * MaxN + scalar_handle.D().id]`. Both are plain integers. Copying them is a register copy. No heap allocation. No re-anchoring. No `std::vector<scalar_span>`.

#### 13.9.4 What This Eliminates

| Current cost | With registry + handles |
|---|---|
| `field u1{u0}` deep-copies all `std::vector<real>` buffers | `registry.deep_copy(slot_u1, slot_u0)` — explicit, no implicit copy |
| `field_span` by-value param → heap alloc `std::vector<scalar_span>` | `field_ref` by-value param → copy 1 integer |
| `field_view` captured 3× in `system::rhs` → 3 heap allocs | `field_ref` captured 3× → 3 integer copies |
| `swap(u0, u1)` → `swap_ranges` over nested tuples | `std::swap(slot_u0, slot_u1)` → swap 2 integers |
| `ensure_size` → move-assign a temporary `field` | `registry.allocate(slot, sizes)` — direct View allocation |
| `field_view io_view{u, error}` → constructs `std::vector<scalar_view>` | Construct a `field_ref` pointing to the I/O slot — zero alloc |

#### 13.9.5 The field_ref / field_span / field_view Unification

With a registry, the three variants collapse into a single type:

```cpp
struct field_ref {
    int slot;            // registry slot ID
    int n_scalars;       // active count (for iteration)
    int n_vectors;

    // Mutable or const access is controlled by how you use it,
    // not by the type. The registry returns View<real*> (always mutable).
    // Const-correctness is enforced by passing field_ref to functions
    // that take `const field_registry&` (read-only) vs `field_registry&` (writable).
};
```

The `field_view` vs `field_span` distinction moves from the type system to the function signature:
- `void rhs(const field_registry&, field_ref input, field_registry&, field_ref output, real time)` — input registry is const, output is mutable
- Or more idiomatically: `void rhs(field_ref input, field_ref output, real time)` where the registry is an implicit context (class member or global)

This is a significant simplification. The 3 type aliases (`field`, `field_span`, `field_view`), the `detail::field<S,V>` template, the 6 scalar/vector type aliases, and the entire `ConstructibleFromRange` conversion machinery all collapse into one integer + two counts.

#### 13.9.6 Integration With the Existing System Interface

The `system` variant methods currently take `field_view` and `field_span` by value. With the registry design:

```cpp
// Current:
std::function<void(field_span)> rhs(field_view, real);

// Proposed:
std::function<void(field_ref)> rhs(field_ref input, real time);
// The std::function captures field_ref (4 bytes) instead of field_view (heap-allocated vector of spans).
```

The `std::function<void(field_ref)>` captures a `field_ref` by value — a 12-byte trivially-copyable struct. No heap allocation for the capture if the `std::function` uses small-buffer optimization (most implementations have a 16-32 byte SBO buffer). This eliminates the 3× heap allocation in `system::rhs`.

The invocable assignment pattern (`u1 = integrate(...)`) becomes:
```cpp
// Current: u1 = std::function<void(field_span)>{...}; // invocable assignment
// Proposed: integrate(..., field_ref{slot_u1}); // writes directly into slot_u1
```

No intermediate `std::function` needed — the integrator writes directly into the output slot.

#### 13.9.7 Swap Becomes Trivial

```cpp
// Current: swap(u0, u1) → swap_ranges over nested tuple hierarchy
// Proposed:
std::swap(u0_ref.slot, u1_ref.slot);  // swap two integers
```

The buffers don't move. The handles just swap which slot they refer to. This is the same as swapping two pointers, but even cheaper — it's two integer assignments.

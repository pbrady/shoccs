# Phase 10: Expression Templates and Parallel Dispatch

**Goal:** Replace `zip_transform_view` lazy trees with expression templates that carry pre-extracted `real*` pointers and materialize via `Kokkos::parallel_for` on `DefaultHostExecutionSpace`.

**Depends on:** Phase 8 (registry, handles), Phase 9 (field lifecycle on registry)

**Read first:**
- `src/fields/field_registry.hpp` (Phase 8)
- `src/fields/handle.hpp` (handle types)
- `src/fields/lazy_views.hpp` (`zip_transform_view`, `repeat_n_view`)
- `src/fields/tuple_math.hpp` (current operator+/-/*/÷ via `zip_transform`)
- `src/fields/field_math.hpp` (field-level arithmetic)
- `src/fields/algorithms.hpp` (`dot`, `max`, `minmax`)
- `src/fields/tuple_utils.hpp` (`for_each`, `transform`, `resize_and_copy`)
- `src/systems/heat.cpp` (representative arithmetic: `u_rhs *= diffusivity`, `u_rhs += src`)
- `src/systems/scalar_wave.cpp` (`du = grad(u)`, `u_rhs = dot(grad_G, du)`)

**Test commands:**
```bash
cmake --build build
ctest --test-dir build -R t-expr
ctest --test-dir build
```

---

## Design Decisions (from proposal review)

- **D-ET1:** Expression nodes carry pre-extracted `real*` pointers, not registry references. The registry is consulted once at expression-construction time on the host.
- **D-ET2:** `Op` functors in expression nodes must be trivially copyable (`static_assert` enforced). This ensures future GPU compatibility.
- **D-ET3:** Aliasing detection: `assign()` checks at compile time (via handle ID comparison) or runtime (pointer comparison) that the destination does not appear in the source expression. If aliasing is detected, the assign stages through a temporary.
- **D-ET4:** Reductions (`max`, `minmax`, `dot` producing a scalar `real`) use `Kokkos::parallel_reduce`.
- **D-ET5:** Multi-buffer assignment (a scalar has 4 buffers of different sizes) dispatches one `parallel_for` per sub-buffer via `handle_for_each`.

---

## Items

### 10.1 — Expression node types (TDD)

- [ ] **10.1a** Create `src/fields/expr.t.cpp` testing expression node types:
  - `handle_expr{real* ptr}`: `operator()(int i)` returns `ptr[i]`.
  - `scalar_literal_expr{real value}`: `operator()(int i)` returns `value`.
  - `binary_expr<Op, Lhs, Rhs>`: `operator()(int i)` returns `op(lhs(i), rhs(i))`.
  - `unary_expr<Op, Arg>`: `operator()(int i)` returns `op(arg(i))`.
  - All types are trivially copyable (`static_assert`).
  - Test: construct `handle_expr` from a `real[]`, evaluate at several indices.
  - Test: construct `binary_expr<std::plus<>, handle_expr, handle_expr>`, evaluate.
  - Test: nested: `binary_expr<mult, binary_expr<plus, ...>, scalar_literal_expr>`, evaluate.
  - File: `src/fields/expr.t.cpp` (new)
  - CMake: `add_unit_test(expr "fields" fields Kokkos::kokkos)`
  - Test: `ctest --test-dir build -R t-expr`

- [ ] **10.1b** Implement `src/fields/expr.hpp`:
  - `handle_expr`, `scalar_literal_expr`, `binary_expr<Op, Lhs, Rhs>`, `unary_expr<Op, Arg>`.
  - `static_assert(std::is_trivially_copyable_v<Op>)` in `binary_expr`/`unary_expr`.
  - File: `src/fields/expr.hpp` (new)
  - Test: `ctest --test-dir build -R t-expr`

### 10.2 — Assign with parallel_for (TDD)

- [ ] **10.2a** Add tests to `expr.t.cpp` for `assign()`:
  - `assign(real* dst, int n, expr)`: fills `dst[0..n-1]` via `Kokkos::parallel_for`.
  - Test: `assign(dst, 100, binary_expr<plus>(handle_expr{a}, handle_expr{b}))` → `dst[i] == a[i] + b[i]`.
  - Test: `assign` with `scalar_literal_expr` → fills with constant.
  - Test aliasing detection: `assign(a, n, handle_expr{a})` raises or stages through temporary.
  - Test: `ctest --test-dir build -R t-expr`

- [ ] **10.2b** Implement `assign()` in `src/fields/expr.hpp`:
  - `template <typename Expr> void assign(real* dst, int n, Expr expr)` using `Kokkos::parallel_for(n, KOKKOS_LAMBDA(int i) { dst[i] = expr(i); })`.
  - Aliasing check: compare `dst` pointer against all `handle_expr.ptr` values in the expression tree at runtime. If aliased, allocate a temporary, evaluate into it, then copy.
  - File: `src/fields/expr.hpp`
  - Test: `ctest --test-dir build -R t-expr`

### 10.3 — Scalar-field expression operators (TDD)

- [ ] **10.3a** Add tests for operator overloads on handle-based scalar fields:
  - Given two `scalar_handle`s `a`, `b` with registry-backed storage:
    - `a + b` returns a `scalar_expr` with 4 `binary_expr<plus>` sub-expressions (D, Rx, Ry, Rz).
    - `a * 3.14` returns a `scalar_expr` with 4 `binary_expr<mult, handle_expr, scalar_literal_expr>`.
    - `assign_scalar(registry, dst_handle, a + b)` dispatches 4 `parallel_for` calls and produces correct results.
  - Test: `ctest --test-dir build -R t-expr`

- [ ] **10.3b** Implement scalar expression operators:
  - `scalar_expr` struct: holds 4 sub-expressions (D, Rx, Ry, Rz).
  - `operator+(scalar_bound_expr, scalar_bound_expr)` → `scalar_expr`.
  - `operator*(scalar_bound_expr, real)` → `scalar_expr`.
  - `assign_scalar(registry, field_ref, scalar_handle, scalar_expr)` → 4× `assign()`.
  - A `scalar_bound_expr` is created by binding a `scalar_handle` to a registry: `bind(registry, field_ref, scalar_handle)` returns a struct with 4 `handle_expr` (pre-extracted `real*` pointers).
  - File: `src/fields/expr.hpp`
  - Test: `ctest --test-dir build -R t-expr`

### 10.4 — Reduction operations (TDD)

- [ ] **10.4a** Add tests for `reduce()`:
  - `reduce_max(real* data, int n)` → `real` via `Kokkos::parallel_reduce`.
  - `reduce_min(real* data, int n)` → `real`.
  - `reduce_sum(real* data, int n)` → `real`.
  - `reduce_expr_max(expr, int n)` → `real` (reduce over an expression without materializing).
  - Test: `ctest --test-dir build -R t-expr`

- [ ] **10.4b** Implement reductions in `src/fields/expr.hpp`:
  - Use `Kokkos::parallel_reduce` with `Kokkos::Max<real>`, `Kokkos::Min<real>`, `Kokkos::Sum<real>`.
  - File: `src/fields/expr.hpp`
  - Test: `ctest --test-dir build -R t-expr`

### 10.5 — Mutating operators (+=, -=, *=, /=)

- [ ] **10.5a** Add tests for in-place operators:
  - `plus_assign(registry, field_ref, scalar_handle, scalar_expr)`: `dst[i] += expr(i)`.
  - Same for `minus_assign`, `times_assign`, `divide_assign`.
  - Test aliasing: `a += a * 2` must work (read-before-write is element-wise safe for `+=`).
  - Test: `ctest --test-dir build -R t-expr`

- [ ] **10.5b** Implement mutating operators:
  - `plus_assign` dispatches `parallel_for(n, [=](int i) { dst[i] += expr(i); })`.
  - These are always safe for element-wise aliasing (each thread reads and writes only `dst[i]`).
  - File: `src/fields/expr.hpp`
  - Test: `ctest --test-dir build -R t-expr`

### 10.6 — Integration with heat system

- [ ] **10.6a** Replace the field arithmetic in `heat::rhs` with expression template calls:
  - `u_rhs *= diffusivity` → `times_assign(registry, ..., scalar_literal_expr{diffusivity})`.
  - `u_rhs += src` → `plus_assign(registry, ..., src_expr)`.
  - Keep the operator calls (`lap(u, neumann_u)`) unchanged — they still use span-based `scalar_view`/`scalar_span`.
  - File: `src/systems/heat.cpp`
  - Test: `ctest --test-dir build -R t-heat`

---

## Completion Criteria

- Expression node types are trivially copyable.
- `assign()` dispatches via `Kokkos::parallel_for` on `DefaultHostExecutionSpace`.
- Aliasing is detected and handled (temporary staging).
- Scalar expression operators produce correct results for `+`, `-`, `*`, `/` and compound `+=`, etc.
- Reductions (`max`, `min`, `sum`) use `Kokkos::parallel_reduce`.
- Heat system arithmetic uses expression templates.
- All existing tests pass.

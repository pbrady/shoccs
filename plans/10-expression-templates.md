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
- `src/temporal/slot_ops.hpp` (`slot_assign_lc`, `slot_accumulate` — serial loops to parallelize)

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
- **D-ET3:** Aliasing detection: `assign()` checks at runtime (pointer comparison via `contains_ptr`) that the destination does not appear in the source expression. If aliasing is detected, the assign stages through a temporary `Kokkos::View<real*>`. Mutating operators (`+=`, `-=`, `*=`, `/=`) skip aliasing checks because element-wise read-modify-write of `dst[i]` is always safe.
- **D-ET4:** Reductions (`max`, `min`, `sum` producing a scalar `real`) use `Kokkos::parallel_reduce`.
- **D-ET5:** Multi-buffer assignment (a scalar has 4 buffers of different sizes) dispatches one `parallel_for` per sub-buffer. The `scalar_expr<Expr>` template holds `std::array<Expr, 4>` and `std::array<int, 4>` (sizes).

---

## Items

### 10.1 — Expression node types (TDD)

- [x] **10.1a** Create `src/fields/expr.t.cpp` with Kokkos-enabled test main:
  - Uses custom `main()` with `Kokkos::ScopeGuard` (pattern from `field_registry.t.cpp`, per D-R9) because later tests (10.2+) allocate `Kokkos::View`.
  - CMake addition in `src/fields/CMakeLists.txt` (manual test executable, NOT `add_unit_test`):
    ```cmake
    if (BUILD_TESTING)
      add_executable(t-expr expr.t.cpp)
      target_link_libraries(t-expr Catch2::Catch2 fields Kokkos::kokkos)
      add_test(NAME t-expr COMMAND t-expr)
      set_tests_properties(t-expr PROPERTIES LABELS "fields")
    endif()
    ```
  - Tests (use plain `real[]` stack arrays — no Views needed yet):
    - `handle_expr{ptr}`: verify `operator()(0)`, `operator()(1)`, `operator()(99)` return `ptr[i]`
    - `scalar_literal_expr{3.14}`: verify returns `3.14` for any `i`
    - `binary_expr<std::plus<>, handle_expr, handle_expr>`: evaluate at several indices
    - `unary_expr<std::negate<>, handle_expr>`: evaluate at several indices
    - Nested: `binary_expr<std::multiplies<>, binary_expr<std::plus<>, A, B>, scalar_literal_expr>` — verify `(a[i]+b[i]) * c`
    - `STATIC_REQUIRE(std::is_trivially_copyable_v<handle_expr>)` and same for all node types
  - Files: `src/fields/expr.t.cpp` (new), `src/fields/CMakeLists.txt` (append)
  - Test: `ctest --test-dir build -R t-expr`

- [x] **10.1b** Implement `src/fields/expr.hpp`:
  - Node types:
    ```cpp
    struct handle_expr {
        real* ptr;
        constexpr real operator()(int i) const { return ptr[i]; }
    };
    struct scalar_literal_expr {
        real value;
        constexpr real operator()(int i) const { return value; }
    };
    template <typename Op, typename Lhs, typename Rhs>
    struct binary_expr {
        static_assert(std::is_trivially_copyable_v<Op>);
        static_assert(std::is_trivially_copyable_v<Lhs>);
        static_assert(std::is_trivially_copyable_v<Rhs>);
        Op op; Lhs lhs; Rhs rhs;
        constexpr real operator()(int i) const { return op(lhs(i), rhs(i)); }
    };
    template <typename Op, typename Arg>
    struct unary_expr {
        static_assert(std::is_trivially_copyable_v<Op>);
        static_assert(std::is_trivially_copyable_v<Arg>);
        Op op; Arg arg;
        constexpr real operator()(int i) const { return op(arg(i)); }
    };
    ```
  - **No explicit CTAD deduction guides needed:** C++20 aggregate CTAD (P1816R0) applies because `binary_expr` and `unary_expr` are aggregates (only public data members, no user-declared constructors). Expressions like `binary_expr{std::plus<>{}, handle_expr{a}, handle_expr{b}}` deduce template arguments automatically.
  - **Namespace:** All types and free functions in `expr.hpp` live in `namespace ccs`, consistent with `handle.hpp`, `field_registry.hpp`, and `kokkos_types.hpp`.
  - Aliasing-detection helper (used by `assign()` in 10.2):
    ```cpp
    // Base cases:
    inline bool contains_ptr(const handle_expr& e, const real* target)
        { return e.ptr == target; }
    inline bool contains_ptr(const scalar_literal_expr&, const real*)
        { return false; }
    // Recursive cases:
    template <typename Op, typename Lhs, typename Rhs>
    inline bool contains_ptr(const binary_expr<Op, Lhs, Rhs>& e, const real* target)
        { return contains_ptr(e.lhs, target) || contains_ptr(e.rhs, target); }
    template <typename Op, typename Arg>
    inline bool contains_ptr(const unary_expr<Op, Arg>& e, const real* target)
        { return contains_ptr(e.arg, target); }
    ```
  - `static_assert` trivially-copyable for all node types (at namespace scope, after struct definitions).
  - **`constexpr` vs `KOKKOS_INLINE_FUNCTION`:** Per D1 (host-only execution), `constexpr` is sufficient for all `operator()` methods. On host, `KOKKOS_LAMBDA` is just `[=]`, so `constexpr` member functions are callable. Phase 14 (GPU) may add `KOKKOS_INLINE_FUNCTION` annotations; for now, keep it simple.
  - Includes: `"kokkos_types.hpp"` (provides `ccs::execution_space`, `ccs::memory_space`, and transitively `<Kokkos_Core.hpp>`), `"shoccs_config.hpp"` (provides `ccs::real`), `<type_traits>`, `<functional>`. No need to include `<Kokkos_Core.hpp>` directly — it comes via `kokkos_types.hpp`.
  - File: `src/fields/expr.hpp` (new)
  - Test: `ctest --test-dir build -R t-expr`

### 10.2 — Assign with parallel_for (TDD)

- [x] **10.2a** Add tests to `expr.t.cpp` for `assign()`:
  - Test setup uses `Kokkos::View<real*>` for buffers (requires the ScopeGuard main from 10.1a).
  - Test: `assign(dst.data(), 100, binary_expr{std::plus<>{}, handle_expr{a.data()}, handle_expr{b.data()}})` — verify `dst(i) == a(i) + b(i)` for all `i`.
  - Test: `assign(dst.data(), 50, scalar_literal_expr{7.0})` — all 50 elements are 7.0.
  - Test aliasing: `assign(a.data(), n, handle_expr{a.data()})` where `a` was filled with `{1,2,...,n}` — after assign, `a[i]` is still `i+1` (staged through temporary, no data race).
  - Test non-aliasing: `assign(dst.data(), n, handle_expr{src.data()})` — verify `dst` equals `src` (direct path, no temporary).
  - Depends on: 10.1a, 10.1b
  - File: `src/fields/expr.t.cpp` (append)
  - Test: `ctest --test-dir build -R t-expr`

- [x] **10.2b** Implement `assign()` in `src/fields/expr.hpp`:
  - Signature: `template <typename Expr> void assign(real* dst, int n, Expr expr)`
  - Implementation:
    1. Check aliasing: `if (contains_ptr(expr, dst))`
    2. If no alias: `Kokkos::parallel_for(Kokkos::RangePolicy<execution_space>(0, n), KOKKOS_LAMBDA(int i) { dst[i] = expr(i); });`
    3. If aliased: allocate `Kokkos::View<real*, memory_space> tmp("expr_tmp", n)`, evaluate into `tmp.data()` via step 2 (recursive call with `tmp.data()` as dst — guaranteed non-aliasing), then `Kokkos::deep_copy(dst_um, tmp)` where `dst_um` is `Kokkos::View<real*, memory_space, Kokkos::MemoryUnmanaged>(dst, n)`. Both views must explicitly specify `memory_space` (`ccs::memory_space` from `kokkos_types.hpp` = `execution_space::memory_space`) to avoid mismatch if `DefaultExecutionSpace` differs from `DefaultHostExecutionSpace`.
  - Uses `ccs::execution_space` from `kokkos_types.hpp` (`DefaultHostExecutionSpace`).
  - Depends on: 10.1b
  - File: `src/fields/expr.hpp`
  - Test: `ctest --test-dir build -R t-expr`

### 10.3 — Scalar-field expression operators (TDD)

- [x] **10.3a** Add tests for `bind_scalar` and scalar expression operators:
  - Test setup: `field_registry<2, 1, 0> reg;` — allocate scalar 0 in slots 0 and 1 with known sizes (e.g., `reg.allocate_scalar(0, 0, 100, 5, 3, 2)` and same for slot 1). Fill with known values via `reg.view(ref, bh)(i) = ...`. Use `constexpr auto sh = scalar_handle{0};`.
  - Test `bind_scalar(reg, ref, sh)`:
    - Returns `scalar_expr<handle_expr>` with 4 valid pointers matching `reg.data(ref, bh)` for each of `sh.D()`, `sh.Rx()`, `sh.Ry()`, `sh.Rz()`.
    - `.sizes[0]` == `reg.size(ref, sh.D())` == 100, `.sizes[1]` == 5, `.sizes[2]` == 3, `.sizes[3]` == 2.
  - Test `a + b` where `a`, `b` are `scalar_expr<handle_expr>`:
    - Result type is `scalar_expr<binary_expr<std::plus<>, handle_expr, handle_expr>>`.
    - `result.exprs[0](i)` == `a_D[i] + b_D[i]` for several `i`.
  - Test `a * 3.14`:
    - `result.exprs[0](i)` == `a_D[i] * 3.14`.
  - Test `3.14 * a` (scalar-left):
    - Same result as `a * 3.14`.
  - Test `assign_scalar(reg, dst_ref, sh, a + b)`:
    - After call, all 4 registry buffers of `dst_ref` contain the sum of `a` and `b`.
    - Verify D, Rx, Ry, Rz buffers separately (they have different sizes).
  - Depends on: 10.2b
  - File: `src/fields/expr.t.cpp` (append)
  - Test: `ctest --test-dir build -R t-expr`

- [x] **10.3b** Implement scalar expression operators in `src/fields/expr.hpp`:
  - `scalar_expr<Expr>` template:
    ```cpp
    template <typename Expr>
    struct scalar_expr {
        std::array<Expr, 4> exprs;  // [0]=D, [1]=Rx, [2]=Ry, [3]=Rz
        std::array<int, 4> sizes;   // buffer lengths, same index order
    };
    ```
    Index ordering matches `scalar_handle::all()` which returns `{D(), Rx(), Ry(), Rz()}`. This consistency is critical for `bind_scalar` and `assign_scalar` to use the same index `i` for both `sh.all()[i]` and `expr.exprs[i]`/`expr.sizes[i]`.
  - `bind_scalar` factory (mutable registry → `real*` in handle_expr, per D-R14):
    ```cpp
    template <int MS, int MaxS, int MaxV>
    scalar_expr<handle_expr> bind_scalar(
        field_registry<MS, MaxS, MaxV>& reg,
        field_ref ref, scalar_handle sh);
    ```
    Implementation: iterate `auto bufs = sh.all();` (returns `std::array<buf_handle, 4>` in order D, Rx, Ry, Rz). For each `i` in `[0,4)`:
    - `result.exprs[i] = handle_expr{reg.data(ref, bufs[i])};`
    - `result.sizes[i] = reg.size(ref, bufs[i]);`
    Per D-R14, only the mutable overload is provided in Phase 10. A const-registry variant (requiring `const_handle_expr` or templatized `handle_expr<T>`) is deferred to Phase 14.
  - Binary operators — free functions found via ADL on `scalar_expr`:
    ```cpp
    template <typename E1, typename E2>
    auto operator+(scalar_expr<E1> a, scalar_expr<E2> b)
        -> scalar_expr<binary_expr<std::plus<>, E1, E2>>;
    // Same for operator-, operator*, operator/
    ```
    Implementation: for each `i` in `[0,4)`, build `binary_expr{std::plus<>{}, a.exprs[i], b.exprs[i]}`. Copy `sizes` from `a` (both must agree; no runtime check — caller responsibility, matching existing tuple_math semantics).
  - Scalar-right: `scalar_expr<E> op real` → broadcasts `scalar_literal_expr{v}`:
    ```cpp
    template <typename E>
    auto operator*(scalar_expr<E> a, real v)
        -> scalar_expr<binary_expr<std::multiplies<>, E, scalar_literal_expr>>;
    ```
  - Scalar-left: `real op scalar_expr<E>` — for commutative ops (`+`, `*`), delegate to scalar-right; for non-commutative (`-`, `/`), build `scalar_literal_expr` as Lhs:
    ```cpp
    // Commutative: delegate
    template <typename E>
    auto operator+(real v, scalar_expr<E> a) { return a + v; }
    template <typename E>
    auto operator*(real v, scalar_expr<E> a) { return a * v; }
    // Non-commutative: scalar on left
    template <typename E>
    auto operator-(real v, scalar_expr<E> a)
        -> scalar_expr<binary_expr<std::minus<>, scalar_literal_expr, E>>;
    template <typename E>
    auto operator/(real v, scalar_expr<E> a)
        -> scalar_expr<binary_expr<std::divides<>, scalar_literal_expr, E>>;
    ```
    Implementation for non-commutative: for each `i` in `[0,4)`, build `binary_expr{std::minus<>{}, scalar_literal_expr{v}, a.exprs[i]}`. Copy `sizes` from `a`.
  - `assign_scalar`:
    ```cpp
    template <int MS, int MaxS, int MaxV, typename Expr>
    void assign_scalar(field_registry<MS, MaxS, MaxV>& reg,
                       field_ref ref, scalar_handle sh,
                       const scalar_expr<Expr>& expr);
    ```
    Implementation: `auto bufs = sh.all();` (D, Rx, Ry, Rz order), for each `i` in `[0,4)` call `assign(reg.data(ref, bufs[i]), expr.sizes[i], expr.exprs[i])`. Note: `expr.sizes[i]` and `bufs[i]` use the same index ordering (0=D, 1=Rx, 2=Ry, 3=Rz) because `bind_scalar` populates arrays using `sh.all()` in the same order.
  - Depends on: 10.2b
  - File: `src/fields/expr.hpp`
  - Test: `ctest --test-dir build -R t-expr`

- [x] **10.3c** Add tests for non-commutative scalar-left operators (review follow-up):
  - The `real - scalar_expr` and `real / scalar_expr` operators have distinct code paths where `scalar_literal_expr` is placed as LHS of the `binary_expr`, unlike the commutative `+`/`*` which delegate to scalar-right. An argument-order swap would silently produce wrong results. Currently untested.
  - Test `5.0 - a` where `a = bind_scalar(...)` with known values: verify `result.exprs[0](i) == 5.0 - a_D[i]` (not `a_D[i] - 5.0`).
  - Test `10.0 / a` where `a` has nonzero values: verify `result.exprs[0](i) == 10.0 / a_D[i]` (not `a_D[i] / 10.0`).
  - Depends on: 10.3b
  - File: `src/fields/expr.t.cpp` (append)
  - Test: `ctest --test-dir build -R t-expr`

### 10.4 — Reduction operations (TDD)

- [x] **10.4a** Add tests for reductions to `expr.t.cpp`:
  - `reduce_max(real* data, int n)` → `real`: fill with `{1,2,...,n}`, verify returns `n`.
  - `reduce_min(real* data, int n)` → `real`: verify returns `1`.
  - `reduce_sum(real* data, int n)` → `real`: verify returns `n*(n+1)/2`.
  - `reduce_max(expr, n)` → reduce over expression without materializing: `reduce_max(binary_expr{plus, handle_expr{a}, handle_expr{b}}, n)`.
  - Edge case: `n == 0` returns identity element (`std::numeric_limits<real>::lowest()` for max, `std::numeric_limits<real>::max()` for min, `0` for sum). These match `Kokkos::reduction_identity` defaults.
  - Depends on: 10.1b
  - File: `src/fields/expr.t.cpp` (append)
  - Test: `ctest --test-dir build -R t-expr`

- [x] **10.4b** Implement reductions in `src/fields/expr.hpp`:
  - Buffer-level:
    ```cpp
    inline real reduce_max(const real* data, int n);  // parallel_reduce + Kokkos::Max<real>
    inline real reduce_min(const real* data, int n);  // parallel_reduce + Kokkos::Min<real>
    inline real reduce_sum(const real* data, int n);  // parallel_reduce + default sum
    ```
  - Expression-level (generic):
    ```cpp
    template <typename Expr>
    real reduce_max(Expr expr, int n);  // parallel_reduce over expr(i)
    template <typename Expr>
    real reduce_sum(Expr expr, int n);
    ```
  - All use `Kokkos::RangePolicy<execution_space>(0, n)`.
  - Identity values: `Max` → `std::numeric_limits<real>::lowest()`, `Min` → `std::numeric_limits<real>::max()`, `Sum` → `0.0`.
  - Depends on: 10.1b
  - File: `src/fields/expr.hpp`
  - Test: `ctest --test-dir build -R t-expr`

### 10.5 — Mutating operators (+=, -=, *=, /=)

- [x] **10.5a** Add tests for buffer-level and scalar-level mutating operators:
  - Buffer-level: `plus_assign(dst, n, handle_expr{src})` → `dst[i] += src[i]` for all `i`.
  - Buffer-level: `times_assign(dst, n, scalar_literal_expr{2.0})` → `dst[i] *= 2.0`.
  - Buffer-level: same for `minus_assign`, `divide_assign`.
  - Scalar-level: `plus_assign_scalar(reg, ref, sh, bound_b)` → all 4 buffers of slot `ref` get `+= b[i]`.
  - Scalar-level: `times_assign_scalar(reg, ref, sh, 3.14)` → all 4 buffers multiplied by constant.
  - Aliasing safety test: `plus_assign(a.data(), n, handle_expr{a.data()})` where `a = {1,2,...,n}` → after call, `a[i] == 2*i` (element-wise safe, no temporary).
  - Depends on: 10.1b, 10.3b (scalar-level tests need `bind_scalar`)
  - File: `src/fields/expr.t.cpp` (append)
  - Test: `ctest --test-dir build -R t-expr`

- [x] **10.5b** Implement mutating operators in `src/fields/expr.hpp`:
  - Buffer-level functions:
    ```cpp
    template <typename Expr> void plus_assign(real* dst, int n, Expr expr);
    template <typename Expr> void minus_assign(real* dst, int n, Expr expr);
    template <typename Expr> void times_assign(real* dst, int n, Expr expr);
    template <typename Expr> void divide_assign(real* dst, int n, Expr expr);
    ```
    Each dispatches `Kokkos::parallel_for(RangePolicy<execution_space>(0, n), KOKKOS_LAMBDA(int i) { dst[i] OP= expr(i); })`.
  - Scalar-level convenience:
    ```cpp
    template <int MS, int MaxS, int MaxV, typename Expr>
    void plus_assign_scalar(field_registry<MS,MaxS,MaxV>& reg,
                            field_ref ref, scalar_handle sh,
                            const scalar_expr<Expr>& expr);
    // Iterates sh.all(), calls plus_assign(data, size, sub_expr) for each buffer.

    template <int MS, int MaxS, int MaxV>
    void times_assign_scalar(field_registry<MS,MaxS,MaxV>& reg,
                             field_ref ref, scalar_handle sh, real value);
    // Iterates sh.all(), calls times_assign(data, size, scalar_literal_expr{value}) for each buffer.
    ```
  - No aliasing check: element-wise compound-assign is always safe (each thread reads/writes only `dst[i]`).
  - Depends on: 10.3b (for `scalar_expr`)
  - File: `src/fields/expr.hpp`
  - Test: `ctest --test-dir build -R t-expr`

### 10.6 — Integration with heat system

- [x] **10.6a** Replace `u_rhs *= diffusivity` in `heat::rhs` with expression-template dispatch:
  - Current code (`heat.cpp:111`): `u_rhs *= diffusivity;` — uses `tuple_math::operator*=` which runs sequential loops over all 4 spans in `scalar_span`.
  - Context: `heat::rhs` (line 102) already has `constexpr auto sh = scalar_handle{0};` at line 105, and parameters `sim_registry& out_reg, field_ref output` in the signature. These are exactly what `times_assign_scalar` needs.
  - New code: `times_assign_scalar(out_reg, output, sh, diffusivity);` — dispatches 4 `Kokkos::parallel_for` calls (one per buffer: D, Rx, Ry, Rz).
  - Add `#include "fields/expr.hpp"` to `heat.cpp` (heat.cpp is in `src/systems/`, include root is `src/`).
  - The `extract_scalar_span` call for `u_rhs` on line 107 is still needed for:
    - `u_rhs = lap(u, neumann_u)` (line 110: operator call uses `scalar_span`/`scalar_view`)
    - `u_rhs | m.fluid_all(...) += src` (line 117: selection-based — Phase 11 scope)
    - `u_rhs | m.dirichlet(...) = 0` (line 118: selection-based — Phase 11 scope)
  - **Out of scope for 10.6:** selection-based operations (`|`) stay as-is; they require Phase 11 (selector migration). Only the full-field `*= diffusivity` is replaced.
  - Depends on: 10.5b
  - File: `src/systems/heat.cpp`
  - Test: `ctest --test-dir build -R t-heat` and `ctest --test-dir build` (full suite)

### 10.7 — Parallelize slot_ops.hpp

- [x] **10.7a** Replace serial `for` loops in `slot_ops.hpp` with `Kokkos::parallel_for`:
  - `slot_assign_lc` (line 33): `for (int i = 0; i < n; ++i) d[i] = s0[i] + coeff * r[i]` →
    ```cpp
    Kokkos::parallel_for(Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(int i) { d[i] = s0[i] + coeff * r[i]; });
    ```
    All captures (`d`, `s0`, `r` are `real*` / `const real*`; `coeff` is `real`) are trivially copyable scalars/pointers — safe for `KOKKOS_LAMBDA` (which is `[=]` on host).
  - `slot_accumulate` (line 49): `for (int i = 0; i < n; ++i) d[i] += coeff * r[i]` →
    ```cpp
    Kokkos::parallel_for(Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(int i) { d[i] += coeff * r[i]; });
    ```
  - `slot_zero` already uses `Kokkos::deep_copy` — no change needed.
  - `ccs::execution_space` is already available via the existing `#include "fields/field_registry.hpp"` (line 3 of `slot_ops.hpp`), which transitively includes `kokkos_types.hpp` → `Kokkos_Core.hpp`. No new includes needed.
  - **No explicit fencing needed:** On `DefaultHostExecutionSpace` (Serial/OpenMP), `Kokkos::parallel_for` is synchronous — it returns only after the kernel completes. Consecutive `parallel_for` calls in `slot_assign_lc` (iterating over scalars × 4 buffers) execute sequentially without race conditions.
  - Depends on: none (independent of 10.1-10.6, but logically part of Phase 10's parallel dispatch goal)
  - File: `src/temporal/slot_ops.hpp`
  - Test: `ctest --test-dir build` (existing rk4/euler/system tests exercise these paths through simulation_cycle)

---

## Ordering Constraints

```
10.1a → 10.1b (tests before impl)
10.1b → 10.2a → 10.2b (assign depends on node types)
10.2b → 10.3a → 10.3b (scalar expr depends on assign)
10.3b → 10.3c (non-commutative scalar-left tests, review follow-up)
10.1b → 10.4a → 10.4b (reductions depend on node types, independent of 10.3)
10.3b → 10.5a → 10.5b (mutating ops depend on scalar_expr)
10.5b → 10.6a (heat integration depends on mutating ops)
10.7a is independent (can be done in parallel with 10.1-10.6)
```

---

## Scope Notes

- **Scalar-only in Phase 10:** Vector expression templates (for `dot(grad_G, du)` in scalar_wave) are future work. Phase 10 covers scalar fields only.
- **No selector integration:** Selection-based operations (`u | sel::D`, `u | m.fluid_all(...)`) remain tuple-based. Phase 11 (selector migration) handles those.
- **Coexistence:** The expression template system is additive — it does not remove the existing `tuple_math`/`field_math` operators. Phase 12 (legacy removal) handles cleanup.

---

## Completion Criteria

- Expression node types are trivially copyable.
- `assign()` dispatches via `Kokkos::parallel_for` on `DefaultHostExecutionSpace`.
- Aliasing is detected and handled (temporary staging via `Kokkos::View`).
- `scalar_expr<E>` operators produce correct results for `+`, `-`, `*`, `/`.
- Scalar-level `assign_scalar`, `times_assign_scalar`, `plus_assign_scalar` dispatch 4× `parallel_for`.
- Reductions (`max`, `min`, `sum`) use `Kokkos::parallel_reduce`.
- Heat system `u_rhs *= diffusivity` uses `times_assign_scalar`.
- `slot_ops.hpp` serial loops replaced with `Kokkos::parallel_for`.
- All existing tests pass.

#pragma once

#include "field_registry.hpp"
#include "kokkos_types.hpp"
#include "shoccs_config.hpp"

#include <array>
#include <functional>
#include <limits>
#include <type_traits>

namespace ccs
{

// ---------------------------------------------------------------------------
// Expression node types for expression templates.
//
// Each node type carries pre-extracted data (pointers or values) and provides
// operator()(int i) to evaluate at index i. All types are trivially copyable
// to ensure safe capture in Kokkos lambdas (D-ET2).
// ---------------------------------------------------------------------------

struct handle_expr {
    real* ptr;
    constexpr real operator()(int i) const { return ptr[i]; }
};

struct scalar_literal_expr {
    real value;
    constexpr real operator()(int i) const { (void)i; return value; }
};

template <typename Op, typename Lhs, typename Rhs>
struct binary_expr {
    static_assert(std::is_trivially_copyable_v<Op>);
    static_assert(std::is_trivially_copyable_v<Lhs>);
    static_assert(std::is_trivially_copyable_v<Rhs>);
    Op op;
    Lhs lhs;
    Rhs rhs;
    constexpr real operator()(int i) const { return op(lhs(i), rhs(i)); }
};

template <typename Op, typename Arg>
struct unary_expr {
    static_assert(std::is_trivially_copyable_v<Op>);
    static_assert(std::is_trivially_copyable_v<Arg>);
    Op op;
    Arg arg;
    constexpr real operator()(int i) const { return op(arg(i)); }
};

// Trivially-copyable assertions at namespace scope.
static_assert(std::is_trivially_copyable_v<handle_expr>);
static_assert(std::is_trivially_copyable_v<scalar_literal_expr>);

// ---------------------------------------------------------------------------
// Aliasing detection: contains_ptr checks if a destination pointer appears
// anywhere in an expression tree (D-ET3).
// ---------------------------------------------------------------------------

inline bool contains_ptr(const handle_expr& e, const real* target)
{
    return e.ptr == target;
}

inline bool contains_ptr(const scalar_literal_expr&, const real*)
{
    return false;
}

template <typename Op, typename Lhs, typename Rhs>
inline bool contains_ptr(const binary_expr<Op, Lhs, Rhs>& e, const real* target)
{
    return contains_ptr(e.lhs, target) || contains_ptr(e.rhs, target);
}

template <typename Op, typename Arg>
inline bool contains_ptr(const unary_expr<Op, Arg>& e, const real* target)
{
    return contains_ptr(e.arg, target);
}

// ---------------------------------------------------------------------------
// assign: evaluate expression into destination buffer via parallel_for.
// If the destination pointer appears in the expression tree (aliasing),
// stage through a temporary Kokkos::View to avoid data races (D-ET3).
// ---------------------------------------------------------------------------

template <typename Expr>
void assign(real* dst, int n, Expr expr)
{
    if (contains_ptr(expr, dst)) {
        // Alias detected: evaluate into temporary, then copy back.
        Kokkos::View<real*, memory_space> tmp("expr_tmp", n);
        real* tmp_ptr = tmp.data();
        Kokkos::parallel_for(
            Kokkos::RangePolicy<execution_space>(0, n),
            KOKKOS_LAMBDA(int i) { tmp_ptr[i] = expr(i); });
        Kokkos::View<real*, memory_space, Kokkos::MemoryUnmanaged> dst_um(dst, n);
        Kokkos::deep_copy(dst_um, tmp);
    } else {
        Kokkos::parallel_for(
            Kokkos::RangePolicy<execution_space>(0, n),
            KOKKOS_LAMBDA(int i) { dst[i] = expr(i); });
    }
}

// ---------------------------------------------------------------------------
// scalar_expr: expression wrapper for all 4 buffers of a scalar field (D-ET5).
// ---------------------------------------------------------------------------

template <typename Expr>
struct scalar_expr {
    std::array<Expr, 4> exprs; // [0]=D, [1]=Rx, [2]=Ry, [3]=Rz
    std::array<int, 4> sizes;  // buffer lengths, same index order
};

// ---------------------------------------------------------------------------
// bind_scalar: extract pointers from registry into scalar_expr<handle_expr>.
// ---------------------------------------------------------------------------

template <int MS, int MaxS, int MaxV>
scalar_expr<handle_expr> bind_scalar(field_registry<MS, MaxS, MaxV>& reg,
                                     field_ref ref,
                                     scalar_handle sh)
{
    auto bufs = sh.all();
    scalar_expr<handle_expr> result{};
    for (int i = 0; i < 4; ++i) {
        result.exprs[i] = handle_expr{reg.data(ref, bufs[i])};
        result.sizes[i] = reg.size(ref, bufs[i]);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Binary operators for scalar_expr (found via ADL).
// ---------------------------------------------------------------------------

template <typename E1, typename E2>
auto operator+(scalar_expr<E1> a, scalar_expr<E2> b)
    -> scalar_expr<binary_expr<std::plus<>, E1, E2>>
{
    scalar_expr<binary_expr<std::plus<>, E1, E2>> result{};
    for (int i = 0; i < 4; ++i) {
        result.exprs[i] = {std::plus<>{}, a.exprs[i], b.exprs[i]};
        result.sizes[i] = a.sizes[i];
    }
    return result;
}

template <typename E1, typename E2>
auto operator-(scalar_expr<E1> a, scalar_expr<E2> b)
    -> scalar_expr<binary_expr<std::minus<>, E1, E2>>
{
    scalar_expr<binary_expr<std::minus<>, E1, E2>> result{};
    for (int i = 0; i < 4; ++i) {
        result.exprs[i] = {std::minus<>{}, a.exprs[i], b.exprs[i]};
        result.sizes[i] = a.sizes[i];
    }
    return result;
}

template <typename E1, typename E2>
auto operator*(scalar_expr<E1> a, scalar_expr<E2> b)
    -> scalar_expr<binary_expr<std::multiplies<>, E1, E2>>
{
    scalar_expr<binary_expr<std::multiplies<>, E1, E2>> result{};
    for (int i = 0; i < 4; ++i) {
        result.exprs[i] = {std::multiplies<>{}, a.exprs[i], b.exprs[i]};
        result.sizes[i] = a.sizes[i];
    }
    return result;
}

template <typename E1, typename E2>
auto operator/(scalar_expr<E1> a, scalar_expr<E2> b)
    -> scalar_expr<binary_expr<std::divides<>, E1, E2>>
{
    scalar_expr<binary_expr<std::divides<>, E1, E2>> result{};
    for (int i = 0; i < 4; ++i) {
        result.exprs[i] = {std::divides<>{}, a.exprs[i], b.exprs[i]};
        result.sizes[i] = a.sizes[i];
    }
    return result;
}

// ---------------------------------------------------------------------------
// Scalar-right operators: scalar_expr<E> op real.
// ---------------------------------------------------------------------------

template <typename E>
auto operator+(scalar_expr<E> a, real v)
    -> scalar_expr<binary_expr<std::plus<>, E, scalar_literal_expr>>
{
    scalar_expr<binary_expr<std::plus<>, E, scalar_literal_expr>> result{};
    for (int i = 0; i < 4; ++i) {
        result.exprs[i] = {std::plus<>{}, a.exprs[i], scalar_literal_expr{v}};
        result.sizes[i] = a.sizes[i];
    }
    return result;
}

template <typename E>
auto operator-(scalar_expr<E> a, real v)
    -> scalar_expr<binary_expr<std::minus<>, E, scalar_literal_expr>>
{
    scalar_expr<binary_expr<std::minus<>, E, scalar_literal_expr>> result{};
    for (int i = 0; i < 4; ++i) {
        result.exprs[i] = {std::minus<>{}, a.exprs[i], scalar_literal_expr{v}};
        result.sizes[i] = a.sizes[i];
    }
    return result;
}

template <typename E>
auto operator*(scalar_expr<E> a, real v)
    -> scalar_expr<binary_expr<std::multiplies<>, E, scalar_literal_expr>>
{
    scalar_expr<binary_expr<std::multiplies<>, E, scalar_literal_expr>> result{};
    for (int i = 0; i < 4; ++i) {
        result.exprs[i] = {std::multiplies<>{}, a.exprs[i], scalar_literal_expr{v}};
        result.sizes[i] = a.sizes[i];
    }
    return result;
}

template <typename E>
auto operator/(scalar_expr<E> a, real v)
    -> scalar_expr<binary_expr<std::divides<>, E, scalar_literal_expr>>
{
    scalar_expr<binary_expr<std::divides<>, E, scalar_literal_expr>> result{};
    for (int i = 0; i < 4; ++i) {
        result.exprs[i] = {std::divides<>{}, a.exprs[i], scalar_literal_expr{v}};
        result.sizes[i] = a.sizes[i];
    }
    return result;
}

// ---------------------------------------------------------------------------
// Scalar-left operators: real op scalar_expr<E>.
// Commutative ops delegate to scalar-right; non-commutative build scalar on left.
// ---------------------------------------------------------------------------

template <typename E>
auto operator+(real v, scalar_expr<E> a)
{
    return a + v;
}

template <typename E>
auto operator*(real v, scalar_expr<E> a)
{
    return a * v;
}

template <typename E>
auto operator-(real v, scalar_expr<E> a)
    -> scalar_expr<binary_expr<std::minus<>, scalar_literal_expr, E>>
{
    scalar_expr<binary_expr<std::minus<>, scalar_literal_expr, E>> result{};
    for (int i = 0; i < 4; ++i) {
        result.exprs[i] = {std::minus<>{}, scalar_literal_expr{v}, a.exprs[i]};
        result.sizes[i] = a.sizes[i];
    }
    return result;
}

template <typename E>
auto operator/(real v, scalar_expr<E> a)
    -> scalar_expr<binary_expr<std::divides<>, scalar_literal_expr, E>>
{
    scalar_expr<binary_expr<std::divides<>, scalar_literal_expr, E>> result{};
    for (int i = 0; i < 4; ++i) {
        result.exprs[i] = {std::divides<>{}, scalar_literal_expr{v}, a.exprs[i]};
        result.sizes[i] = a.sizes[i];
    }
    return result;
}

// ---------------------------------------------------------------------------
// Reductions: parallel_reduce over raw buffers and expressions (D-ET4).
// Identity values match Kokkos::reduction_identity defaults.
// ---------------------------------------------------------------------------

inline real reduce_max(const real* data, int n)
{
    real result = std::numeric_limits<real>::lowest();
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(int i, real& val) {
            if (data[i] > val) val = data[i];
        },
        Kokkos::Max<real>(result));
    return result;
}

inline real reduce_min(const real* data, int n)
{
    real result = std::numeric_limits<real>::max();
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(int i, real& val) {
            if (data[i] < val) val = data[i];
        },
        Kokkos::Min<real>(result));
    return result;
}

inline real reduce_sum(const real* data, int n)
{
    real result = 0.0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(int i, real& val) { val += data[i]; },
        result);
    return result;
}

template <typename Expr>
    requires(!std::is_pointer_v<Expr>)
real reduce_max(Expr expr, int n)
{
    real result = std::numeric_limits<real>::lowest();
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(int i, real& val) {
            real v = expr(i);
            if (v > val) val = v;
        },
        Kokkos::Max<real>(result));
    return result;
}

template <typename Expr>
    requires(!std::is_pointer_v<Expr>)
real reduce_sum(Expr expr, int n)
{
    real result = 0.0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(int i, real& val) { val += expr(i); },
        result);
    return result;
}

// ---------------------------------------------------------------------------
// assign_scalar: evaluate scalar_expr into all 4 registry buffers.
// ---------------------------------------------------------------------------

template <int MS, int MaxS, int MaxV, typename Expr>
void assign_scalar(field_registry<MS, MaxS, MaxV>& reg,
                   field_ref ref,
                   scalar_handle sh,
                   const scalar_expr<Expr>& expr)
{
    auto bufs = sh.all();
    for (int i = 0; i < 4; ++i) {
        assign(reg.data(ref, bufs[i]), expr.sizes[i], expr.exprs[i]);
    }
}

// ---------------------------------------------------------------------------
// Mutating operators (+=, -=, *=, /=): no aliasing check needed (D-ET3).
// Element-wise compound-assign is always safe since each thread accesses
// only dst[i].
// ---------------------------------------------------------------------------

template <typename Expr>
void plus_assign(real* dst, int n, Expr expr)
{
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(int i) { dst[i] += expr(i); });
}

template <typename Expr>
void minus_assign(real* dst, int n, Expr expr)
{
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(int i) { dst[i] -= expr(i); });
}

template <typename Expr>
void times_assign(real* dst, int n, Expr expr)
{
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(int i) { dst[i] *= expr(i); });
}

template <typename Expr>
void divide_assign(real* dst, int n, Expr expr)
{
    Kokkos::parallel_for(
        Kokkos::RangePolicy<execution_space>(0, n),
        KOKKOS_LAMBDA(int i) { dst[i] /= expr(i); });
}

// ---------------------------------------------------------------------------
// Scalar-level mutating operators.
// ---------------------------------------------------------------------------

template <int MS, int MaxS, int MaxV, typename Expr>
void plus_assign_scalar(field_registry<MS, MaxS, MaxV>& reg,
                        field_ref ref,
                        scalar_handle sh,
                        const scalar_expr<Expr>& expr)
{
    auto bufs = sh.all();
    for (int i = 0; i < 4; ++i) {
        plus_assign(reg.data(ref, bufs[i]), expr.sizes[i], expr.exprs[i]);
    }
}

template <int MS, int MaxS, int MaxV>
void times_assign_scalar(field_registry<MS, MaxS, MaxV>& reg,
                         field_ref ref,
                         scalar_handle sh,
                         real value)
{
    auto bufs = sh.all();
    for (int i = 0; i < 4; ++i) {
        times_assign(
            reg.data(ref, bufs[i]),
            reg.size(ref, bufs[i]),
            scalar_literal_expr{value});
    }
}

} // namespace ccs

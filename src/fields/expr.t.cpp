#include "fields/expr.hpp"
#include "fields/field_registry.hpp"
#include "fields/handle.hpp"

#include <functional>
#include <limits>

#include <Kokkos_Core.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace ccs;

// ---------------------------------------------------------------------------
// Custom main: Kokkos must be initialized before any test allocates Views.
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard kokkos(argc, argv);
    return Catch::Session().run(argc, argv);
}

// ---------------------------------------------------------------------------
// 10.1a — Expression node type tests
// ---------------------------------------------------------------------------

TEST_CASE("handle_expr reads from pointer")
{
    real data[] = {10.0, 20.0, 30.0};
    handle_expr e{data};
    REQUIRE(e(0) == 10.0);
    REQUIRE(e(1) == 20.0);
    REQUIRE(e(2) == 30.0);
}

TEST_CASE("scalar_literal_expr returns constant for any index")
{
    scalar_literal_expr e{3.14};
    REQUIRE(e(0) == 3.14);
    REQUIRE(e(1) == 3.14);
    REQUIRE(e(99) == 3.14);
}

TEST_CASE("binary_expr with std::plus")
{
    real a[] = {1.0, 2.0, 3.0};
    real b[] = {10.0, 20.0, 30.0};
    binary_expr expr{std::plus<>{}, handle_expr{a}, handle_expr{b}};
    REQUIRE(expr(0) == 11.0);
    REQUIRE(expr(1) == 22.0);
    REQUIRE(expr(2) == 33.0);
}

TEST_CASE("unary_expr with std::negate")
{
    real a[] = {1.0, -2.0, 3.0};
    unary_expr expr{std::negate<>{}, handle_expr{a}};
    REQUIRE(expr(0) == -1.0);
    REQUIRE(expr(1) == 2.0);
    REQUIRE(expr(2) == -3.0);
}

TEST_CASE("nested expression: (a + b) * c")
{
    real a[] = {1.0, 2.0, 3.0};
    real b[] = {10.0, 20.0, 30.0};
    auto sum = binary_expr{std::plus<>{}, handle_expr{a}, handle_expr{b}};
    auto expr = binary_expr{std::multiplies<>{}, sum, scalar_literal_expr{2.0}};
    REQUIRE(expr(0) == 22.0);
    REQUIRE(expr(1) == 44.0);
    REQUIRE(expr(2) == 66.0);
}

TEST_CASE("contains_ptr detects aliasing")
{
    real a[] = {1.0};
    real b[] = {2.0};

    REQUIRE(contains_ptr(handle_expr{a}, a));
    REQUIRE_FALSE(contains_ptr(handle_expr{a}, b));
    REQUIRE_FALSE(contains_ptr(scalar_literal_expr{1.0}, a));

    auto sum = binary_expr{std::plus<>{}, handle_expr{a}, handle_expr{b}};
    REQUIRE(contains_ptr(sum, a));
    REQUIRE(contains_ptr(sum, b));
    real c[] = {3.0};
    REQUIRE_FALSE(contains_ptr(sum, c));

    auto neg = unary_expr{std::negate<>{}, handle_expr{a}};
    REQUIRE(contains_ptr(neg, a));
    REQUIRE_FALSE(contains_ptr(neg, b));
}

// ---------------------------------------------------------------------------
// 10.2a — assign() tests
// ---------------------------------------------------------------------------

TEST_CASE("assign with binary expression")
{
    constexpr int n = 100;
    Kokkos::View<real*, memory_space> a("a", n);
    Kokkos::View<real*, memory_space> b("b", n);
    Kokkos::View<real*, memory_space> dst("dst", n);

    for (int i = 0; i < n; ++i) {
        a(i) = static_cast<real>(i);
        b(i) = static_cast<real>(i * 10);
    }

    assign(dst.data(), n,
           binary_expr{std::plus<>{}, handle_expr{a.data()}, handle_expr{b.data()}});

    for (int i = 0; i < n; ++i) {
        REQUIRE(dst(i) == static_cast<real>(i + i * 10));
    }
}

TEST_CASE("assign with scalar_literal_expr")
{
    constexpr int n = 50;
    Kokkos::View<real*, memory_space> dst("dst", n);

    assign(dst.data(), n, scalar_literal_expr{7.0});

    for (int i = 0; i < n; ++i) {
        REQUIRE(dst(i) == 7.0);
    }
}

TEST_CASE("assign detects aliasing and stages through temporary")
{
    constexpr int n = 100;
    Kokkos::View<real*, memory_space> a("a", n);
    for (int i = 0; i < n; ++i) a(i) = static_cast<real>(i + 1);

    // dst[i] = dst[i] + dst[i] — aliased, must stage through temporary.
    assign(a.data(), n,
           binary_expr{std::plus<>{}, handle_expr{a.data()}, handle_expr{a.data()}});

    for (int i = 0; i < n; ++i) {
        REQUIRE(a(i) == static_cast<real>(2 * (i + 1)));
    }
}

TEST_CASE("assign without aliasing copies directly")
{
    constexpr int n = 100;
    Kokkos::View<real*, memory_space> src("src", n);
    Kokkos::View<real*, memory_space> dst("dst", n);

    for (int i = 0; i < n; ++i) {
        src(i) = static_cast<real>(i * 3);
    }

    assign(dst.data(), n, handle_expr{src.data()});

    for (int i = 0; i < n; ++i) {
        REQUIRE(dst(i) == static_cast<real>(i * 3));
    }
}

// ---------------------------------------------------------------------------
// 10.3a — scalar expression operator tests
// ---------------------------------------------------------------------------

TEST_CASE("bind_scalar extracts pointers and sizes from registry")
{
    field_registry<2, 1, 0> reg;
    auto ref0 = reg.allocate_scalar(0, 0, 100, 5, 3, 2);
    constexpr auto sh = scalar_handle{0};

    // Fill buffers with known values.
    for (int i = 0; i < 100; ++i) reg.view(ref0, sh.D())(i) = static_cast<real>(i);
    for (int i = 0; i < 5; ++i) reg.view(ref0, sh.Rx())(i) = static_cast<real>(i * 10);
    for (int i = 0; i < 3; ++i) reg.view(ref0, sh.Ry())(i) = static_cast<real>(i * 100);
    for (int i = 0; i < 2; ++i) reg.view(ref0, sh.Rz())(i) = static_cast<real>(i * 1000);

    auto bound = bind_scalar(reg, ref0, sh);

    // Check pointers match registry.
    REQUIRE(bound.exprs[0].ptr == reg.data(ref0, sh.D()));
    REQUIRE(bound.exprs[1].ptr == reg.data(ref0, sh.Rx()));
    REQUIRE(bound.exprs[2].ptr == reg.data(ref0, sh.Ry()));
    REQUIRE(bound.exprs[3].ptr == reg.data(ref0, sh.Rz()));

    // Check sizes.
    REQUIRE(bound.sizes[0] == 100);
    REQUIRE(bound.sizes[1] == 5);
    REQUIRE(bound.sizes[2] == 3);
    REQUIRE(bound.sizes[3] == 2);

    // Check values through expressions.
    REQUIRE(bound.exprs[0](0) == 0.0);
    REQUIRE(bound.exprs[0](99) == 99.0);
    REQUIRE(bound.exprs[1](2) == 20.0);
}

TEST_CASE("scalar_expr operator+ combines two scalar expressions")
{
    field_registry<2, 1, 0> reg;
    auto ref0 = reg.allocate_scalar(0, 0, 100, 5, 3, 2);
    auto ref1 = reg.allocate_scalar(1, 0, 100, 5, 3, 2);
    constexpr auto sh = scalar_handle{0};

    for (int i = 0; i < 100; ++i) reg.view(ref0, sh.D())(i) = static_cast<real>(i);
    for (int i = 0; i < 5; ++i) reg.view(ref0, sh.Rx())(i) = static_cast<real>(i);
    for (int i = 0; i < 100; ++i) reg.view(ref1, sh.D())(i) = static_cast<real>(i * 10);
    for (int i = 0; i < 5; ++i) reg.view(ref1, sh.Rx())(i) = static_cast<real>(i * 10);

    auto a = bind_scalar(reg, ref0, sh);
    auto b = bind_scalar(reg, ref1, sh);
    auto result = a + b;

    // Type check.
    static_assert(std::is_same_v<decltype(result),
                                 scalar_expr<binary_expr<std::plus<>, handle_expr, handle_expr>>>);

    // Value checks for D buffer.
    REQUIRE(result.exprs[0](0) == 0.0);
    REQUIRE(result.exprs[0](1) == 11.0);
    REQUIRE(result.exprs[0](50) == 550.0);

    // Value check for Rx buffer.
    REQUIRE(result.exprs[1](2) == 22.0);

    // Sizes preserved.
    REQUIRE(result.sizes[0] == 100);
    REQUIRE(result.sizes[1] == 5);
}

TEST_CASE("scalar_expr operator* with real (scalar-right)")
{
    field_registry<2, 1, 0> reg;
    auto ref0 = reg.allocate_scalar(0, 0, 100, 5, 3, 2);
    constexpr auto sh = scalar_handle{0};

    for (int i = 0; i < 100; ++i) reg.view(ref0, sh.D())(i) = static_cast<real>(i);

    auto a = bind_scalar(reg, ref0, sh);
    auto result = a * 2.0;

    REQUIRE(result.exprs[0](0) == 0.0);
    REQUIRE(result.exprs[0](1) == 2.0);
    REQUIRE(result.exprs[0](10) == 20.0);
    REQUIRE(result.sizes[0] == 100);
}

TEST_CASE("scalar_expr operator* with real (scalar-left)")
{
    field_registry<2, 1, 0> reg;
    auto ref0 = reg.allocate_scalar(0, 0, 100, 5, 3, 2);
    constexpr auto sh = scalar_handle{0};

    for (int i = 0; i < 100; ++i) reg.view(ref0, sh.D())(i) = static_cast<real>(i);

    auto a = bind_scalar(reg, ref0, sh);
    auto result = 2.0 * a;

    REQUIRE(result.exprs[0](0) == 0.0);
    REQUIRE(result.exprs[0](1) == 2.0);
    REQUIRE(result.exprs[0](10) == 20.0);
    REQUIRE(result.sizes[0] == 100);
}

TEST_CASE("assign_scalar materializes expression into registry buffers")
{
    field_registry<3, 1, 0> reg;
    auto ref0 = reg.allocate_scalar(0, 0, 100, 5, 3, 2);
    auto ref1 = reg.allocate_scalar(1, 0, 100, 5, 3, 2);
    auto dst_ref = reg.allocate_scalar(2, 0, 100, 5, 3, 2);
    constexpr auto sh = scalar_handle{0};

    // Fill slot 0.
    for (int i = 0; i < 100; ++i) reg.view(ref0, sh.D())(i) = static_cast<real>(i);
    for (int i = 0; i < 5; ++i) reg.view(ref0, sh.Rx())(i) = static_cast<real>(i * 2);
    for (int i = 0; i < 3; ++i) reg.view(ref0, sh.Ry())(i) = static_cast<real>(i * 3);
    for (int i = 0; i < 2; ++i) reg.view(ref0, sh.Rz())(i) = static_cast<real>(i * 4);

    // Fill slot 1.
    for (int i = 0; i < 100; ++i) reg.view(ref1, sh.D())(i) = static_cast<real>(i * 10);
    for (int i = 0; i < 5; ++i) reg.view(ref1, sh.Rx())(i) = static_cast<real>(i * 20);
    for (int i = 0; i < 3; ++i) reg.view(ref1, sh.Ry())(i) = static_cast<real>(i * 30);
    for (int i = 0; i < 2; ++i) reg.view(ref1, sh.Rz())(i) = static_cast<real>(i * 40);

    auto a = bind_scalar(reg, ref0, sh);
    auto b = bind_scalar(reg, ref1, sh);
    assign_scalar(reg, dst_ref, sh, a + b);

    // Verify D buffer (size 100).
    for (int i = 0; i < 100; ++i) {
        REQUIRE(reg.view(dst_ref, sh.D())(i) == static_cast<real>(i + i * 10));
    }
    // Verify Rx buffer (size 5).
    for (int i = 0; i < 5; ++i) {
        REQUIRE(reg.view(dst_ref, sh.Rx())(i) == static_cast<real>(i * 2 + i * 20));
    }
    // Verify Ry buffer (size 3).
    for (int i = 0; i < 3; ++i) {
        REQUIRE(reg.view(dst_ref, sh.Ry())(i) == static_cast<real>(i * 3 + i * 30));
    }
    // Verify Rz buffer (size 2).
    for (int i = 0; i < 2; ++i) {
        REQUIRE(reg.view(dst_ref, sh.Rz())(i) == static_cast<real>(i * 4 + i * 40));
    }
}

// ---------------------------------------------------------------------------
// 10.3c — Non-commutative scalar-left operator tests
// ---------------------------------------------------------------------------

TEST_CASE("scalar-left subtraction: real - scalar_expr")
{
    field_registry<2, 1, 0> reg;
    auto ref0 = reg.allocate_scalar(0, 0, 100, 5, 3, 2);
    constexpr auto sh = scalar_handle{0};

    // Fill D buffer with known values: 1, 2, 3, ...
    for (int i = 0; i < 100; ++i)
        reg.view(ref0, sh.D())(i) = static_cast<real>(i + 1);
    // Fill Rx buffer.
    for (int i = 0; i < 5; ++i)
        reg.view(ref0, sh.Rx())(i) = static_cast<real>(i + 1);

    auto a = bind_scalar(reg, ref0, sh);
    auto result = 5.0 - a;

    // Verify result is 5.0 - a[i], NOT a[i] - 5.0.
    REQUIRE(result.exprs[0](0) == 5.0 - 1.0);  // 4.0
    REQUIRE(result.exprs[0](1) == 5.0 - 2.0);  // 3.0
    REQUIRE(result.exprs[0](9) == 5.0 - 10.0); // -5.0
    REQUIRE(result.sizes[0] == 100);

    // Also verify Rx buffer.
    REQUIRE(result.exprs[1](0) == 5.0 - 1.0);
    REQUIRE(result.exprs[1](4) == 5.0 - 5.0);
    REQUIRE(result.sizes[1] == 5);
}

TEST_CASE("scalar-left division: real / scalar_expr")
{
    field_registry<2, 1, 0> reg;
    auto ref0 = reg.allocate_scalar(0, 0, 100, 5, 3, 2);
    constexpr auto sh = scalar_handle{0};

    // Fill D buffer with nonzero values: 1, 2, 3, ...
    for (int i = 0; i < 100; ++i)
        reg.view(ref0, sh.D())(i) = static_cast<real>(i + 1);
    // Fill Rx buffer.
    for (int i = 0; i < 5; ++i)
        reg.view(ref0, sh.Rx())(i) = static_cast<real>(i + 1);

    auto a = bind_scalar(reg, ref0, sh);
    auto result = 10.0 / a;

    // Verify result is 10.0 / a[i], NOT a[i] / 10.0.
    REQUIRE(result.exprs[0](0) == 10.0 / 1.0);  // 10.0
    REQUIRE(result.exprs[0](1) == 10.0 / 2.0);  // 5.0
    REQUIRE(result.exprs[0](3) == 10.0 / 4.0);  // 2.5
    REQUIRE(result.exprs[0](9) == 10.0 / 10.0); // 1.0
    REQUIRE(result.sizes[0] == 100);

    // Also verify Rx buffer.
    REQUIRE(result.exprs[1](0) == 10.0 / 1.0);
    REQUIRE(result.exprs[1](4) == 10.0 / 5.0);
    REQUIRE(result.sizes[1] == 5);
}

// ---------------------------------------------------------------------------
// 10.4a — Reduction operation tests
// ---------------------------------------------------------------------------

TEST_CASE("reduce_max over raw buffer")
{
    constexpr int n = 50;
    Kokkos::View<real*, memory_space> buf("buf", n);
    auto* ptr = buf.data();
    // Fill with {1, 2, ..., n}.
    for (int i = 0; i < n; ++i) ptr[i] = static_cast<real>(i + 1);

    REQUIRE(reduce_max(ptr, n) == static_cast<real>(n));
}

TEST_CASE("reduce_min over raw buffer")
{
    constexpr int n = 50;
    Kokkos::View<real*, memory_space> buf("buf", n);
    auto* ptr = buf.data();
    for (int i = 0; i < n; ++i) ptr[i] = static_cast<real>(i + 1);

    REQUIRE(reduce_min(ptr, n) == 1.0);
}

TEST_CASE("reduce_sum over raw buffer")
{
    constexpr int n = 50;
    Kokkos::View<real*, memory_space> buf("buf", n);
    auto* ptr = buf.data();
    for (int i = 0; i < n; ++i) ptr[i] = static_cast<real>(i + 1);

    REQUIRE(reduce_sum(ptr, n) == static_cast<real>(n * (n + 1) / 2));
}

TEST_CASE("reduce_max over expression (no materialization)")
{
    constexpr int n = 30;
    Kokkos::View<real*, memory_space> a_buf("a", n);
    Kokkos::View<real*, memory_space> b_buf("b", n);
    auto* a = a_buf.data();
    auto* b = b_buf.data();
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<real>(i);
        b[i] = static_cast<real>(n - i);
    }
    // a[i] + b[i] == n for all i.
    auto expr = binary_expr{std::plus<>{}, handle_expr{a}, handle_expr{b}};
    REQUIRE(reduce_max(expr, n) == static_cast<real>(n));
}

TEST_CASE("reduce_max with n == 0 returns identity")
{
    real dummy = 42.0;
    REQUIRE(reduce_max(&dummy, 0) == std::numeric_limits<real>::lowest());
}

TEST_CASE("reduce_min with n == 0 returns identity")
{
    real dummy = 42.0;
    REQUIRE(reduce_min(&dummy, 0) == std::numeric_limits<real>::max());
}

TEST_CASE("reduce_sum with n == 0 returns identity")
{
    real dummy = 42.0;
    REQUIRE(reduce_sum(&dummy, 0) == 0.0);
}

// ---------------------------------------------------------------------------
// 10.5a — Mutating operator tests
// ---------------------------------------------------------------------------

TEST_CASE("plus_assign: dst[i] += src[i]")
{
    constexpr int n = 40;
    Kokkos::View<real*, memory_space> dst_buf("dst", n);
    Kokkos::View<real*, memory_space> src_buf("src", n);
    auto* dst = dst_buf.data();
    auto* src = src_buf.data();
    for (int i = 0; i < n; ++i) {
        dst[i] = static_cast<real>(i);
        src[i] = static_cast<real>(i * 10);
    }

    plus_assign(dst, n, handle_expr{src});

    for (int i = 0; i < n; ++i) {
        REQUIRE(dst[i] == static_cast<real>(i + i * 10));
    }
}

TEST_CASE("minus_assign: dst[i] -= src[i]")
{
    constexpr int n = 40;
    Kokkos::View<real*, memory_space> dst_buf("dst", n);
    Kokkos::View<real*, memory_space> src_buf("src", n);
    auto* dst = dst_buf.data();
    auto* src = src_buf.data();
    for (int i = 0; i < n; ++i) {
        dst[i] = static_cast<real>(i * 10);
        src[i] = static_cast<real>(i);
    }

    minus_assign(dst, n, handle_expr{src});

    for (int i = 0; i < n; ++i) {
        REQUIRE(dst[i] == static_cast<real>(i * 10 - i));
    }
}

TEST_CASE("times_assign: dst[i] *= 2.0")
{
    constexpr int n = 40;
    Kokkos::View<real*, memory_space> dst_buf("dst", n);
    auto* dst = dst_buf.data();
    for (int i = 0; i < n; ++i) dst[i] = static_cast<real>(i + 1);

    times_assign(dst, n, scalar_literal_expr{2.0});

    for (int i = 0; i < n; ++i) {
        REQUIRE(dst[i] == static_cast<real>((i + 1) * 2));
    }
}

TEST_CASE("divide_assign: dst[i] /= 2.0")
{
    constexpr int n = 40;
    Kokkos::View<real*, memory_space> dst_buf("dst", n);
    auto* dst = dst_buf.data();
    for (int i = 0; i < n; ++i) dst[i] = static_cast<real>((i + 1) * 4);

    divide_assign(dst, n, scalar_literal_expr{2.0});

    for (int i = 0; i < n; ++i) {
        REQUIRE(dst[i] == static_cast<real>((i + 1) * 2));
    }
}

TEST_CASE("plus_assign aliasing safety: dst[i] += dst[i]")
{
    constexpr int n = 40;
    Kokkos::View<real*, memory_space> buf("buf", n);
    auto* a = buf.data();
    for (int i = 0; i < n; ++i) a[i] = static_cast<real>(i + 1);

    plus_assign(a, n, handle_expr{a});

    for (int i = 0; i < n; ++i) {
        REQUIRE(a[i] == static_cast<real>(2 * (i + 1)));
    }
}

TEST_CASE("plus_assign_scalar: all 4 buffers get += b")
{
    field_registry<3, 1, 0> reg;
    auto ref_a = reg.allocate_scalar(0, 0, 100, 5, 3, 2);
    auto ref_b = reg.allocate_scalar(1, 0, 100, 5, 3, 2);
    constexpr auto sh = scalar_handle{0};

    // Fill a and b.
    for (int i = 0; i < 100; ++i) {
        reg.view(ref_a, sh.D())(i) = static_cast<real>(i);
        reg.view(ref_b, sh.D())(i) = static_cast<real>(i * 10);
    }
    for (int i = 0; i < 5; ++i) {
        reg.view(ref_a, sh.Rx())(i) = static_cast<real>(i * 2);
        reg.view(ref_b, sh.Rx())(i) = static_cast<real>(i * 20);
    }
    for (int i = 0; i < 3; ++i) {
        reg.view(ref_a, sh.Ry())(i) = static_cast<real>(i * 3);
        reg.view(ref_b, sh.Ry())(i) = static_cast<real>(i * 30);
    }
    for (int i = 0; i < 2; ++i) {
        reg.view(ref_a, sh.Rz())(i) = static_cast<real>(i * 4);
        reg.view(ref_b, sh.Rz())(i) = static_cast<real>(i * 40);
    }

    auto bound_b = bind_scalar(reg, ref_b, sh);
    plus_assign_scalar(reg, ref_a, sh, bound_b);

    for (int i = 0; i < 100; ++i)
        REQUIRE(reg.view(ref_a, sh.D())(i) == static_cast<real>(i + i * 10));
    for (int i = 0; i < 5; ++i)
        REQUIRE(reg.view(ref_a, sh.Rx())(i) == static_cast<real>(i * 2 + i * 20));
    for (int i = 0; i < 3; ++i)
        REQUIRE(reg.view(ref_a, sh.Ry())(i) == static_cast<real>(i * 3 + i * 30));
    for (int i = 0; i < 2; ++i)
        REQUIRE(reg.view(ref_a, sh.Rz())(i) == static_cast<real>(i * 4 + i * 40));
}

TEST_CASE("times_assign_scalar: all 4 buffers *= constant")
{
    field_registry<2, 1, 0> reg;
    auto ref0 = reg.allocate_scalar(0, 0, 100, 5, 3, 2);
    constexpr auto sh = scalar_handle{0};

    for (int i = 0; i < 100; ++i) reg.view(ref0, sh.D())(i) = static_cast<real>(i + 1);
    for (int i = 0; i < 5; ++i) reg.view(ref0, sh.Rx())(i) = static_cast<real>(i + 1);
    for (int i = 0; i < 3; ++i) reg.view(ref0, sh.Ry())(i) = static_cast<real>(i + 1);
    for (int i = 0; i < 2; ++i) reg.view(ref0, sh.Rz())(i) = static_cast<real>(i + 1);

    times_assign_scalar(reg, ref0, sh, 3.14);

    for (int i = 0; i < 100; ++i)
        REQUIRE(reg.view(ref0, sh.D())(i) == static_cast<real>((i + 1) * 3.14));
    for (int i = 0; i < 5; ++i)
        REQUIRE(reg.view(ref0, sh.Rx())(i) == static_cast<real>((i + 1) * 3.14));
    for (int i = 0; i < 3; ++i)
        REQUIRE(reg.view(ref0, sh.Ry())(i) == static_cast<real>((i + 1) * 3.14));
    for (int i = 0; i < 2; ++i)
        REQUIRE(reg.view(ref0, sh.Rz())(i) == static_cast<real>((i + 1) * 3.14));
}

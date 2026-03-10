#include "vector.hpp"
#include "scalar.hpp"

#include "selector.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <ranges>
#include <tuple>
#include <vector>

using namespace ccs;
using namespace si;
using namespace vi;

constexpr auto plus = lift(std::plus{});

// Test-local helper: eager cartesian product as vector of tuples
template <typename A, typename B, typename C>
auto cartesian_product_vec(const A& a, const B& b, const C& c)
{
    using T = std::tuple<std::ranges::range_value_t<A>,
                         std::ranges::range_value_t<B>,
                         std::ranges::range_value_t<C>>;
    std::vector<T> result;
    for (const auto& ai : a)
        for (const auto& bj : b)
            for (const auto& ck : c)
                result.emplace_back(ai, bj, ck);
    return result;
}

// construct simple mesh geometry
namespace g
{

const auto x = std::vector<real>{0, 1, 2, 3, 4};
const auto y = std::vector<real>{-2, -1};
const auto z = std::vector<real>{6, 7, 8, 9};
const auto rx = std::vector<real3>{{0.5, -2, 6}, {1.5, -1, 9}};
const auto ry = std::vector<real3>{{1, -1.75, 7}, {4, -1.25, 7}, {3, -1.1, 9}};
const auto rz = std::vector<real3>{{0, -2, 6.1}};

const auto sloc = tuple{tuple{cartesian_product_vec(x, y, z)},
                        tuple{std::views::all(rx), std::views::all(ry), std::views::all(rz)}};
const auto loc = tuple{sloc, sloc, sloc};
} // namespace g

TEST_CASE("construction")
{
    using T = std::vector<int>;

    vector<T> v{tuple{tuple{std::views::iota(0, 10)},
                      tuple{std::views::iota(0, 3), std::views::iota(1, 3), std::views::iota(-1, 1)}},
                tuple{tuple{std::views::iota(10, 20)},
                      tuple{std::views::iota(1, 4), std::views::iota(2, 5), std::views::iota(2, 10)}},
                tuple{tuple{std::views::iota(20, 30)},
                      tuple{std::views::iota(1, 5), std::views::iota(5, 10), std::views::iota(8, 10)}}};

    auto&& [X, Y, Z] = v;
    REQUIRE(std::ranges::equal(get<0>(X), std::views::iota(0, 10)));
    REQUIRE(std::ranges::equal(get<0>(Y), std::views::iota(10, 20)));
    REQUIRE(std::ranges::equal(get<0>(Z), std::views::iota(20, 30)));

    REQUIRE(std::ranges::equal(get<Dx>(v), std::views::iota(0, 10)));
    REQUIRE(std::ranges::equal(get<xRx>(v), std::views::iota(0, 3)));
    REQUIRE(std::ranges::equal(get<xRy>(v), std::views::iota(1, 3)));
    REQUIRE(std::ranges::equal(get<xRz>(v), std::views::iota(-1, 1)));
    REQUIRE(std::ranges::equal(get<yRx>(v), std::views::iota(1, 4)));
    REQUIRE(std::ranges::equal(get<yRy>(v), std::views::iota(2, 5)));
    REQUIRE(std::ranges::equal(get<yRz>(v), std::views::iota(2, 10)));
    REQUIRE(std::ranges::equal(get<zRx>(v), std::views::iota(1, 5)));
    REQUIRE(std::ranges::equal(get<zRy>(v), std::views::iota(5, 10)));
    REQUIRE(std::ranges::equal(get<zRz>(v), std::views::iota(8, 10)));
}

TEST_CASE("conversion")
{
    using T = std::vector<int>;

    vector<T> v{tuple{tuple{std::views::iota(0, 10)},
                      tuple{std::views::iota(0, 3), std::views::iota(1, 3), std::views::iota(-1, 1)}},
                tuple{tuple{std::views::iota(10, 20)},
                      tuple{std::views::iota(1, 4), std::views::iota(2, 5), std::views::iota(2, 10)}},
                tuple{tuple{std::views::iota(20, 30)},
                      tuple{std::views::iota(1, 5), std::views::iota(5, 10), std::views::iota(8, 10)}}};

    vector<std::span<const int>> r = v;
    REQUIRE((r == v));

    // auto f = [](Simplevector<std::span<int>> x) {
    //     auto q = Simplevector<std::vector<int>>{
    //         &global::loc,
    //         tuple{std::vector{1, 2, 3}, std::vector{4, 5, 6}, std::vector{7, 8, 9}},
    //         tuple{std::vector{1}, std::vector{2, 3}, std::vector{4, 5, 6}}};
    //     x = 2 * q;
    // };

    // s = f;
    // REQUIRE(std::ranges::equal(s | sel::Dz, std::vector{14, 16, 18}));
    // REQUIRE(std::ranges::equal(s | sel::Rx, std::vector{2}));
    // REQUIRE(std::ranges::equal(s | sel::Ry, std::vector{4, 6}));
    // REQUIRE(std::ranges::equal(s | sel::Rz, std::vector{8, 10, 12}));
}

TEST_CASE("to scalar")
{
    using T = std::vector<real>;

    vector<T> v{tuple{tuple{std::views::iota(0, 10)},
                      tuple{std::views::iota(0, 3), std::views::iota(1, 3), std::views::iota(-1, 1)}},
                tuple{tuple{std::views::iota(10, 20)},
                      tuple{std::views::iota(1, 4), std::views::iota(2, 5), std::views::iota(2, 10)}},
                tuple{tuple{std::views::iota(20, 30)},
                      tuple{std::views::iota(1, 5), std::views::iota(5, 10), std::views::iota(8, 10)}}};

    {
        scalar_view sx = get<X>(v);
        REQUIRE((sx == get<0>(v)));
    }

    {
        scalar_span sx = get<X>(v);
        REQUIRE((sx == get<0>(v)));
        sx | sel::D = 1;
        REQUIRE(std::ranges::equal(get<Dx>(v), std::vector<real>(10, 1.0)));
    }

    {
        scalar_view sy = get<Y>(v);
        REQUIRE((sy == get<1>(v)));
    }

    {
        scalar_span sy = get<Y>(v);
        REQUIRE((sy == get<1>(v)));
        sy | sel::D = 1;
        REQUIRE(std::ranges::equal(get<Dy>(v), std::vector<real>(10, 1.0)));
    }

    {
        scalar_view sz = get<Z>(v);
        REQUIRE((sz == get<2>(v)));
    }

    {
        scalar_span sz = get<Z>(v);
        REQUIRE((sz == get<2>(v)));
        sz | sel::D = 1;
        REQUIRE(std::ranges::equal(get<Dz>(v), std::vector<real>(10, 1.0)));
    }

    {
        vector_view vvc = v;
        scalar_view sx = get<X>(vvc);
        REQUIRE((sx == get<0>(vvc)));
    }
}

TEST_CASE("selection")
{
    using T = std::vector<real>;

    vector<T> v_{tuple{tuple{std::views::iota(0, 10)},
                       tuple{std::views::iota(0, 3), std::views::iota(1, 3), std::views::iota(-1, 1)}},
                 tuple{tuple{std::views::iota(10, 20)},
                       tuple{std::views::iota(1, 4), std::views::iota(2, 5), std::views::iota(2, 10)}},
                 tuple{tuple{std::views::iota(20, 30)},
                       tuple{std::views::iota(1, 5), std::views::iota(5, 10), std::views::iota(8, 10)}}};
    vector_span v = v_;

    REQUIRE(std::ranges::equal(v | sel::Dx, std::views::iota(0, 10)));
    REQUIRE(std::ranges::equal(std::views::iota(10, 20), v | sel::Dy));
    REQUIRE(std::ranges::equal(std::views::iota(20, 30), v | sel::Dz));
    REQUIRE(std::ranges::equal(std::views::iota(0, 3), get<X>(v | sel::Rx)));
    REQUIRE(std::ranges::equal(std::views::iota(2, 5), get<Y>(v | sel::Ry)));
    REQUIRE(std::ranges::equal(std::views::iota(8, 10), get<Z>(v | sel::Rz)));

    // modify selection
    v | sel::D = 0;
    REQUIRE(std::ranges::equal(get<Dx>(v_), std::vector<real>(10, 0.0)));
    REQUIRE(std::ranges::equal(get<Dy>(v_), std::vector<real>(10, 0.0)));
    REQUIRE(std::ranges::equal(get<Dz>(v_), std::vector<real>(10, 0.0)));

    v | sel::Rz = -1;
    REQUIRE(std::ranges::equal(get<xRz>(v_), std::vector<real>(2, -1.0)));
    REQUIRE(std::ranges::equal(get<yRz>(v_), std::vector<real>(8, -1.0)));
    REQUIRE(std::ranges::equal(get<zRz>(v_), std::vector<real>(2, -1.0)));

    v | sel::R = 2;
    REQUIRE(std::ranges::equal(get<xRx>(v_), std::vector<real>(3, 2.0)));
    REQUIRE(std::ranges::equal(get<xRz>(v_), std::vector<real>(2, 2.0)));
    REQUIRE(std::ranges::equal(get<yRx>(v_), std::vector<real>(3, 2.0)));
}

TEST_CASE("math")
{
    using T = std::vector<int>;

    vector<T> v{tuple{tuple{std::views::iota(0, 10)},
                      tuple{std::views::iota(0, 3), std::views::iota(1, 3), std::views::iota(-1, 1)}},
                tuple{tuple{std::views::iota(10, 20)},
                      tuple{std::views::iota(1, 4), std::views::iota(2, 5), std::views::iota(2, 10)}},
                tuple{tuple{std::views::iota(20, 30)},
                      tuple{std::views::iota(1, 5), std::views::iota(5, 10), std::views::iota(8, 10)}}};

    auto q = v + 1;
    REQUIRE(std::ranges::equal(std::views::iota(1, 11), q | sel::Dx));
    REQUIRE(std::ranges::equal(std::views::iota(11, 21), q | sel::Dy));
    REQUIRE(std::ranges::equal(std::views::iota(21, 31), q | sel::Dz));
    REQUIRE(std::ranges::equal(std::views::iota(1, 4), get<xRx>(q)));
    REQUIRE(std::ranges::equal(std::views::iota(3, 6), get<yRy>(q)));
    REQUIRE(std::ranges::equal(std::views::iota(9, 11), get<zRz>(q)));

    auto r = q + v;

    REQUIRE(std::ranges::equal(plus(std::views::iota(1, 11), std::views::iota(0, 10)), r | sel::Dx));
    REQUIRE(std::ranges::equal(plus(std::views::iota(2, 6), std::views::iota(1, 5)), get<zRx>(r)));

    vector<T> t = r;
    REQUIRE(std::ranges::equal(t | sel::Dz, r | sel::Dz));
    REQUIRE(std::ranges::equal(t | sel::yRy, r | sel::yRy));

    vector<T> a{v};
    vector<std::span<int>> b = a;
    b = r;
    REQUIRE(std::ranges::equal(b | sel::Dy, r | sel::Dy));
    REQUIRE(std::ranges::equal(b | sel::zRx, r | sel::zRx));
}

TEST_CASE("lifting single arg")
{
    using T = std::vector<int>;

    vector<T> v{tuple{tuple{std::views::iota(0, 10)},
                      tuple{std::views::iota(0, 3), std::views::iota(1, 3), std::views::iota(-1, 1)}},
                tuple{tuple{std::views::iota(10, 20)},
                      tuple{std::views::iota(1, 4), std::views::iota(2, 5), std::views::iota(2, 10)}},
                tuple{tuple{std::views::iota(20, 30)},
                      tuple{std::views::iota(1, 5), std::views::iota(5, 10), std::views::iota(8, 10)}}};

    constexpr auto f = lift([](auto&& x) { return std::abs(x) + 1; });

    auto j = f(v);
    auto k = v + 1;

    REQUIRE(std::ranges::equal(j | sel::Dx, k | sel::Dx));
    REQUIRE(std::ranges::equal(j | sel::Dy, k | sel::Dy));
    REQUIRE(std::ranges::equal(j | sel::Dz, k | sel::Dz));
    REQUIRE(std::ranges::equal(j | sel::xRx, k | sel::xRx));
    REQUIRE(std::ranges::equal(j | sel::yRy, k | sel::yRy));
    REQUIRE(std::ranges::equal(j | sel::zRz, k | sel::zRz));
}

TEST_CASE("mesh location")
{
    using namespace g;

    using T = std::vector<real>;

    const auto sz = std::vector<real>(x.size() * y.size() * z.size(), 0.0);
    const auto x_sz = std::vector<real>(rx.size(), 0.0);
    const auto y_sz = std::vector<real>(ry.size(), 0.0);
    const auto z_sz = std::vector<real>(rz.size(), 0.0);

    vector<T> v{tuple{tuple{sz}, tuple{x_sz, y_sz, z_sz}},
                tuple{tuple{sz}, tuple{x_sz, y_sz, z_sz}},
                tuple{tuple{sz}, tuple{x_sz, y_sz, z_sz}}};

    constexpr auto t = std::views::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * z;
    });

    const auto sol = loc | t;

    v | sel::D = sol;
    REQUIRE(std::ranges::equal(v | sel::Dx, cartesian_product_vec(x, y, z) | t));

    v | sel::Rx = sol;
    REQUIRE(std::ranges::equal(v | sel::xRx, rx | t));
    REQUIRE(std::ranges::equal(v | sel::yRx, rx | t));

    v | sel::Ry = sol;
    REQUIRE(std::ranges::equal(v | sel::xRy, ry | t));
    REQUIRE(std::ranges::equal(v | sel::zRy, ry | t));

    v | sel::Rz = sol;
    REQUIRE(std::ranges::equal(v | sel::zRz, rz | t));

    constexpr auto u = std::views::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * y * z;
    });

    v | sel::R = loc | u;
    REQUIRE(std::ranges::equal(v | sel::xRx, rx | u));
    REQUIRE(std::ranges::equal(v | sel::yRy, ry | u));
    REQUIRE(std::ranges::equal(v | sel::zRz, rz | u));
}

template <auto I>
struct loc_fn {
    constexpr auto operator()(auto&& loc)
    {
        auto&& [x, y, z] = loc;
        return x * y * z * std::get<I>(loc);
    }
};

TEST_CASE("mesh location span")
{
    using namespace g;

    using T = std::vector<real>;

    const auto sz = std::vector<real>(x.size() * y.size() * z.size(), 0.0);
    const auto x_sz = std::vector<real>(rx.size(), 0.0);
    const auto y_sz = std::vector<real>(ry.size(), 0.0);
    const auto z_sz = std::vector<real>(rz.size(), 0.0);

    vector<T> v_{tuple{tuple{sz}, tuple{x_sz, y_sz, z_sz}},
                 tuple{tuple{sz}, tuple{x_sz, y_sz, z_sz}},
                 tuple{tuple{sz}, tuple{x_sz, y_sz, z_sz}}};
    vector_span v = v_;

    constexpr auto t = std::views::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * z;
    });

    const auto sol = loc | t;

    REQUIRE(Vector<decltype(sol)>);

    v | sel::D = sol;
    REQUIRE(std::ranges::equal(v | sel::Dx, cartesian_product_vec(x, y, z) | t));

    v | sel::Rx = sol;
    REQUIRE(std::ranges::equal(v | sel::xRx, rx | t));
    REQUIRE(std::ranges::equal(v | sel::yRx, rx | t));

    v | sel::Ry = sol;
    REQUIRE(std::ranges::equal(v | sel::xRy, ry | t));
    REQUIRE(std::ranges::equal(v | sel::zRy, ry | t));

    v | sel::Rz = sol;
    REQUIRE(std::ranges::equal(v | sel::zRz, rz | t));

    constexpr auto u = std::views::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * y * z;
    });

    v | sel::R = loc | u;
    REQUIRE(std::ranges::equal(v | sel::xRx, rx | u));
    REQUIRE(std::ranges::equal(v | sel::yRy, ry | u));
    REQUIRE(std::ranges::equal(v | sel::zRz, rz | u));

    REQUIRE(Vector<decltype(loc | tuple{std::views::transform(loc_fn<0>{}),
                                        std::views::transform(loc_fn<1>{}),
                                        std::views::transform(loc_fn<2>{})})>);

    v | sel::Rx = loc | sel::Rx |
                  tuple{std::views::transform(loc_fn<0>{}),
                        std::views::transform(loc_fn<1>{}),
                        std::views::transform(loc_fn<2>{})};

    REQUIRE(std::ranges::equal(v | sel::xRx, rx | std::views::transform(loc_fn<0>{})));
    REQUIRE(std::ranges::equal(v | sel::yRx, rx | std::views::transform(loc_fn<1>{})));
    REQUIRE(std::ranges::equal(v | sel::zRx, rx | std::views::transform(loc_fn<2>{})));
}

#include "selector.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <ranges>
#include <tuple>
#include <vector>

using namespace ccs;
using namespace si;

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

const auto loc = tuple{tuple{cartesian_product_vec(x, y, z)},
                       tuple{std::views::all(rx), std::views::all(ry), std::views::all(rz)}};
} // namespace g

TEST_CASE("construction")
{
    using T = std::vector<int>;
    scalar<T> s{tuple{std::views::iota(0, 10)},
                tuple{std::views::iota(0, 3), std::views::iota(3, 6), std::views::iota(-1, 2)}};

    auto&& [D, Rxyz] = s;
    REQUIRE((D == std::views::iota(0, 10)));
    REQUIRE((Rxyz == tuple{std::views::iota(0, 3), std::views::iota(3, 6), std::views::iota(-1, 2)}));

    static_assert(std::same_as<decltype(get<0, 0>(s)), decltype(get<0>(get<0>(s)))>);
    static_assert(std::same_as<decltype(get<si::D>(s)), decltype(get<0>(get<0>(s)))>);
}

TEST_CASE("conversion")
{
    using T = std::vector<int>;

    auto s = scalar<T>{tuple{std::vector{1, 2, 3}},
                       tuple{std::vector{1}, std::vector{2, 3}, std::vector{4, 5, 6}}};

    scalar<std::span<const int>> r = s;

    REQUIRE((r == s));

    auto f = [](scalar<std::span<int>> x) { x *= 2; };
    s = f;
    REQUIRE((r == scalar<T>{tuple{T{2, 4, 6}}, tuple{T{2}, T{4, 6}, T{8, 10, 12}}}));
}

TEST_CASE("selection")
{
    // some initialization
    auto v = std::vector{1, 2};
    auto rx = std::vector{3, 4};
    auto ry = std::vector{5, 6};
    auto rz = std::vector{7, 8};
    auto s = scalar<std::vector<int>&>(tuple{v}, tuple{rx, ry, rz});

    // Add tests to verify the type of the containers and views.
    // THey may point to scalar needing to be a tuple of nested tuples
    REQUIRE(std::ranges::equal(v, s | sel::D));
    REQUIRE(std::ranges::equal(rx, s | sel::Rx));
    REQUIRE(std::ranges::equal(ry, s | sel::Ry));
    REQUIRE(std::ranges::equal(rz, s | sel::Rz));

    // can compose these with pipe syntax since selections are just view closures
    REQUIRE(std::ranges::equal(
        s | sel::D | std::views::transform([](auto&& x) { return x * x * x; }),
        std::vector{1, 8}));

    // modify selection
    s | sel::D = 0;
    REQUIRE(std::ranges::equal(v, std::vector<int>(2, 0)));

    s | sel::Rz = -1;
    REQUIRE(std::ranges::equal(rz, std::vector<int>(2, -1)));

    s | sel::R = 2;
    REQUIRE(std::ranges::equal(v, std::vector<int>(2, 0)));
    REQUIRE(std::ranges::equal(rx, std::vector<int>(2, 2)));
}

TEST_CASE("selection assignment")
{
    auto nx = g::x.size();
    auto ny = g::y.size();
    auto nz = g::z.size();

    auto s = scalar<std::vector<real>>{};
    s = g::loc | std::views::transform([](auto&& xyz) { return get<0>(xyz); });
    REQUIRE(std::ranges::size(s | sel::D) == nx * ny * nz);

    s | sel::D = g::loc | std::views::transform([](auto&& xyz) { return get<2>(xyz); });

    {
        std::vector<real> expected;
        for (std::size_t i = 0; i < nx * ny; ++i)
            expected.insert(expected.end(), g::z.begin(), g::z.end());
        REQUIRE(std::ranges::equal(s | sel::D, expected));
    }
}

TEST_CASE("math")
{
    auto s = scalar<std::vector<int>>{
        tuple{std::views::iota(1, 5)}, tuple{std::views::iota(6, 10), std::views::iota(6, 12), std::views::iota(10, 15)}};

    auto q = s + 1;
    REQUIRE(std::ranges::equal(std::views::iota(2, 6), get<0>(q)));
    REQUIRE(std::ranges::equal(std::views::iota(7, 11), get<Rx>(q)));
    REQUIRE(std::ranges::equal(std::views::iota(7, 13), get<Ry>(q)));
    REQUIRE(std::ranges::equal(std::views::iota(11, 16), get<Rz>(q)));
    auto r = q + s;

    REQUIRE((r == tuple{tuple{plus(std::views::iota(2, 6), std::views::iota(1, 5))},
                        tuple{plus(std::views::iota(6, 10), std::views::iota(7, 11)),
                              plus(std::views::iota(6, 12), std::views::iota(7, 13)),
                              plus(std::views::iota(10, 15), std::views::iota(11, 16))}}));

    scalar<std::vector<int>> t = r;
    REQUIRE((t == r));

    scalar<std::vector<int>> a{s};
    scalar<std::span<int>> b = a;
    b = r;
    REQUIRE((b == r));
}

TEST_CASE("lifting single arg")
{
    auto s = scalar<std::vector<int>>{
        tuple{std::views::iota(1, 5)}, tuple{std::views::iota(6, 10), std::views::iota(6, 12), std::views::iota(10, 15)}};

    constexpr auto f = lift([](auto&& x) { return std::abs(x) + 1; });

    auto j = f(s);
    auto k = s + 1;

    REQUIRE((j == k));
}

TEST_CASE("lifting multiple args")
{
    auto x = scalar<std::vector<int>>{
        tuple{std::views::iota(1, 5)},
        tuple{std::views::iota(6, 10), std::views::iota(6, 12), std::vector{-4, 5, -10, 6}}};
    auto y = scalar<std::vector<int>>{
        tuple{std::views::iota(2, 6)},
        tuple{std::views::iota(5, 9), std::views::iota(-6, 0), std::vector{10, -8, 2, -7}}};

    constexpr auto f =
        lift([](auto&& x, auto&& y) { return std::max(std::abs(x), std::abs(y)); });

    auto z = f(x, y);

    REQUIRE(std::ranges::equal(get<0>(z), std::views::iota(2, 6)));
    REQUIRE(std::ranges::equal(get<Rx>(z), std::views::iota(6, 10)));
    REQUIRE(std::ranges::equal(get<Ry>(z), std::views::iota(6, 12)));
    REQUIRE(std::ranges::equal(get<Rz>(z), std::vector{10, 8, 10, 7}));
}

TEST_CASE("mesh location vector")
{
    using namespace g;

    constexpr auto t = std::views::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * z;
    });

    auto sol = loc | t;

    REQUIRE(std::ranges::equal(sol | sel::D, cartesian_product_vec(x, y, z) | t));
    REQUIRE(std::ranges::equal(sol | sel::Rx, rx | t));
    REQUIRE(std::ranges::equal(sol | sel::Ry, ry | t));
    REQUIRE(std::ranges::equal(sol | sel::Rz, rz | t));
    REQUIRE(std::ranges::equal(get<0>(sol | sel::R), rx | t));
    REQUIRE(std::ranges::equal(get<1>(sol | sel::R), ry | t));
    REQUIRE(std::ranges::equal(get<2>(sol | sel::R), rz | t));

    auto s = scalar<std::vector<real>>{loc | std::views::transform([](auto&&) { return 0; })};

    s | sel::Rx = sol;
    REQUIRE(std::ranges::equal(get<Rx>(s), rx | t));

    s | sel::Ry = sol;
    REQUIRE(std::ranges::equal(get<Ry>(s), ry | t));

    s | sel::Rz = sol;
    REQUIRE(std::ranges::equal(get<Rz>(s), rz | t));

    constexpr auto u = std::views::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * y * z;
    });
    // the problem with this statement has to with the assignability
    // of the view_tuples components

    s | sel::R = loc | u;
    REQUIRE(std::ranges::equal(s | sel::Rx, rx | u));
    REQUIRE(std::ranges::equal(s | sel::Ry, ry | u));
    REQUIRE(std::ranges::equal(s | sel::Rz, rz | u));

    REQUIRE(std::ranges::equal(s | sel::D,
        std::vector<real>(std::ranges::size(loc | sel::D), 0.0)));
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

    auto t = std::vector<real>(x.size() * y.size() * z.size());
    auto u = std::vector<real>(rx.size());
    auto v = std::vector<real>(ry.size());
    auto w = std::vector<real>(rz.size());

    auto s = scalar<std::span<real>>{tuple{t}, tuple{u, v, w}};

    constexpr auto tr = std::views::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * z;
    });

    const auto sol = loc | tr;

    s | sel::D = sol;
    REQUIRE(std::ranges::equal(s | sel::D, cartesian_product_vec(x, y, z) | tr));
    REQUIRE(std::ranges::equal(s | sel::Rx, std::vector<real>(rx.size(), 0.0)));

    s | sel::Rx = sol;
    REQUIRE(std::ranges::equal(s | sel::Rx, rx | tr));

    s | sel::Ry = sol;
    REQUIRE(std::ranges::equal(s | sel::Ry, ry | tr));

    s | sel::Rz = sol;
    REQUIRE(std::ranges::equal(s | sel::Rz, rz | tr));

    constexpr auto ur = std::views::transform([](auto&& loc) {
        auto&& [x, y, z] = loc;
        return x * y * y * z;
    });

    s | sel::R = loc | ur;
    REQUIRE(std::ranges::equal(s | sel::Rx, rx | ur));
    REQUIRE(std::ranges::equal(s | sel::Ry, ry | ur));
    REQUIRE(std::ranges::equal(s | sel::Rz, rz | ur));

    s | sel::R = loc | sel::R |
                 tuple{std::views::transform(loc_fn<0>{}),
                       std::views::transform(loc_fn<1>{}),
                       std::views::transform(loc_fn<2>{})};

    REQUIRE(std::ranges::equal(u, rx | std::views::transform(loc_fn<0>{})));
    REQUIRE(std::ranges::equal(v, ry | std::views::transform(loc_fn<1>{})));
    REQUIRE(std::ranges::equal(w, rz | std::views::transform(loc_fn<2>{})));
}

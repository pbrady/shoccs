#include "container_tuple.hpp"
#include "view_tuple.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <ranges>
#include <vector>

using namespace ccs;

TEST_CASE("default construction")
{
    using namespace ccs;

    using T = std::vector<real>;

    {
        T t{};
        REQUIRE(std::ranges::size(t) == 0);
    }

    {
        std::tuple<T> t{};
        REQUIRE(std::ranges::size(get<0>(t)) == 0);
    }

    {
        container_tuple<T> t{};
        REQUIRE(std::ranges::size(get<0>(t)) == 0);
    }

    {
        container_tuple<T, T> t{};
        REQUIRE(std::ranges::size(get<0>(t)) == 0);
        REQUIRE(std::ranges::size(get<1>(t)) == 0);
    }

    {
        container_tuple<container_tuple<T>, container_tuple<T, T>> t{};
        auto&& [a, b] = t;
        REQUIRE(std::ranges::size(get<0>(a)) == 0);
        REQUIRE(std::ranges::size(get<0>(b)) == 0);
        REQUIRE(std::ranges::size(get<1>(b)) == 0);
    }
}

TEST_CASE("Structured Binding")
{
    using T = std::vector<real>;

    auto x =
        container_tuple<T, T>{std::vector<real>{0, 1, 2}, std::vector<real>{3, 4, 5}};
    static_assert(std::tuple_size_v<std::remove_cvref_t<decltype(x)>> == 2u);

    {
        // get references to the ranges, changes are visible
        auto&& [a, b] = x;
        for (auto&& i : a) i *= 3;
        b[0] = 1;
        b[1] = 2;
        b[2] = 3;
    }

    auto&& [a, b] = x;
    REQUIRE(std::ranges::equal(a, std::vector<real>{0, 3, 6}));
    REQUIRE(std::ranges::equal(b, std::vector<real>{1, 2, 3}));

    {
        // copy the ranges, changes are not visible
        auto [a, b] = x;
        for (auto&& i : a) i *= 3;
        b[0] = 1;
        b[1] = 2;
        b[2] = 3;
    }

    auto&& [c, d] = x;
    REQUIRE(std::ranges::equal(c, std::vector<real>{0, 3, 6}));
    REQUIRE(std::ranges::equal(d, std::vector<real>{1, 2, 3}));
}

TEST_CASE("Assignment")
{
    using T = std::vector<real>;

    auto x =
        container_tuple<T, T>{std::vector<real>{0, 1, 2}, std::vector<real>{3, 4, 5}};
    x = 1;

    REQUIRE(std::ranges::equal(get<0>(x), std::vector<real>{1, 1, 1}));
}

TEST_CASE("Construction from Ranges")
{
    using T = std::vector<real>;
    auto x = container_tuple<T>{std::views::iota(0, 10)};
    {
        auto&& [v] = x;
        REQUIRE(std::ranges::equal(v, std::views::iota(0, 10)));
    }

    auto y = container_tuple<T, T>{std::views::iota(0, 10), std::views::iota(1, 6)};
    {
        REQUIRE(std::ranges::equal(get<0>(y), std::views::iota(0, 10)));
        REQUIRE(std::ranges::equal(get<1>(y), std::views::iota(1, 6)));
    }
}

TEST_CASE("Construction from OneViewTuples")
{
    using T = std::vector<real>;
    const auto i = std::views::iota(0, 100);

    {
        auto v = view_tuple{i};
        container_tuple<T> x{};
        x = v;
        REQUIRE(std::ranges::equal(get<0>(x), i));
    }

    {
        auto v = view_tuple{i};
        container_tuple<T> x{v};
        REQUIRE(std::ranges::equal(get<0>(x), i));
    }

    {
        auto v = view_tuple{i};
        container_tuple<T> x{};
        x = MOVE(v);
        REQUIRE(std::ranges::equal(get<0>(x), i));
    }

    {
        auto v = view_tuple{i};
        container_tuple<T> x{MOVE(v)};
        REQUIRE(std::ranges::equal(get<0>(x), i));
    }
}

TEST_CASE("Construction from TwoViewTuples")
{
    using T = std::vector<real>;
    using U = container_tuple<T, T>;
    const auto i = std::views::iota(0, 100);
    const auto j = std::views::iota(-1, 1);

    {
        auto v = view_tuple{i, j};
        U x{};
        x = v;
        REQUIRE(std::ranges::equal(get<0>(x), i));
        REQUIRE(std::ranges::equal(get<1>(x), j));
    }

    {
        auto v = view_tuple{i, j};
        U x{v};
        REQUIRE(std::ranges::equal(get<0>(x), i));
        REQUIRE(std::ranges::equal(get<1>(x), j));
    }

    {
        auto v = view_tuple{i, j};
        U x{};
        x = MOVE(v);
        REQUIRE(std::ranges::equal(get<0>(x), i));
        REQUIRE(std::ranges::equal(get<1>(x), j));
    }

    {
        auto v = view_tuple{i, j};
        U x{MOVE(v)};
        REQUIRE(std::ranges::equal(get<0>(x), i));
        REQUIRE(std::ranges::equal(get<1>(x), j));
    }
}

TEST_CASE("Copy")
{
    using T = std::vector<real>;

    SECTION("copy assignment OneTuple")
    {
        auto x = container_tuple<T>{};
        const auto y = container_tuple<T>{std::views::iota(0, 10)};

        x = y;
        REQUIRE(std::ranges::equal(get<0>(x), get<0>(y)));
    }
    SECTION("copy construction OneTuple")
    {
        const auto y = container_tuple<T>{std::views::iota(0, 10)};
        container_tuple<T> x{y};

        REQUIRE(std::ranges::equal(get<0>(x), get<0>(y)));
    }

    SECTION("copy assignment TwoTuple")
    {
        auto x = container_tuple<T, T>{};
        const auto y = container_tuple<T, T>{std::views::iota(0, 10), std::views::iota(1, 4)};

        x = y;
        REQUIRE(std::ranges::equal(get<0>(x), get<0>(y)));
        REQUIRE(std::ranges::equal(get<1>(x), get<1>(y)));
    }

    SECTION("copy construction TwoTuple")
    {
        const auto y = container_tuple<T, T>{std::views::iota(0, 10), std::views::iota(1, 4)};
        container_tuple<T, T> x{y};

        REQUIRE(std::ranges::equal(get<0>(x), get<0>(y)));
        REQUIRE(std::ranges::equal(get<1>(x), get<1>(y)));
    }

    SECTION("copy assignment for OtherContainerTuple")
    {
        auto x = container_tuple<T, T>{};
        const auto y = container_tuple<std::vector<int>, std::vector<int>>{
            std::views::iota(0, 10), std::views::iota(1, 4)};

        x = y;
        REQUIRE(std::ranges::equal(get<0>(x), get<0>(y)));
        REQUIRE(std::ranges::equal(get<1>(x), get<1>(y)));
    }
}

TEST_CASE("Move")
{
    using T = std::vector<real>;

    const auto i = std::views::iota(0, 10);
    const auto j = std::views::iota(1, 4);

    SECTION("move assignment OneTuple")
    {
        auto x = container_tuple<T>{};
        const auto y = container_tuple<T>{i};

        x = MOVE(y);
        REQUIRE(std::ranges::equal(get<0>(x), i));
    }
    SECTION("move construction OneTuple")
    {
        auto y = container_tuple<T>{i};
        container_tuple<T> x{MOVE(y)};

        REQUIRE(std::ranges::equal(get<0>(x), i));
    }

    SECTION("move assignment TwoTuple")
    {
        auto x = container_tuple<T, T>{};
        auto y = container_tuple<T, T>{i, j};

        x = MOVE(y);
        REQUIRE(std::ranges::equal(get<0>(x), i));
        REQUIRE(std::ranges::equal(get<1>(x), j));
    }

    SECTION("move construction TwoTuple")
    {
        auto y = container_tuple<T, T>{i, j};
        container_tuple<T, T> x{MOVE(y)};

        REQUIRE(std::ranges::equal(get<0>(x), i));
        REQUIRE(std::ranges::equal(get<1>(x), j));
    }

    SECTION("move assignment for OtherContainerTuple")
    {
        auto x = container_tuple<T, T>{};
        auto y = container_tuple<std::vector<int>, std::vector<int>>{i, j};

        x = MOVE(y);
        REQUIRE(std::ranges::equal(get<0>(x), i));
        REQUIRE(std::ranges::equal(get<1>(x), j));
    }
}

TEST_CASE("Nested")
{
    using T = std::vector<real>;

    {
        container_tuple<container_tuple<T, T>> x{
            container_tuple<T, T>{std::views::iota(0, 10), std::views::iota(4, 7)}};

        auto&& [c] = x;

        auto&& [a, b] = c;

        REQUIRE(std::ranges::equal(a, std::views::iota(0, 10)));
        REQUIRE(std::ranges::equal(b, std::views::iota(4, 7)));
    }

    {
        container_tuple<container_tuple<T, T>, container_tuple<T>> x{
            container_tuple<T, T>{std::views::iota(0, 10), std::views::iota(4, 7)},
            container_tuple<T>{T{-1, -2, -3}}};

        auto&& [u, v] = x;

        auto&& [a, b] = u;
        REQUIRE(std::ranges::equal(a, std::views::iota(0, 10)));
        REQUIRE(std::ranges::equal(b, std::views::iota(4, 7)));

        auto&& [c] = v;
        REQUIRE(std::ranges::equal(c, T{-1, -2, -3}));
    }
}

TEST_CASE("constexpr")
{
    [[maybe_unused]] constexpr auto x =
        container_tuple{std::views::transform([](auto&& i) { return i + 1; })};
}

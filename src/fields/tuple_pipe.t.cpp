#include "tuple.hpp"

#include "types.hpp"

#include <catch2/catch_test_macros.hpp>

#include "ccs_range_utils.hpp"

#include <algorithm>
#include <ranges>
#include <vector>

using namespace ccs;

TEST_CASE("concepts")
{
    using T1 = tuple<std::vector<real>>;
    using G = decltype([](auto&& i) { return i; });
    using F = decltype(std::views::transform([](auto&& i) { return i; }));

    REQUIRE(PipeableOver<F, T1>);
    REQUIRE(!PipeableOver<G, T1>);

    using T3 = tuple<std::span<int>, std::span<int>, std::span<const int>>;
    REQUIRE(PipeableOver<F, T3>);
    REQUIRE(!PipeableOver<G, T3>);

    using S = tuple<T1, T3>;
    REQUIRE(PipeableOver<F, S>);

    using V = tuple<T3, T3>;
    REQUIRE(PipeableOver<F, V>);
}

TEST_CASE("Pipe syntax for Owning OneTuples")
{
    auto x = tuple{std::vector{1, 2, 3}};
    auto y = x | std::views::transform([](auto&& i) { return i * i; });
    REQUIRE(std::ranges::equal(y, std::vector{1, 4, 9}));

    auto z = tuple<std::vector<int>>{};
    z = y | std::views::transform([](auto&& i) { return i * i; });
    REQUIRE(std::ranges::equal(z, std::vector{1, 16, 81}));
}

TEST_CASE("Pipe syntax for Non-Owning OneTuples")
{
    auto x = tuple{std::vector{1, 2, 3}};
    auto y = x | std::views::transform([](auto&& i) { return i * i; });
    REQUIRE(std::ranges::equal(y, std::vector{1, 4, 9}));

    // copying should resize the vector when needed
    auto zz = std::vector<int>(3);

    auto z = tuple{zz};
    z = y | std::views::transform([](auto&& i) { return i * i; });
    REQUIRE(std::ranges::size(zz) == std::ranges::size(y));
    REQUIRE(std::ranges::equal(zz, std::vector{1, 16, 81}));
    REQUIRE(std::ranges::equal(z, zz));
}

TEST_CASE("Pipe syntax for ThreeTuples")
{
    auto x = tuple{std::vector{1, 2, 3}, std::vector{4, 5}, std::vector{4, 3, 2, 1}};
    auto y = x | std::views::transform([](auto&& i) { return i * i; });
    REQUIRE(ThreeTuple<decltype(y)>);
    REQUIRE((y ==
            tuple{std::vector{1, 4, 9}, std::vector{16, 25}, std::vector{16, 9, 4, 1}}));

    auto z = tuple<std::vector<int>, std::vector<int>, std::vector<int>>{};
    z = y | std::views::transform([](auto&& i) { return i + i; });
    REQUIRE((z ==
            tuple{std::vector{2, 8, 18}, std::vector{32, 50}, std::vector{32, 18, 8, 2}}));
}

TEST_CASE("Pipe syntax for Non-Owning ThreeTuples")
{
    auto x = tuple{std::vector{1, 2, 3}, std::vector{4, 5}, std::vector{4, 3, 2, 1}};
    auto y = x | std::views::transform([](auto&& i) { return i * i; });
    REQUIRE(ThreeTuple<decltype(y)>);
    REQUIRE((y ==
            tuple{std::vector{1, 4, 9}, std::vector{16, 25}, std::vector{16, 9, 4, 1}}));

    auto a = std::vector<int>(3);
    auto b = std::vector<int>(2);
    auto c = std::vector<int>(4);
    auto z = tuple{a, b, c};

    z = y | std::views::transform([](auto&& i) { return i + i; });
    REQUIRE((z ==
            tuple{std::vector{2, 8, 18}, std::vector{32, 50}, std::vector{32, 18, 8, 2}}));
}

TEST_CASE("ThreeTuples with ThreeTuplePipes")
{
    constexpr auto f = std::views::transform([](auto&& i) { return i * i; });
    constexpr auto g = std::views::transform([](auto&& i) { return i + i; });
    constexpr auto h = std::views::transform([](auto&& i) { return i * i * i; });

    auto x = tuple{std::vector{1, 2, 3}, std::vector{4, 5}, std::vector{4, 3, 2, 1}};
    auto y = x | std::tuple{f, g, h};
    REQUIRE(ThreeTuple<decltype(y)>);
    REQUIRE((y ==
            tuple{std::vector{1, 4, 9}, std::vector{8, 10}, std::vector{64, 27, 8, 1}}));

    auto a = std::vector<int>(3);
    auto b = std::vector<int>(2);
    auto c = std::vector<int>(4);
    auto z = tuple{a, b, c};
    z = y | g; // std::views::transform([](auto&& i) { return i + i; });
    REQUIRE((z == tuple{std::vector{2, 8, 18},
                       std::vector{16, 20},
                       std::vector{128, 54, 16, 2}}));

    auto q = tuple<std::span<int>, std::span<int>, std::span<int>>{a, b, c};

    q | std::tuple{f, g, h};

    q = y | tuple{f, g, h};

    REQUIRE(std::ranges::equal(a, std::vector{1, 16, 81}));
    REQUIRE(std::ranges::equal(b, std::vector{16, 20}));
    REQUIRE(std::ranges::equal(c, std::vector{262144, 19683, 512, 1}));

    // closure | tuple composition requires ccs::view_closure (not std:: adaptor)
    auto vc_id = ccs::make_view_closure(std::views::transform([](auto&& i) { return i; }));
    vc_id | tuple{ccs::make_view_closure(std::views::transform([](auto&& i) { return i * i; })),
                  ccs::make_view_closure(std::views::transform([](auto&& i) { return i + i; })),
                  ccs::make_view_closure(std::views::transform([](auto&& i) { return i * i * i; }))};
}

TEST_CASE("Nested Pipe")
{
    auto x =
        tuple{tuple{std::vector{1, 2, 3}, std::vector{4, 5}, std::vector{4, 3, 2, 1}},
              tuple{std::vector{-1, -2, -3}}};
    auto y = x | std::views::transform([](auto&& i) { return i + 2; });

    REQUIRE((y ==
            tuple{tuple{std::vector{3, 4, 5}, std::vector{6, 7}, std::vector{6, 5, 4, 3}},
                  tuple{std::vector{1, 0, -1}}}));
}

TEST_CASE("constexpr tuples of transforms")
{
    constexpr auto tup = tuple{std::views::transform([](auto&& i) { return i * i; }),
                               std::views::transform([](auto&& i) { return i + i; }),
                               std::views::transform([](auto&& i) { return i * i * i; })};

    constexpr auto x =
        tuple{std::array{1, 2, 3}, std::array{2, 3, 4}, std::array{3, 4, 5}};

    REQUIRE(((x | tup) ==
            tuple{std::array{1, 4, 9}, std::array{4, 6, 8}, std::array{27, 64, 125}}));
}

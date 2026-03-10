#include "scalar.hpp"
#include "tuple.hpp"
#include "vector.hpp"

#include "selector.hpp"

#include "types.hpp"

#include "algorithms.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <ranges>
#include <vector>

using namespace ccs;

TEST_CASE("dot product")
{
    auto a = vector<std::vector<int>>{
        tuple{tuple{std::vector{1, 2, 3}},
              tuple{std::vector{1}, std::vector{2}, std::vector{3, 4}}},
        tuple{tuple{std::vector{4, 5, 6}},
              tuple{std::vector{2}, std::vector{3}, std::vector{4, 5}}},
        tuple{tuple{std::vector{7, 8, 9}},
              tuple{std::vector{3}, std::vector{4}, std::vector{5, 6}}}};
    vector<std::vector<int>> b = a;
    b *= 10;

    scalar<std::vector<int>> s = dot(a, b);

    REQUIRE(std::ranges::equal(s | sel::D,
                      std::vector{1 * 10 + 4 * 40 + 7 * 70,
                                  2 * 20 + 5 * 50 + 8 * 80,
                                  3 * 30 + 6 * 60 + 9 * 90}));

    REQUIRE(std::ranges::equal(s | sel::Rx, std::vector{10 + 2 * 20 + 3 * 30}));
    REQUIRE(std::ranges::equal(s | sel::Ry, std::vector{2 * 20 + 3 * 30 + 4 * 40}));
    REQUIRE(std::ranges::equal(s | sel::Rz,
                      std::vector{3 * 30 + 4 * 40 + 5 * 50, 4 * 40 + 5 * 50 + 6 * 60}));
};

TEST_CASE("minmax")
{
    scalar_real s{tuple{std::views::iota(0, 12)},
                  tuple{std::views::iota(-10, -8), std::views::iota(-8, 6), std::views::iota(12, 13)}};

    auto&& [smin, smax] = minmax(s);

    REQUIRE(smin == -10);
    REQUIRE(smax == 12);
}

TEST_CASE("max")
{
    scalar_real s{tuple{std::views::iota(0, 12)},
                  tuple{std::views::iota(-10, -8), std::views::iota(-8, 6), std::views::iota(12, 13)}};

    auto smax = max(s);

    REQUIRE(smax == 12);
}

#include "field_utils.hpp"
#include "field.hpp"

#include <algorithm>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

using namespace ccs;
using T = std::vector<real>;

TEST_CASE("for_each")
{
    auto x = field{system_size{1, 0, tuple{tuple{5}, tuple{1, 2, 3}}}};

    for_each([](auto& v) { v += 1; }, x);

    {
        auto&& xs = x.scalars(0);
        REQUIRE(std::ranges::equal(xs | sel::D, std::vector<int>(5, 1)));
        REQUIRE(std::ranges::equal(xs | sel::Rx, std::vector<int>(1, 1)));
        REQUIRE(std::ranges::equal(xs | sel::Ry, std::vector<int>(2, 1)));
        REQUIRE(std::ranges::equal(xs | sel::Rz, std::vector<int>(3, 1)));
    }

    auto y = field{system_size{1, 0, tuple{tuple{5}, tuple{1, 2, 3}}}};

    for_each(
        [](auto& u, auto& v) {
            v += 1;
            u += v;
        },
        x,
        y);

    {
        auto&& xs = x.scalars(0);
        REQUIRE(std::ranges::equal(xs | sel::D, std::vector<int>(5, 2)));
        REQUIRE(std::ranges::equal(xs | sel::Rx, std::vector<int>(1, 2)));
        REQUIRE(std::ranges::equal(xs | sel::Ry, std::vector<int>(2, 2)));
        REQUIRE(std::ranges::equal(xs | sel::Rz, std::vector<int>(3, 2)));
    }
}

TEST_CASE("transform")
{
    auto x = field{system_size{1, 0, tuple{tuple{5}, tuple{1, 2, 3}}}};

    auto y = transform([](auto&& v) { return v + 1; }, x);

    {
        auto&& ys = y.scalars(0);
        // auto ys = y.scalars()[0];
        REQUIRE(std::ranges::equal(ys | sel::D, std::vector<int>(5, 1)));
        REQUIRE(std::ranges::equal(ys | sel::Rx, std::vector<int>(1, 1)));
        REQUIRE(std::ranges::equal(ys | sel::Ry, std::vector<int>(2, 1)));
        REQUIRE(std::ranges::equal(ys | sel::Rz, std::vector<int>(3, 1)));
    }
}

// Compile-instantiation test for selections.hpp (7.2e).
// selections.hpp is dead code (not #include'd by any production TU), so the
// migrated YPlaneView and FView templates have never been compiled.  This file
// forces instantiation and verifies they satisfy the expected range concepts.

#include "selections.hpp"

#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <ranges>
#include <span>
#include <vector>

using namespace ccs;

// ---- YPlaneView compile-instantiation checks ----

using YPV = ccs::views::detail::YPlaneView<std::views::all_t<std::span<int>>>;

static_assert(std::ranges::random_access_range<YPV>);
static_assert(std::ranges::sized_range<YPV>);

// ---- FView compile-instantiation checks ----

using FV = ccs::views::detail::FView<std::views::all_t<std::span<int>>>;

static_assert(std::ranges::random_access_range<FV>);
static_assert(std::ranges::sized_range<FV>);

// A minimal runtime test to exercise begin/end/size for YPlaneView.
TEST_CASE("YPlaneView instantiation")
{
    // 2x3x4 grid, select y-plane j=1
    std::vector<int> data(2 * 3 * 4);
    std::iota(data.begin(), data.end(), 0);

    const int3 extents{2, 3, 4};
    auto view = ccs::views::detail::YPlaneView(std::span(data), extents, 1);

    REQUIRE(view.size() == extents[0] * extents[2]); // 2 * 4 = 8
    REQUIRE(std::ranges::distance(view.begin(), view.end()) == view.size());
}

// A minimal runtime test to exercise begin/end/size for FView.
TEST_CASE("FView instantiation")
{
    // Single line spanning indices 2..5 in a 10-element range (stride=1, no objects).
    std::vector<int> data(10);
    std::iota(data.begin(), data.end(), 0);

    const index_extents ext{int3{1, 1, 10}};
    // line: stride=1, start at mesh coord (0,0,2) no object, end at (0,0,5) no object
    const line ln{1, boundary{int3{0, 0, 2}, std::nullopt}, boundary{int3{0, 0, 5}, std::nullopt}};
    std::array<line, 1> lines_arr{ln};

    auto view = ccs::views::detail::FView(
        std::span(data), ext, std::span<const line>(lines_arr));

    // ext(int3{0,0,2}) = 2, ext(int3{0,0,5}) = 5, no objects → i0=2, i1=6 → size=4
    REQUIRE(view.size() == 4);
    REQUIRE(std::ranges::distance(view.begin(), view.end()) == view.size());

    // Check the actual values: should be data[2], data[3], data[4], data[5]
    auto it = view.begin();
    REQUIRE(*it == 2);
    ++it;
    REQUIRE(*it == 3);
    ++it;
    REQUIRE(*it == 4);
    ++it;
    REQUIRE(*it == 5);
}

// Multi-line FView test exercising forward iteration, backward iteration from
// end(), and operator+=(-n) across line boundaries (regression test for 7.2f).
TEST_CASE("FView multi-line forward and backward traversal")
{
    // 20-element flat range
    std::vector<int> data(20);
    std::iota(data.begin(), data.end(), 0);

    const index_extents ext{int3{1, 1, 20}};

    // Line 0: (0,0,2)..(0,0,5), no objects → i0=2, i1=6 → elements {2,3,4,5}
    // Line 1: (0,0,10)..(0,0,14), no objects → i0=10, i1=15 → elements {10,11,12,13,14}
    const line ln0{1,
                   boundary{int3{0, 0, 2}, std::nullopt},
                   boundary{int3{0, 0, 5}, std::nullopt}};
    const line ln1{1,
                   boundary{int3{0, 0, 10}, std::nullopt},
                   boundary{int3{0, 0, 14}, std::nullopt}};
    std::array<line, 2> lines_arr{ln0, ln1};

    auto view = ccs::views::detail::FView(
        std::span(data), ext, std::span<const line>(lines_arr));

    REQUIRE(view.size() == 9); // 4 + 5

    SECTION("forward iteration produces correct values across line boundary")
    {
        std::vector<int> expected{2, 3, 4, 5, 10, 11, 12, 13, 14};
        std::vector<int> result;
        for (auto v : view) result.push_back(v);
        REQUIRE(result == expected);
    }

    SECTION("operator-- from end() produces last element")
    {
        auto it = view.end();
        --it;
        REQUIRE(*it == 14);
    }

    SECTION("operator-- from end() crosses line boundary correctly")
    {
        auto it = view.end();
        // Walk backward through all 9 elements
        std::vector<int> result;
        for (int n = 0; n < 9; ++n) {
            --it;
            result.push_back(*it);
        }
        REQUIRE(it == view.begin());
        std::vector<int> expected{14, 13, 12, 11, 10, 5, 4, 3, 2};
        REQUIRE(result == expected);
    }

    SECTION("operator+=(-n) from end() crosses line boundary correctly")
    {
        auto it = view.end();
        it += -9;
        REQUIRE(it == view.begin());
        REQUIRE(*it == 2);

        // Jump from end to the middle of line 0
        it = view.end();
        it += -7; // 9 - 7 = 3rd element (index 2) → data[4]
        REQUIRE(*it == 4);

        // Jump from end to the start of line 1
        it = view.end();
        it += -5; // last 5 elements are line 1 → data[10]
        REQUIRE(*it == 10);
    }
}

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

#include "container_tuple.hpp"
#include "index_extents.hpp"
#include "types.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <ranges>

#include "ccs_range_utils.hpp"
#include "lazy_views.hpp"

#include <cstdlib>
#include <vector>

using namespace ccs;

TEST_CASE("Output Ranges")
{
    REQUIRE(std::ranges::output_range<std::vector<real>&, real>);
    REQUIRE(std::ranges::output_range<std::vector<real>&, int>);
    REQUIRE(OutputRange<std::vector<real>&>);
    REQUIRE(OutputRange<std::vector<real>&, real>);

    REQUIRE(!std::ranges::output_range<const std::vector<real>&, real>);
    REQUIRE(!OutputRange<const std::vector<real>&>);

    REQUIRE(OutputRange<std::span<real>>);
    REQUIRE(
        std::ranges::output_range<std::span<real>,
                                  std::ranges::range_value_t<decltype(std::views::iota(0, 10))>>);
    REQUIRE(!OutputRange<std::span<const real>>);
    REQUIRE(OutputRange<std::span<real>, std::span<const real>>);
}

TEST_CASE("OutputTuple")
{
    using T = std::vector<real>;

    REQUIRE(OutputTuple<std::tuple<T, T>, real>);
    REQUIRE(OutputTuple<std::tuple<std::tuple<T>, std::tuple<T, T, T>>, real>);
    REQUIRE(OutputTuple<std::tuple<T, T>, int>);
    REQUIRE(OutputTuple<std::tuple<T&, T&>, T>);
    REQUIRE(!OutputTuple<std::tuple<const T&, const T&>, T>);

    REQUIRE(OutputTuple<std::tuple<std::span<real>>, T>);
    REQUIRE(!OutputTuple<std::tuple<std::span<const real>>, std::span<const real>>);

    REQUIRE(OutputTuple<std::tuple<std::span<real>, std::span<real>>,
                        std::tuple<const T&, const T&>>);
    REQUIRE(!OutputTuple<std::tuple<const T&, const T&>,
                         std::tuple<std::span<real>, std::span<real>>>);
}

TEST_CASE("Modify Containers from Views")
{
    auto x = std::vector{1, 2, 3};
    auto y = std::views::all(x);

    REQUIRE(OutputRange<decltype(y)>);

    for (auto&& i : y) i *= 2;

    REQUIRE(std::ranges::equal(x, std::vector{2, 4, 6}));

    {
        constexpr auto f = [](auto&& i) { return i; };
        auto a = x | std::views::transform(f);
        auto b = y | std::views::transform(f);
        static_assert(std::same_as<decltype(a), decltype(b)>);
    }
}

TEST_CASE("TupleLike")
{
    REQUIRE(TupleLike<std::tuple<int, int>>);
    REQUIRE(TupleLike<container_tuple<std::vector<int>>>);

    // take results in a custom range type.  Ensure we do not treat it as one of ours.
    auto x = std::vector<int>(50);
    REQUIRE(!TupleLike<decltype(x | std::views::take(5))>);
}

TEST_CASE("NotTupleRanges")
{
    REQUIRE(NonTupleRange<std::vector<real>>);
    REQUIRE(NonTupleRange<std::vector<real>&>);
    REQUIRE(NonTupleRange<const std::vector<real>&>);
    REQUIRE(NonTupleRange<std::span<real>>);
    REQUIRE(NonTupleRange<std::span<const real>>);
}

TEST_CASE("From")
{
    using T = std::vector<int>;
    using I = decltype(std::views::iota(0, 10));
    using Z = decltype(ccs::zip_transform(std::plus{}, std::views::iota(0, 10), std::views::iota(1, 11)));

    REQUIRE(is_constructible_from_range<std::span<const int>, T>::value);
    REQUIRE(is_constructible_from_range<T, I>::value);
    REQUIRE(is_constructible_from<std::span<const int>, const T&>::value);
    REQUIRE(is_constructible_from<T, I>::value);
    REQUIRE(is_constructible_from<T, Z>::value);

    REQUIRE(TupleFromTuple<std::tuple<T>, std::tuple<I>>);
    REQUIRE(TupleFromTuple<std::tuple<T, T>, std::tuple<Z, I>>);
    REQUIRE(TupleFromTuple<std::tuple<std::tuple<T>, std::tuple<T, T>>,
                           std::tuple<std::tuple<Z>, std::tuple<I, Z>>>);
}

TEST_CASE("tuple shape")
{
    REQUIRE(SimilarTuples<std::tuple<int>, std::tuple<double>>);
    REQUIRE(SimilarTuples<std::tuple<std::tuple<int>>, std::tuple<std::tuple<void*>>>);
    REQUIRE(!SimilarTuples<std::tuple<std::tuple<int>>, std::tuple<void*>>);
    REQUIRE(
        SimilarTuples<std::tuple<std::tuple<int>, std::tuple<int, int, int>>,
                      std::tuple<std::tuple<void*>, std::tuple<void*, char, double>>>);
    REQUIRE(
        !SimilarTuples<std::tuple<std::tuple<int, int, int>, std::tuple<void*>>,
                       std::tuple<std::tuple<void*>, std::tuple<void*, char, double>>>);
}

TEST_CASE("levels")
{
    REQUIRE(tuple_levels_v<std::tuple<int>> == 1);
    REQUIRE(tuple_levels_v<std::tuple<int, double, float, char, void*>> == 1);
    REQUIRE(tuple_levels_v<std::tuple<std::tuple<int>>> == 2);
    REQUIRE(tuple_levels_v<std::tuple<std::tuple<int>,
                                      std::tuple<float, double, char>,
                                      std::tuple<void*>>> == 2);
}

TEST_CASE("view closures")
{
    using I = decltype(ccs::make_view_closure([](auto&& rng) { return rng; }));
    REQUIRE(ViewClosure<I>);
    REQUIRE(ViewClosures<I>);
    REQUIRE(ViewClosures<std::tuple<I, I>>);
}

TEST_CASE("list index")
{
    using L = list_index<4, 5, 6>;
    REQUIRE(ListIndex<L>);
    static_assert(index_v<L, 0> == 4);
    static_assert(index_v<L, 1> == 5);
    static_assert(index_v<L, 2> == 6);

    using ListOfL = mp_list<list_index<0, 1, 2>, list_index<1>>;
    REQUIRE(ListIndices<ListOfL>);
}

TEST_CASE("viewable ranges")
{
    static_assert(std::same_as<viewable_range_by_value<std::span<int>>, std::span<int>>);
    static_assert(std::same_as<viewable_range_by_value<std::span<int>&>, std::span<int>>);
    static_assert(std::same_as<viewable_range_by_value<std::span<const int>>,
                               std::span<const int>>);
    static_assert(std::same_as<viewable_range_by_value<std::span<const int>&>,
                               std::span<const int>>);
    static_assert(
        std::same_as<viewable_range_by_value<std::vector<int>&>, std::vector<int>&>);
    static_assert(std::same_as<viewable_range_by_value<const std::vector<int>&>,
                               const std::vector<int>&>);
}

TEST_CASE("NumericTuple")
{
    REQUIRE(NumericTuple<real3>);
    REQUIRE(NumericTuple<std::tuple<real, int>>);
    REQUIRE(NumericTuple<std::tuple<real&, const int&>>);
    REQUIRE(!NumericTuple<std::tuple<std::vector<int>>>);

    REQUIRE(ArrayFromTuple<real3, std::tuple<int, int, int>>);
}

struct a {
    int x, y;

    auto f(int q) { return q ? y : x; }
};

TEST_CASE("projection")
{
    std::vector<a> v{{1, 0}, {2, 3}, {6, 2}};
    auto u = v | std::views::transform(&a::x);

    REQUIRE(std::ranges::equal(u, std::vector{1, 2, 6}));

    // auto q = v | std::views::transform(&a::f(3));
}

TEST_CASE("generate_n")
{
    using T = std::vector<int>;

    T v;
    for (int i = 0; i < 3; i++) {
        v.push_back(0);
        v.push_back(1);
    }

    REQUIRE(v == T{0, 1, 0, 1, 0, 1});
}

// xplane
namespace det
{
template <typename Rng>
using X =
    decltype(std::declval<Rng>() | std::views::drop(int{}) | std::views::take(integer{}));

template <typename Rng>
using Z = decltype(ccs::stride(std::declval<Rng>() | std::views::drop(int{}), integer{}));

} // namespace det

template <typename Rng>
class x_plane_view : public det::X<Rng>
{
    using base = det::X<Rng>;

public:
    x_plane_view() = default;
    explicit constexpr x_plane_view(Rng&& rng, index_extents extents, int i)
        : base{FWD(rng) | std::views::drop(i * extents[1] * extents[2]) |
               std::views::take(extents[1] * extents[2])}
    {
    }
};

template <typename Rng>
class z_plane_view : public det::Z<Rng>
{
    using base = det::Z<Rng>;

public:
    z_plane_view() = default;
    explicit constexpr z_plane_view(Rng&& rng, index_extents extents, int i)
        : base{ccs::stride(FWD(rng) | std::views::drop(i), extents[2])}
    {
    }
};

template <typename Rng>
x_plane_view(Rng&&, index_extents, int) -> x_plane_view<Rng>;

template <typename Rng>
z_plane_view(Rng&&, index_extents, int) -> z_plane_view<Rng>;

TEST_CASE("x_plane_view")
{

    index_extents i{.extents = int3{2, 3, 4}};
    auto t_ = std::views::iota(0, 2 * 3 * 4);
    std::vector<int> t{std::ranges::begin(t_), std::ranges::end(t_)};

    REQUIRE(std::ranges::equal(x_plane_view(t, i, 0), std::views::iota(0, 12)));

    using T = decltype(x_plane_view(t, i, 0));
    REQUIRE(std::ranges::sized_range<T>);
    REQUIRE(std::ranges::random_access_range<T>);

    using Rng = decltype(std::views::iota(0, 2 * 3 * 4));
    using U = decltype(ccs::stride(std::declval<Rng>() | std::views::drop(int{}), integer{}));

    U u{};
}

TEST_CASE("z_plane_view")
{

    index_extents i{.extents = int3{2, 3, 4}};
    auto t_ = std::views::iota(0, 2 * 3 * 4);
    std::vector<int> t{std::ranges::begin(t_), std::ranges::end(t_)};

    // verify z_plane_view is valid
    auto zpv = z_plane_view(t, i, 0);
    REQUIRE(std::ranges::size(zpv) > 0);

    using T = decltype(z_plane_view(t, i, 0));
    REQUIRE(std::ranges::sized_range<T>);
    REQUIRE(std::ranges::random_access_range<T>);
}

TEST_CASE("underlying_range")
{
    using T = std::vector<int>;

    static_assert(std::same_as<underlying_range_t<T>, T>);
    static_assert(std::same_as<underlying_range_t<std::tuple<T>>, T>);
    static_assert(std::same_as<underlying_range_t<std::tuple<T, T>>, T>);
    static_assert(
        std::same_as<underlying_range_t<std::tuple<std::tuple<T>, std::tuple<T, T, T>>>,
                     T>);
}

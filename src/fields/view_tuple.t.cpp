#include "view_tuple.hpp"
#include "container_tuple.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <functional>
#include <ranges>
#include <vector>

#include "ccs_range_utils.hpp"
#include "lazy_views.hpp"

using namespace ccs;

TEST_CASE("construction")
{
    using T = std::vector<real>;

    REQUIRE(std::constructible_from<view_tuple<T&>, T&>);

    auto v = T{1, 2, 3};
    view_tuple<T&> yy{v};
    view_tuple<T&> zz{MOVE(yy)};
    view_tuple<T&> x{zz};

    REQUIRE(x == v);

    auto [y] = x;
    REQUIRE(std::ranges::equal(y, v));

    auto vv = T{3, 4, 5};
    view_tuple<T&, T&> z{v, vv};
    auto [a, b] = z;

    REQUIRE(std::ranges::equal(a, std::vector<real>{1, 2, 3}));
    REQUIRE(std::ranges::equal(b, std::vector<real>{3, 4, 5}));
    for (auto&& i : b) i *= 2;
    REQUIRE(vv == T{6, 8, 10});
}

TEST_CASE("single_view")
{
    auto a = single_view{std::views::iota(0, 10)};
    REQUIRE(std::ranges::size(a) == 10u);
    REQUIRE(std::ranges::equal(a, std::views::iota(0, 10)));

    auto b = a;
    REQUIRE(std::ranges::size(a) == 10u);
    REQUIRE(std::ranges::size(b) == std::ranges::size(a));
    REQUIRE(std::ranges::equal(b, std::views::iota(0, 10)));

    auto c = MOVE(a);
    REQUIRE(std::ranges::size(c) == 10u);
    REQUIRE(std::ranges::equal(c, std::views::iota(0, 10)));
}

TEST_CASE("Assignment with view_tuple<Vector&>")
{
    using T = std::vector<real>;
    using U = view_tuple<T&>;

    REQUIRE(std::is_assignable_v<U&, real>);
    REQUIRE(std::is_assignable_v<U&, T>);
    REQUIRE(std::is_assignable_v<U&, std::views::all_t<T&>>);

    auto v = T{1, 2, 3};

    U x{v};

    x = -1;
    REQUIRE(v == T{-1, -1, -1});
    REQUIRE(x == v);
    REQUIRE(std::ranges::equal(get<0>(x), v));

    x = std::views::iota(1, 10);
    REQUIRE(x.size() == 9u);
    REQUIRE(std::ranges::equal(v, std::views::iota(1, 10)));

    x = T{-1, -2};
    REQUIRE(x.size() == 2u);
    REQUIRE(v == T{-1, -2});
}

TEST_CASE("Assignment from Container")
{
    using T = std::vector<real>;
    using U = view_tuple<T&>;

    auto v = T{};
    U x{v};

    auto c = container_tuple<T>{std::views::iota(0, 10)};
    x = c;

    REQUIRE(x == std::views::iota(0, 10));
}

TEST_CASE("Copy with view_tuple<Vector&>")
{
    using T = std::vector<real>;

    auto v = T{1, 2, 3};

    view_tuple<T&> x{v};

    auto u = T{4, 5, 6};
    view_tuple<T&> y{u};

    x = y;
    REQUIRE(v == T{4, 5, 6});

    auto w = T{3, 4, 5, 6};
    x = view_tuple<T&>{w};
    REQUIRE(w == T{3, 4, 5, 6});
    REQUIRE(x == w);
}

TEST_CASE("Assignment with view_tuple<Vector&, Vector&>")
{
    using T = std::vector<real>;

    auto u = T{1, 2, 3};
    auto v = T{4, 5, 6, 7};

    view_tuple<T&, T&> x{u, v};

    x = -1;
    REQUIRE((x == view_tuple{ccs::repeat_n(-1, u.size()), ccs::repeat_n(-1, v.size())}));
    REQUIRE(std::ranges::equal(v, get<1>(x)));
    REQUIRE(std::ranges::equal(get<0>(x), u));

    auto q = T{6, 7, 8, 9};
    auto r = T{10, 11, 12, 13, 14};
    x = view_tuple<T&, T&>{q, r};

    REQUIRE((view_tuple{u, v} == view_tuple{q, r}));
}

TEST_CASE("Assignment with view_tuple<span>")
{
    using T = std::vector<real>;

    auto v = T{1, 2, 3};

    view_tuple<std::span<real>> x{v};

    x = -1;
    REQUIRE(v == T{-1, -1, -1});
    REQUIRE(x == v);
    REQUIRE(std::ranges::equal(get<0>(x), v));

    x = std::views::iota(1, 10);
    REQUIRE(x.size() == 3u);
    REQUIRE(v == T{1, 2, 3});

    x = T{-1, -2};
    REQUIRE(x.size() == 3u);
    REQUIRE(v == T{-1, -2, 3});
}

TEST_CASE("Assignment with view_tuple of Const")
{
    constexpr auto f = []<typename... Args>()
    {
        static_assert(!requires(view_tuple<Args...> t) { t = 1; });
    };
    f.template operator()<const std::vector<int>&>();
    f.template operator()<std::span<const int>>();
    f.template operator()<std::span<const int>&>();
}

TEST_CASE("Copying NonOutput OneTuples")
{
    using X = decltype(std::views::iota(0, 5));
    using T = view_tuple<X>;

    T t{std::views::iota(0, 10)};
    REQUIRE(t == std::views::iota(0, 10));

    {
        T u{};
        u = t;
        REQUIRE(u == std::views::iota(0, 10));
    }

    {
        T u{t};
        REQUIRE(u == std::views::iota(0, 10));
    }

    {
        T u{t};
        T v{MOVE(u)};
        REQUIRE(v == std::views::iota(0, 10));
    }

    {
        T v{};
        T u{t};
        v = MOVE(u);
        REQUIRE(v == std::views::iota(0, 10));
    }

    t = T{std::views::iota(5, 10)};
    REQUIRE(t == std::views::iota(5, 10));
}

TEST_CASE("Copying NonOutput TwoTuples")
{
    using X = decltype(std::views::iota(0, 5));
    using T = view_tuple<X, X>;

    T t{std::views::iota(0, 10), std::views::iota(5, 10)};
    {
        auto [x, y] = t;
        REQUIRE(std::ranges::equal(x, std::views::iota(0, 10)));
        REQUIRE(std::ranges::equal(y, std::views::iota(5, 10)));
    }

    {
        T u{};
        u = t;
        REQUIRE((t == u));
    }

    {
        T u{t};
        REQUIRE((t == u));
    }

    {
        T u{t};
        T v{MOVE(u)};
        REQUIRE((t == u));
    }

    {
        T v{};
        T u{t};
        v = MOVE(u);
        REQUIRE((t == v));
    }

    t = T{std::views::iota(2, 5), std::views::iota(3, 10)};
    {
        auto [x, y] = t;
        REQUIRE(std::ranges::equal(x, std::views::iota(2, 5)));
        REQUIRE(std::ranges::equal(y, std::views::iota(3, 10)));
    }
}

TEST_CASE("Non-Modifying Math OneTuples")
{
    using T = std::vector<real>;

    auto u_ = T{1, 2, 3};
    view_tuple<T&> u{u_};

    static_assert(std::tuple_size_v<std::remove_cvref_t<decltype(u)>> == 1u);

    REQUIRE(1 + u + 1 + u == T{4, 6, 8});

    auto v_ = T(3);
    view_tuple<T&> v{v_};

    // conversion
    v = 1 + u + 1 + u;
    REQUIRE(v == T{4, 6, 8});

    view_tuple<std::span<real>> w{v_};
    w = 0;
    REQUIRE(v_ == T{0, 0, 0});

    w = 1 + u + 1 + u;
    REQUIRE(w == T{4, 6, 8});
}

TEST_CASE("Non-Modifying Math TwoTuples")
{
    using T = std::vector<real>;
    using C = container_tuple<T, T>;

    auto a_ = T{1, 2, 3};
    auto b_ = T{3, 4};
    view_tuple<T&, T&> u{a_, b_};

    static_assert(std::tuple_size_v<std::remove_cvref_t<decltype(u)>> == 2u);

    {
        auto [a, b] = 1 + u + 1 + u;
        REQUIRE(std::ranges::equal(a, T{4, 6, 8}));
        REQUIRE(std::ranges::equal(b, T{8, 10}));
    }

    auto c_ = T(a_.size());
    auto d_ = T(b_.size());
    view_tuple<std::span<real>, std::span<real>> v{c_, d_};

    v = 1 + u + 1 + u;
    REQUIRE(v == C{T{4, 6, 8}, T{8, 10}});
}

TEST_CASE("Modifying Math OneTuples")
{
    using T = std::vector<real>;

    auto u_ = T{1, 2, 3};
    view_tuple<T&> u{u_};

    static_assert(std::tuple_size_v<std::remove_cvref_t<decltype(u)>> == 1u);

    u += 2;
    REQUIRE(u == T{3, 4, 5});

    auto v_ = T{1, 2, 3};
    view_tuple<T&> v{v_};

    u += v;
    REQUIRE(u == T{4, 6, 8});
}

TEST_CASE("Pipe Syntax OneTuples")
{
    using T = std::vector<int>;

    auto a = T{1, 2, 3};
    auto u = view_tuple{a};

    REQUIRE(std::ranges::equal(u | std::views::transform([](auto&& i) { return i * i; }), T{1, 4, 9}));

    auto b = T(3);
    auto v = view_tuple{b};

    v = u | std::views::transform([](auto&& i) { return i * i; });
    REQUIRE(b == T{1, 4, 9});
}

TEST_CASE("Pipe Syntax TwoTuples")
{
    const auto i = std::views::iota(0, 10);
    const auto j = std::views::iota(-10, 10);
    constexpr auto f = [](auto&& i) { return i + i; };

    auto v = view_tuple{std::views::iota(0, 10), std::views::iota(-10, 10)};

    REQUIRE(((v | std::views::transform(f)) ==
            view_tuple{ccs::zip_transform(std::plus{}, i, i), ccs::zip_transform(std::plus{}, j, j)}));
}

TEST_CASE("MultiPipe Syntax")
{
    const auto i = std::views::iota(0, 10);
    const auto j = std::views::iota(-10, 10);

    constexpr auto f = [](auto&& i) { return i + i; };
    constexpr auto g = [](auto&& i) { return i * i; };

    auto v = view_tuple{std::views::iota(0, 10), std::views::iota(-10, 10)};

    REQUIRE(((v | std::tuple{std::views::transform(f), std::views::transform(g)}) ==
            view_tuple{ccs::zip_transform(std::plus{}, i, i),
                       ccs::zip_transform(std::multiplies{}, j, j)}));
}

// type for mocking selection
template <typename R>
struct X : R {

    X() = default;
    constexpr X(R r) : R{MOVE(r)} {}

    template <ViewClosures F>
    X& operator=(F f)
    {
        auto rng = *this | f;
        R::operator=(rng);
        return *this;
    }
};

namespace std::ranges
{
template <ccs::All T>
inline constexpr bool enable_view<X<T>> = true;
}

TEST_CASE("pass through assignment - vector")
{
    using T = std::vector<int>;
    using V = view_tuple<T&>;

    constexpr auto closure =
        ccs::make_view_closure([](auto&&) { return std::views::iota(0, 10); });

    constexpr auto closure_plus1 =
        ccs::make_view_closure([](auto&&) {
            return std::views::iota(0, 10) |
                   std::views::transform([](auto&& i) { return i + 1; });
        });

    constexpr auto closure_minus1 =
        ccs::make_view_closure([](auto&&) {
            return std::views::iota(0, 10) |
                   std::views::transform([](auto&& i) { return i - 1; });
        });

    auto t = T{};
    auto a = V{t};
    static_assert(is_ref_view<decltype(get<0>(a))>::value);
    a = std::ranges::empty_view<int>{} | closure_plus1;
    REQUIRE(std::ranges::equal(t, std::views::iota(1, 11)));

    {
        t.clear();
        auto x = X{V{t}};
        x = closure_minus1;
        REQUIRE(std::ranges::equal(t, std::views::iota(-1, 9)));
    }

    {
        t.clear();
        static_assert(All<X<V>>);
        auto x = view_tuple{X{V{t}}};
        static_assert(AssignableDirect<view_tuple_base<X<V>>, decltype(closure)>);
        static_assert(AssignableDirect<view_tuple_base<X<V>>&, decltype(closure)>);
        x = closure_plus1;
        REQUIRE(std::ranges::equal(t, std::views::iota(1, 11)));
    }
}

TEST_CASE("pass through assignment - span")
{
    using T = std::span<int>;
    using V = view_tuple<T>;

    static_assert(All<T>);
    static_assert(All<T&>);
    static_assert(All<V>);
    static_assert(All<view_tuple<std::span<int>&>>);

    constexpr auto closure =
        ccs::make_view_closure([](auto&&) { return std::views::iota(0, 10); });

    constexpr auto closure_plus1 =
        ccs::make_view_closure([](auto&&) {
            return std::views::iota(0, 10) |
                   std::views::transform([](auto&& i) { return i + 1; });
        });

    constexpr auto closure_minus1 =
        ccs::make_view_closure([](auto&&) {
            return std::views::iota(0, 10) |
                   std::views::transform([](auto&& i) { return i - 1; });
        });

    auto t = std::vector<int>(10);
    auto a = V{t};
    a = std::ranges::empty_view<int>{} | closure_plus1;
    REQUIRE(std::ranges::equal(t, std::views::iota(1, 11)));

    {
        auto x = X{V{t}};
        x = closure_minus1;
        REQUIRE(std::ranges::equal(t, std::views::iota(-1, 9)));
    }

    {
        static_assert(All<X<V>>);
        auto x = view_tuple{X{V{t}}};
        static_assert(AssignableDirect<view_tuple_base<X<V>>, decltype(closure)>);
        static_assert(AssignableDirect<view_tuple_base<X<V>>&, decltype(closure)>);
        x = closure_plus1;
        REQUIRE(std::ranges::equal(t, std::views::iota(1, 11)));
    }
}

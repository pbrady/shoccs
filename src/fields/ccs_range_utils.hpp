#pragma once

#include <optional>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ccs
{

// ---------------------------------------------------------------------------
// semiregular_box<Fn>
//
// Wraps any callable Fn to make it default-constructible and assignable
// (semiregular). Uses std::optional<Fn> internally.
// Provides operator() forwarding and implicit conversion to Fn&.
// ---------------------------------------------------------------------------
template <typename Fn>
class semiregular_box
{
    std::optional<Fn> fn_;

public:
    semiregular_box() requires std::default_initializable<Fn> : fn_{Fn{}} {}
    semiregular_box() requires(!std::default_initializable<Fn>) : fn_{std::nullopt} {}

    constexpr semiregular_box(Fn fn) : fn_{std::move(fn)} {}

    semiregular_box(const semiregular_box&) = default;
    semiregular_box(semiregular_box&&) = default;

    // Custom assignment: destroy-and-reconstruct to handle non-assignable types
    // (e.g., capturing lambdas are copy-constructible but not copy-assignable)
    semiregular_box& operator=(const semiregular_box& other)
    {
        fn_.reset();
        if (other.fn_) fn_.emplace(*other.fn_);
        return *this;
    }

    semiregular_box& operator=(semiregular_box&& other) noexcept(
        std::is_nothrow_move_constructible_v<Fn>)
    {
        fn_.reset();
        if (other.fn_) fn_.emplace(std::move(*other.fn_));
        return *this;
    }

    constexpr Fn& get() & { return *fn_; }
    constexpr const Fn& get() const& { return *fn_; }
    constexpr Fn&& get() && { return std::move(*fn_); }

    constexpr operator Fn&() & { return *fn_; }
    constexpr operator const Fn&() const& { return *fn_; }
    constexpr operator Fn&&() && { return std::move(*fn_); }

    template <typename... Args>
        requires std::invocable<Fn&, Args...>
    constexpr decltype(auto) operator()(Args&&... args) &
    {
        return (*fn_)(std::forward<Args>(args)...);
    }

    template <typename... Args>
        requires std::invocable<const Fn&, Args...>
    constexpr decltype(auto) operator()(Args&&... args) const&
    {
        return (*fn_)(std::forward<Args>(args)...);
    }

    template <typename... Args>
        requires std::invocable<Fn, Args...>
    constexpr decltype(auto) operator()(Args&&... args) &&
    {
        return std::move(*fn_)(std::forward<Args>(args)...);
    }
};

// ---------------------------------------------------------------------------
// view_closure<Fn>
//
// A pipeable callable wrapper. Provides operator| so that
// rng | closure invokes closure(rng).
// ---------------------------------------------------------------------------
template <typename Fn>
struct view_closure : Fn {
    view_closure() = default;
    constexpr explicit view_closure(Fn fn) : Fn(std::move(fn)) {}

    // rng | closure  =>  closure(rng)
    template <std::ranges::viewable_range Rng>
        requires std::invocable<Fn const&, Rng>
    friend constexpr auto operator|(Rng&& rng, view_closure const& vc)
    {
        return static_cast<Fn const&>(vc)(std::forward<Rng>(rng));
    }

    template <std::ranges::viewable_range Rng>
        requires std::invocable<Fn, Rng>
    friend constexpr auto operator|(Rng&& rng, view_closure&& vc)
    {
        return static_cast<Fn&&>(vc)(std::forward<Rng>(rng));
    }

    // closure | closure  =>  composed closure (applies left first, then right)
    template <typename OtherFn>
    friend constexpr auto operator|(view_closure lhs, view_closure<OtherFn> rhs)
    {
        auto fn = [l = std::move(lhs), r = std::move(rhs)]<typename Rng>(Rng&& rng)
            requires std::invocable<const Fn&, Rng&&>
        {
            return static_cast<const OtherFn&>(r)(
                static_cast<const Fn&>(l)(std::forward<Rng>(rng)));
        };
        return view_closure<decltype(fn)>{std::move(fn)};
    }
};

// Deduction guide
template <typename Fn>
view_closure(Fn) -> view_closure<Fn>;

// ---------------------------------------------------------------------------
// make_view_closure(fn)
//
// Factory returning view_closure<Fn>.
// ---------------------------------------------------------------------------
struct make_view_closure_fn {
    template <typename Fn>
    constexpr auto operator()(Fn fn) const
    {
        return view_closure<Fn>{std::move(fn)};
    }
};
inline constexpr make_view_closure_fn make_view_closure{};

// ---------------------------------------------------------------------------
// bind_back(fn, args...)
//
// Returns a callable that, when invoked with (cargs...), calls
// fn(cargs..., args...).  All arguments are stored by decay-copy.
// ---------------------------------------------------------------------------
namespace detail
{
template <typename Fn, typename... BoundArgs>
struct bind_back_t {
    Fn fn;
    std::tuple<BoundArgs...> bound;

    template <typename... CallArgs>
        requires std::invocable<const Fn&, CallArgs..., const BoundArgs&...>
    constexpr decltype(auto) operator()(CallArgs&&... cargs) const&
    {
        return std::apply(
            [&](const auto&... bargs) -> decltype(auto) {
                return fn(std::forward<CallArgs>(cargs)..., bargs...);
            },
            bound);
    }

    template <typename... CallArgs>
        requires std::invocable<Fn&, CallArgs..., BoundArgs&...>
    constexpr decltype(auto) operator()(CallArgs&&... cargs) &
    {
        return std::apply(
            [&](auto&... bargs) -> decltype(auto) {
                return fn(std::forward<CallArgs>(cargs)..., bargs...);
            },
            bound);
    }

    template <typename... CallArgs>
        requires std::invocable<Fn, CallArgs..., BoundArgs...>
    constexpr decltype(auto) operator()(CallArgs&&... cargs) &&
    {
        return std::apply(
            [&](auto&&... bargs) -> decltype(auto) {
                return std::move(fn)(std::forward<CallArgs>(cargs)...,
                                     std::move(bargs)...);
            },
            std::move(bound));
    }
};
} // namespace detail

struct bind_back_fn {
    template <typename Fn, typename Arg1, typename... Args>
    constexpr auto operator()(Fn&& fn, Arg1&& arg1, Args&&... args) const
    {
        return detail::bind_back_t<std::decay_t<Fn>,
                                   std::decay_t<Arg1>,
                                   std::decay_t<Args>...>{
            std::forward<Fn>(fn),
            {std::forward<Arg1>(arg1), std::forward<Args>(args)...}};
    }
};
inline constexpr bind_back_fn bind_back{};

// ---------------------------------------------------------------------------
// compose(f, g)
//
// Returns a callable h such that h(args...) = f(g(args...)).
// Parameter order: compose(second, first), i.e. h(x) = second(first(x)).
// ---------------------------------------------------------------------------
struct compose_fn {
    template <typename F, typename G>
    constexpr auto operator()(F f, G g) const
    {
        return [f = std::move(f), g = std::move(g)](auto&&... args)
                   -> decltype(auto) requires requires {
                       f(g(std::forward<decltype(args)>(args)...));
                   }
        {
            return f(g(std::forward<decltype(args)>(args)...));
        };
    }
};
inline constexpr compose_fn compose{};

} // namespace ccs

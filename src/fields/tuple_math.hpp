#pragma once

#include "tuple_utils.hpp"

#include <ranges>
#include <tuple>

#include "lazy_views.hpp"

namespace ccs::detail
{

template <typename T>
struct tuple_math;

class tuple_math_access
{
    template <typename T>
    friend class tuple_math;
};

template <typename T>
struct tuple_math {

private:
#define SHOCCS_GEN_OPERATORS(op, f)                                                      \
    template <typename U, Numeric V>                                                     \
        requires(std::derived_from<std::remove_cvref_t<U>, T> && OutputTuple<U, V>)      \
    constexpr friend U&& op(U&& u, V v)                                                  \
    {                                                                                    \
        for_each(                                                                        \
            [v](auto&& rng) {                                                            \
                for (auto&& x : rng) x f v;                                              \
            },                                                                           \
            u);                                                                          \
                                                                                         \
        return FWD(u);                                                                   \
    }                                                                                    \
                                                                                         \
    template <std::derived_from<T> U, TupleLike V>                                       \
        requires OutputTuple<U, V>                                                       \
    constexpr friend U& op(U& u, V&& v)                                                  \
    {                                                                                    \
        for_each(                                                                        \
            [](auto&& out, auto&& in) {                                                  \
                auto it_o = std::ranges::begin(out);                                      \
                auto it_i = std::ranges::begin(in);                                      \
                for (; it_o != std::ranges::end(out); ++it_o, ++it_i) *it_o f *it_i;    \
            },                                                                           \
            u,                                                                           \
            FWD(v));                                                                     \
                                                                                         \
        return u;                                                                        \
    }                                                                                    \
                                                                                         \
    template <typename U, TupleLike V>                                                   \
        requires(std::derived_from<std::remove_cvref_t<U>, T> && !OutputTuple<U, V> &&   \
                 !SimilarTuples<U, V>)                                                   \
    constexpr friend U& op(U&& u, V&& v)                                                 \
    {                                                                                    \
        auto tp = [&]<auto... Is>(std::index_sequence<Is...>)                            \
        {                                                                                \
            return tuple_cat<U>(get<Is>(u).apply(v)...);                                 \
        }                                                                                \
        (sequence<U>);                                                                   \
        static_assert(OutputTuple<U, decltype(tp)>);                                     \
        return u f tp;                                                                   \
    }

    SHOCCS_GEN_OPERATORS(operator+=, +=)
    SHOCCS_GEN_OPERATORS(operator*=, *=)
    SHOCCS_GEN_OPERATORS(operator-=, -=)
    SHOCCS_GEN_OPERATORS(operator/=, /=)

#undef SHOCCS_GEN_OPERATORS

#define SHOCCS_GEN_OPERATORS(op, f)                                                      \
    template <typename U, Numeric V>                                                     \
        requires std::derived_from<std::remove_cvref_t<U>, T>                            \
    friend constexpr auto op(U&& u, V v)                                                 \
    {                                                                                    \
        return transform(                                                                \
            [v](auto&& rng) {                                                            \
                const auto sz = std::ranges::size(rng);                                    \
                return ccs::zip_transform(f, FWD(rng), ccs::repeat_n(v, sz));            \
            },                                                                           \
            FWD(u));                                                                     \
    }                                                                                    \
                                                                                         \
    template <typename U, Numeric V>                                                     \
        requires std::derived_from<std::remove_cvref_t<U>, T>                            \
    friend constexpr auto op(V v, U&& u)                                                 \
    {                                                                                    \
        return transform(                                                                \
            [v](auto&& rng) {                                                            \
                const auto sz = std::ranges::size(rng);                                    \
                return ccs::zip_transform(f, ccs::repeat_n(v, sz), FWD(rng));            \
            },                                                                           \
            FWD(u));                                                                     \
    }                                                                                    \
                                                                                         \
    template <typename U, typename V>                                                    \
    requires std::derived_from<std::remove_cvref_t<U>, T> &&                             \
        mp_similar<std::remove_cvref_t<U>,                                               \
                   std::remove_cvref_t<V>>::value friend constexpr auto                  \
        op(U&& u, V&& v)                                                                 \
    {                                                                                    \
        return transform(                                                                \
            [](auto&& a, auto&& b) { return ccs::zip_transform(f, FWD(a), FWD(b)); },    \
            FWD(u),                                                                      \
            FWD(v));                                                                     \
    }

    SHOCCS_GEN_OPERATORS(operator+, std::plus{})
    SHOCCS_GEN_OPERATORS(operator-, std::minus{})
    SHOCCS_GEN_OPERATORS(operator*, std::multiplies{})
    SHOCCS_GEN_OPERATORS(operator/, std::divides{})

#undef SHOCCS_GEN_OPERATORS
};

} // namespace ccs::detail

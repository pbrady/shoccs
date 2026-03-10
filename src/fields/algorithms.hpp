#pragma once

#include "vector.hpp"

#include <algorithm>
#include <ranges>

namespace ccs
{

template <TupleLike T>
constexpr auto minmax(T&& t)
{
    using Rng = underlying_range_t<T>;
    using V = std::ranges::range_value_t<Rng>;
    using R = std::ranges::minmax_result<V>;
    return transform_reduce(
        [](auto&& rng) {
            // Need to safeguard the call to minmax in case of an empty range.  Empty
            // ranges are common with boundary condition based selection
            if (std::ranges::begin(rng) != std::ranges::end(rng))
                return std::ranges::minmax(FWD(rng));
            else
                return R{std::numeric_limits<V>::max(), std::numeric_limits<V>::lowest()};
        },
        FWD(t),
        [](auto&& acc, auto&& item) {
            return R{std::ranges::min(acc.min, item.min), std::ranges::max(acc.max, item.max)};
        },
        R{std::numeric_limits<V>::max(), std::numeric_limits<V>::lowest()});
}

template <TupleLike T>
constexpr auto max(T&& t)
{
    using Rng = underlying_range_t<T>;
    using V = std::ranges::range_value_t<Rng>;
    return transform_reduce(
        [](auto&& rng) {
            if (std::ranges::begin(rng) != std::ranges::end(rng))
                return std::ranges::max(FWD(rng));
            else
                return std::numeric_limits<V>::lowest();
        },
        FWD(t),
        [](auto&& acc, auto&& item) { return std::ranges::max(acc, item); },
        V{std::numeric_limits<V>::lowest()});
}

template <Vector T, Vector U>
constexpr auto dot(T&& t, U&& u)
{
    auto [x, y, z] = t * u;
    return x + y + z;
}

} // namespace ccs

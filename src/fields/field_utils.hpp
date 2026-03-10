#pragma once

#include "field_fwd.hpp"

#include "lazy_views.hpp"

namespace ccs
{

template <Field... T, typename F>
constexpr void for_each_scalar(F&& f, T&&... t)
{
    const auto n = std::get<0>(std::forward_as_tuple(t...)).nscalars();
    for (int i = 0; i < n; ++i) {
        f(t.scalars()[i]...);
    }
}

template <Field... T, typename F>
constexpr void for_each_vector(F&& f, T&&... t)
{
    const auto n = std::get<0>(std::forward_as_tuple(t...)).nvectors();
    for (int i = 0; i < n; ++i) {
        f(t.vectors()[i]...);
    }
}

template <Field... T, typename F>
constexpr void for_each(F&& f, T&&... t)
{
    for_each_scalar(f, t...);
    for_each_vector(FWD(f), FWD(t)...);
}

template <Field... T, typename F>
constexpr auto transform_scalar(F&& f, T&&... t)
{
    return ccs::zip_transform(f, FWD(t).scalars()...);
}

template <Field... T, typename F>
constexpr auto transform_vector(F&& f, T&&... t)
{
    return ccs::zip_transform(f, FWD(t).vectors()...);
}

template <Field... T, typename F>
constexpr auto transform(F&& f, T&&... t)
{
    return detail::field{transform_scalar(f, t...), transform_vector(f, t...)};
}

template <Field F>
auto ssize(F&& f)
{
    if (f.nscalars() > 0) {
        return system_size{f.nscalars(), f.nvectors(), ssize(f.scalars(0))};
    } else {
        return system_size{};
    }
}
} // namespace ccs

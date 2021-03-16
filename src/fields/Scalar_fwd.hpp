#pragma once

#include <concepts>
#include <type_traits>

#include "Tuple_fwd.hpp"

namespace ccs::field::tuple
{

namespace detail
{

template <traits::OneTuple U, traits::ThreeTuple V>
using Scalar = Tuple<U, V>;
}

namespace traits
{
template <typename>
struct is_scalar : std::false_type {
};

template <OneTuple U, ThreeTuple V>
struct is_scalar<tuple::detail::Scalar<U, V>> : std::true_type {
};

template <typename T>
concept ScalarType = is_scalar<std::remove_cvref_t<T>>::value;

} // namespace traits

} // namespace ccs::field::tuple

namespace ccs::field
{

template <typename T>
using Scalar = tuple::detail::Scalar<Tuple<T>, Tuple<T, T, T>>;

using ScalarView_Mutable = Scalar<std::span<real>>;
using ScalarView_Const = Scalar<std::span<const real>>;
} // namespace ccs::field

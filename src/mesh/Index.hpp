#pragma once

#include "types.hpp"
#include <tuple>

namespace ccs::mesh
{

struct IndexExtents {
    int3 extents;

    constexpr operator const int3&() const { return extents; }
    constexpr operator int3&() { return extents; }

    constexpr integer operator()(int3 ijk) const
    {
        auto&& [nx, ny, nz] = extents;
        return (integer)ijk[0] * ny * nz + (integer)ijk[1] * nz + ijk[2];
    }

    constexpr auto operator[](int i) const { return extents[i]; }
    constexpr auto& operator[](int i) { return extents[i]; }
};

template <std::size_t I, typename T>
requires std::same_as<IndexExtents, std::remove_cvref_t<T>> constexpr decltype(auto)
get(T&& t) noexcept
{
    return std::get<I>(FWD(t).extents);
}
} // namespace ccs::mesh

// specialize tuple_size
namespace std
{
template <>
struct tuple_size<ccs::mesh::IndexExtents>
    : tuple_size<decltype(declval<ccs::mesh::IndexExtents>().extents)> {
};

template <size_t I>
struct tuple_element<I, ccs::mesh::IndexExtents>
    : tuple_element<I, decltype(declval<ccs::mesh::IndexExtents>().extents)> {
};
} // namespace std
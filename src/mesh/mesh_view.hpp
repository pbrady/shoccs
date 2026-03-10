#pragma once

#include "cartesian.hpp"
#include "indexing.hpp"

#include <vector>

// Include some facilities here for traversing/viewing the mesh.  These functions
// make use of ranges and generators so they are separate from the main mesh class
// since most objects will need to know about the mesh but do not need to pay the
// compile penalty for including ranges.

namespace ccs::mesh
{

template <int I = 2>
std::vector<real3> location_view(const cartesian& m)
{
    constexpr int F = index::dir<I>::fast;
    constexpr int S = index::dir<I>::slow;
    real3 loc;

    const auto iline = m.line(I);
    const auto fline = m.line(F);
    const auto sline = m.line(S);

    std::vector<real3> result;
    result.reserve(iline.n * fline.n * sline.n);

    for (int s = 0; s < sline.n; s++) {
        loc[S] = sline.min + sline.h * s;
        for (int f = 0; f < fline.n; f++) {
            loc[F] = fline.min + fline.h * f;
            for (int i = 0; i < iline.n; i++) {
                loc[I] = iline.min + iline.h * i;
                result.push_back(loc);
            }
        }
    }
    return result;
}

template <int I>
std::vector<real3> location_view(const cartesian& m, int i)
{
    constexpr int F = index::dir<I>::fast;
    constexpr int S = index::dir<I>::slow;
    real3 loc;

    const auto iline = m.line(I);
    const auto fline = m.line(F);
    const auto sline = m.line(S);

    loc[I] = iline.min + iline.h * (i < 0 ? i + iline.n : i);

    std::vector<real3> result;
    result.reserve(fline.n * sline.n);

    for (int s = 0; s < sline.n; s++) {
        loc[S] = sline.min + sline.h * s;
        for (int f = 0; f < fline.n; f++) {
            loc[F] = fline.min + fline.h * f;
            result.push_back(loc);
        }
    }
    return result;
}

} // namespace ccs::mesh

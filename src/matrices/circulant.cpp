#include "circulant.hpp"

#include "kokkos_types.hpp"

#include <cassert>
#include <numeric>

namespace ccs::matrix
{

circulant::circulant(integer rows, std::span<const real> coeffs)
    : matrix_base{rows, rows + (integer)coeffs.size() - 1, (integer)coeffs.size() / 2},
      v{coeffs}
{
}

circulant::circulant(integer rows,
                     integer row_offset,
                     integer stride,
                     std::span<const real> coeffs)
    : matrix_base{rows, rows + (integer)coeffs.size() - 1, row_offset, -1, stride},
      v{coeffs}
{
}

template <typename Op>
void circulant::operator()(std::span<const real> x, std::span<real> b, Op op) const
{
    assert(row_offset() >= stride() * (size() / 2));
    assert((integer)b.size() >= row_offset() + rows() * stride());
    assert((integer)x.size() >= row_offset() + (rows() + (size() / 2) - 1) * stride());
    // move input and output spans to correct position
    const auto st = stride();
    x = x.subspan(row_offset() - st * (size() / 2));
    b = b.subspan(row_offset());

    const auto nr = rows();
    const auto* vp = v.data();
    const auto vs = static_cast<integer>(v.size());
    const auto* xp = x.data();
    auto* bp = b.data();

    // Kokkos forbids nested parallel_for. When circulant is called from within
    // block::operator()'s parallel_for, fall back to serial loops. The outer
    // block-level parallelism still provides the main performance benefit.
    // Standalone calls (not nested) use parallel_for over rows.
    const bool nested = execution_space::in_parallel();

    if (st == 1) {
        if (nested) {
            for (integer i = 0; i < nr; i++) {
                auto dot = std::inner_product(vp, vp + vs, xp + i, 0.0);
                op(bp[i], dot);
            }
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<execution_space>(0, nr),
                [=](int i) {
                    auto dot = std::inner_product(vp, vp + vs, xp + i, 0.0);
                    op(bp[i], dot);
                });
            Kokkos::fence();
        }
    } else {
        if (nested) {
            for (integer i = 0; i < nr; i++) {
                real dot = 0.0;
                for (integer j = 0; j < vs; j++)
                    dot += vp[j] * xp[(i + j) * st];
                op(bp[i * st], dot);
            }
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<execution_space>(0, nr),
                [=](int i) {
                    real dot = 0.0;
                    for (integer j = 0; j < vs; j++)
                        dot += vp[j] * xp[(i + j) * st];
                    op(bp[i * st], dot);
                });
            Kokkos::fence();
        }
    }
}

template void
circulant::operator()<eq_t>(std::span<const real>, std::span<real>, eq_t) const;
template void
circulant::operator()<plus_eq_t>(std::span<const real>, std::span<real>, plus_eq_t) const;

} // namespace ccs::matrix

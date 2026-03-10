#include "circulant.hpp"

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

    if (st == 1) {
        for (integer i = 0; i < rows(); i++) {
            auto dot = std::inner_product(v.begin(), v.end(), x.data() + i, 0.0);
            op(b[i], dot);
        }
    } else {
        for (integer i = 0; i < rows(); i++) {
            real dot = 0.0;
            for (integer j = 0; j < size(); j++)
                dot += v[j] * x[(i + j) * st];
            op(b[i * st], dot);
        }
    }
}

template void
circulant::operator()<eq_t>(std::span<const real>, std::span<real>, eq_t) const;
template void
circulant::operator()<plus_eq_t>(std::span<const real>, std::span<real>, plus_eq_t) const;

} // namespace ccs::matrix

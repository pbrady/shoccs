#include "csr.hpp"

#include <algorithm>

namespace ccs::matrix
{

csr csr::builder::to_csr(integer nrows)
{
    std::vector<int> u(nrows + 1);

    std::ranges::sort(p);
    auto first = p.begin();
    auto last = p.end();

    for (integer i = 0; i < nrows; i++) {
        u[i + 1] = u[i];
        while (first != last && first->row == i) {
            ++u[i + 1];
            ++first;
        }
    }

    std::vector<real> w_vec;
    std::vector<integer> v_vec;
    w_vec.reserve(p.size());
    v_vec.reserve(p.size());
    for (auto& pt : p) {
        w_vec.push_back(pt.v);
        v_vec.push_back(pt.col);
    }
    return csr{w_vec, v_vec, u};
}

void csr::operator()(std::span<const real> x, std::span<real> b) const
{
    for (integer row = 0; row < rows(); row++)
        for (integer i = u[row]; i < u[row + 1]; i++) b[row] += w[i] * x[v[i]];
}

std::span<const integer> csr::column_indices(integer row) const
{
    integer r0 = u[row];
    integer r1 = u[row + 1];
    return std::span(v.data() + r0, r1 - r0);
}

std::span<const real> csr::column_coefficients(integer row) const
{
    integer r0 = u[row];
    integer r1 = u[row + 1];
    return std::span(w.data() + r0, r1 - r0);
}

} // namespace ccs::matrix

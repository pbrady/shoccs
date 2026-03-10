#pragma once

#include "common.hpp"
#include "matrix_visitor.hpp"
#include <vector>

#include <algorithm>
#include <ranges>

namespace ccs::matrix
{

// Simple contiguous storage for dense matrix with lazy operators
class dense : public matrix_base
{
    std::vector<real> v;
    flag f;

public:
    dense() = default;

    template <std::ranges::input_range R>
    dense(integer rows, integer columns, R&& rng, flag boundary = 0)
        : matrix_base{rows, columns}, v(rows * columns), f{boundary}
    {
        std::ranges::copy(rng | std::views::take(v.size()), v.begin());
    }

    template <std::ranges::input_range R>
    dense(integer rows,
          integer columns,
          integer row_offset,
          integer col_offset,
          integer stride,
          R&& rng,
          flag boundary = 0)
        : matrix_base{rows, columns, row_offset, col_offset, stride},
          v(rows * columns),
          f{boundary}
    {
        std::ranges::copy(rng | std::views::take(v.size()), v.begin());
    }

    auto size() const noexcept { return v.size(); }

    template <typename Op = eq_t>
    void operator()(std::span<const real> x, std::span<real> b, Op op = {}) const;

    std::span<const real> data() const { return v; }
    flag flags() const { return f; }
    void flags(flag f_) { f = f_; }
    void visit(visitor& v) const { v.visit(*this); };
};
} // namespace ccs::matrix

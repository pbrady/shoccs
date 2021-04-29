#include "InnerBlock.hpp"

namespace ccs::matrix
{
// Block matrix arising from method-of-lines discretization along a line.  A full domain
// discretization requires many of these to be embedded at various starting rows.

InnerBlock::InnerBlock(Dense&& left, Circulant&& i, Dense&& right)
    : Common{left.rows() + i.rows() + right.rows(),
             left.rows() + i.rows() + right.rows()},
      left_boundary{std::move(left)},
      interior{std::move(i)},
      right_boundary{std::move(right)}
{
}

InnerBlock::InnerBlock(integer columns,
                       integer row_offset,
                       integer col_offset,
                       integer stride,
                       Dense&& left,
                       Circulant&& i,
                       Dense&& right)
    : Common{left.rows() + i.rows() + right.rows(),
             columns,
             row_offset,
             col_offset,
             stride},
      left_boundary{MOVE(left)},
      interior{MOVE(i)},
      right_boundary{MOVE(right)}
{
    // set up the offset and stride of the component matrices
    left_boundary.row_offset(row_offset).col_offset(col_offset).stride(stride);
    interior.row_offset(row_offset + stride * left_boundary.rows()).stride(stride);
    right_boundary
        .row_offset(row_offset + stride * (left_boundary.rows() + interior.rows()))
        .col_offset(col_offset + stride * (columns - right_boundary.columns()))
        .stride(stride);
}

template <typename Op>
void InnerBlock::operator()(std::span<const real> x, std::span<real> b, Op op) const
{
    left_boundary(x, b, op);
    interior(x, b, op);
    right_boundary(x, b, op);
}

template void
InnerBlock::operator()<eq_t>(std::span<const real>, std::span<real>, eq_t) const;

template void InnerBlock::operator()<plus_eq_t>(std::span<const real>,
                                                std::span<real>,
                                                plus_eq_t) const;

} // namespace ccs::matrix
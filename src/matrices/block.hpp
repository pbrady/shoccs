#pragma once

#include "inner_block.hpp"

#include "kokkos_types.hpp"

#include <cassert>
#include <concepts>

namespace ccs::matrix
{
// Block matrix arising from method-of-lines discretization over whole domain.
// Due to the requirements of a cut-cell mesh, the InnerBlocks may not be adjacent to
// eachother.  To simplify construction, a builder class is exposed which computes all
// the zero locations at the end of the construction process
class block
{
    std::vector<inner_block> blocks;

public:
    block() = default;

    block(std::vector<inner_block>&& blocks) : blocks{std::move(blocks)} {}

    integer rows() const
    {
        if (blocks.empty()) return 0;
        const auto& b = blocks.back();
        // doesn't properly handle case when last point in domain is inside an object
        return b.row_offset() + b.rows() * b.stride();
    }

    template <typename Op = eq_t>
    void operator()(std::span<const real> x, std::span<real> b, Op op = {}) const
    {
        const auto n = static_cast<int>(blocks.size());
        const auto* bp = blocks.data();
        Kokkos::parallel_for(
            Kokkos::RangePolicy<execution_space>(0, n),
            [=](int i) { bp[i](x, b, op); });
        Kokkos::fence();
    }

    void visit(visitor& v) const
    {
        for (auto&& block : blocks) { block.visit(v); }
    }

    struct builder;
};

struct block::builder {
    std::vector<inner_block> b;

    builder() = default;

    builder(integer n) { b.reserve(n); }

    template <typename... Args>
        requires std::constructible_from<inner_block, Args...>
    void add_inner_block(Args&&... args) { b.emplace_back(std::forward<Args>(args)...); }

    block to_block() &&
    {
#ifndef NDEBUG
        // Verify output row ranges are disjoint.
        for (std::size_t i = 0; i < b.size(); ++i) {
            auto lo_i = b[i].row_offset();
            auto hi_i = lo_i + b[i].rows() * b[i].stride();
            for (std::size_t j = i + 1; j < b.size(); ++j) {
                auto lo_j = b[j].row_offset();
                auto hi_j = lo_j + b[j].rows() * b[j].stride();
                assert((hi_i <= lo_j || hi_j <= lo_i) &&
                       "block inner_blocks have overlapping output row ranges");
            }
        }
#endif
        return block{MOVE(b)};
    }
};
} // namespace ccs::matrix

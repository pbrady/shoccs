#pragma once

namespace ccs
{

struct divergence {

    template <typename U>
    constexpr int operator()(U&&)
    {
        return 0;
    }
};
} // namespace ccs

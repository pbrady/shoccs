#pragma once

#include "types.hpp"

#include "boundaries.hpp"
#include "fields/scalar.hpp"
#include "matrices/block.hpp"
#include "matrices/csr.hpp"
#include "mesh/mesh.hpp"
#include "stencils/stencil.hpp"

namespace ccs
{
class derivative
{
    int dir;
    // Operators for updating field data
    matrix::block O;
    matrix::csr B;
    matrix::csr N;
    // operators for updating boundary data
    matrix::csr Bf;
    matrix::csr Br;
    std::vector<real> interior_c;

public:
    derivative() = default;

    derivative(int dir,
               const mesh& m,
               const stencil& st,
               const bcs::Grid& grid_bcs,
               const bcs::Object& object_bcs);

    // operator for when neumann conditions are not needed
    template <typename Op = eq_t>
        requires(!Scalar<Op>)
    void operator()(scalar_view, scalar_span, Op op = {}) const;

    // operaotr for when neumann conditions may be applied
    template <typename Op = eq_t>
        requires(!Scalar<Op>)
    void operator()(scalar_view field_values,
                    scalar_view derivative_values,
                    scalar_span,
                    Op op = {}) const;
};
} // namespace ccs

#include "stencil.hpp"

#include <algorithm>

#include <cmath>

/// tension_E4u_1 — uniform-mesh E4-order boundary closure derived from a
/// tension-spline RBF interpolant. Runtime-parameterized by `sigma`, the
/// tension-kernel shape parameter. The constructor solves the small RBF linear
/// system once and caches the resulting 5x7 coefficient block in
/// `cached_coeffs`; `nbs_floating` reads from the cache.
///
/// NOTE: skeleton only. The solver body is filled in by plan 42.5d. Until
/// then, `cached_coeffs` remains zeroed and `nbs_floating`/`nbs_dirichlet`
/// return a zero-filled span.

namespace ccs::stencils
{
struct tension_E4u_1 {

    static constexpr int P = 2;
    static constexpr int R = 5;
    static constexpr int T = 7;
    static constexpr int X = 0;

    real sigma{};
    std::array<real, R * T> cached_coeffs{};

    tension_E4u_1() = default;
    explicit tension_E4u_1(real sigma_in) : sigma{sigma_in}, cached_coeffs{}
    {
        // Solver call inserted in 42.5d.
    }

    info query_max() const { return {P, R, T, X}; }
    info query(bcs::type b) const
    {
        switch (b) {
        case bcs::Dirichlet:
            return {P, R - 1, T, 0};
        case bcs::Floating:
            return {P, R, T, 0};
        case bcs::Neumann:
            return {};
        default:
            return {};
        }
    }
    interp_info query_interp() const { return {}; }

    std::span<const real> interp_interior(real, std::span<real> c) const { return c; }

    std::span<const real> interp_wall(int, real, real, std::span<real> c, bool) const
    {
        return c;
    }

    std::span<const real> interior(real h, std::span<real> c) const
    {
        c[0] = 1 / (12 * h);
        c[1] = -2 / (3 * h);
        c[2] = 0;
        c[3] = -c[1];
        c[4] = -c[0];

        return c.subspan(0, 2 * P + 1);
    }

    std::span<const real> nbs(real h,
                              bcs::type b,
                              real psi,
                              bool right,
                              std::span<real> c,
                              std::span<real>) const
    {
        switch (b) {
        case bcs::Floating:
            return nbs_floating(h, psi, c.subspan(0, R * T), right);
        case bcs::Dirichlet:
            return nbs_dirichlet(h, psi, c.subspan(0, (R - 1) * T), right);
        default:
            return c;
        }
    }

    std::span<const real>
    nbs_floating(real, real, std::span<real> c, bool) const
    {
        std::fill(c.begin(), c.end(), 0.0);
        return c;
    }

    std::span<const real>
    nbs_dirichlet(real, real, std::span<real> c, bool) const
    {
        std::fill(c.begin(), c.end(), 0.0);
        return c;
    }

    void nbs_neumann(real, real, std::span<real>, std::span<real>, bool) const {}
};

} // namespace ccs::stencils

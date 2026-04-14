#include "stencil.hpp"

#include <algorithm>
#include <array>
#include <cmath>

/// multiquadric_E4u_1 — uniform-mesh E4-order boundary closure derived from
/// an inverse-multiquadric-style RBF interpolant using the standard
/// multiquadric kernel φ(r; ε) = √(1 + (ε r)^2). Runtime-parameterized by
/// `epsilon`, the multiquadric shape parameter. The constructor solves the
/// small RBF linear system once and caches the resulting 5x7 coefficient
/// block in `cached_coeffs`; `nbs_floating` reads from the cache.
///
/// Skeleton ships with a stub `solve_multiquadric_coefficients` that only
/// populates the hardcoded row-4 classical E4 centered stencil and zero-fills
/// the remaining rows. The RBF+polynomial augmented solve is wired in 42.6f.

namespace ccs::stencils
{
namespace
{

// Placeholder solver — 42.6e ships a zero-filled boundary block plus the
// hardcoded row-4 classical E4 centered first-derivative stencil.
// 42.6f replaces rows 0..3 with the solved multiquadric-RBF weights.
void solve_multiquadric_coefficients(real /*epsilon*/, std::array<real, 5 * 7>& out)
{
    std::fill(out.begin(), out.end(), 0.0);
    // Row 4: classical E4 centered stencil at h=1.
    out[4 * 7 + 0] = 0.0;
    out[4 * 7 + 1] = 0.0;
    out[4 * 7 + 2] = 1.0 / 12.0;
    out[4 * 7 + 3] = -2.0 / 3.0;
    out[4 * 7 + 4] = 0.0;
    out[4 * 7 + 5] = 2.0 / 3.0;
    out[4 * 7 + 6] = -1.0 / 12.0;
}

} // namespace

struct multiquadric_E4u_1 {

    static constexpr int P = 2;
    static constexpr int R = 5;
    static constexpr int T = 7;
    static constexpr int X = 0;

    real epsilon{};
    std::array<real, R * T> cached_coeffs{};

    multiquadric_E4u_1() = default;
    explicit multiquadric_E4u_1(real epsilon_in) : epsilon{epsilon_in}, cached_coeffs{}
    {
        solve_multiquadric_coefficients(epsilon, cached_coeffs);
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
    nbs_floating(real h, real, std::span<real> c, bool right) const
    {
        std::copy(cached_coeffs.begin(), cached_coeffs.end(), c.begin());

        for (auto&& v : c) v /= h;
        if (right) {
            for (auto&& v : c) v *= -1;
            std::ranges::reverse(c);
        }

        return c;
    }

    std::span<const real>
    nbs_dirichlet(real h, real, std::span<real> c, bool right) const
    {
        // Dirichlet drops the first (wall) row of the floating block.
        std::copy(cached_coeffs.begin() + T, cached_coeffs.end(), c.begin());

        for (auto&& v : c) v /= h;
        if (right) {
            for (auto&& v : c) v *= -1;
            std::ranges::reverse(c);
        }

        return c;
    }

    void nbs_neumann(real, real, std::span<real>, std::span<real>, bool) const {}
};

} // namespace ccs::stencils

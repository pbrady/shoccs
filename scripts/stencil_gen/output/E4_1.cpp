#include "stencil.hpp"

#include <algorithm>

#include <cmath>

namespace ccs::stencils
{
struct E4_1 {

    static constexpr int P = 2;
    static constexpr int R = 4;
    static constexpr int T = 7;
    static constexpr int X = 0;

    std::array<real, 4> alpha;

    E4_1() = default;
    E4_1(std::span<const real> a)
    {
        copy_zero_padded(a, alpha);
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
        c = c.subspan(0, 2 * 2 + 1);
        c[0] = 0.083333333333333329;
        c[1] = -0.66666666666666663;
        c[2] = 0;
        c[3] = 0.66666666666666663;
        c[4] = -0.083333333333333329;
        for (auto&& v : c) v /= h;
        return c;
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
            return nbs_floating(h, psi, c, right);
        case bcs::Dirichlet:
            return nbs_dirichlet(h, psi, c, right);
        default:
            return c;
        }
    }

    std::span<const real>
    nbs_floating(real h, real psi, std::span<real> c, bool right) const
    {
        c = c.subspan(0, R * T);

        real t5 = 11*psi;
        real t6 = psi * psi;
        real t7 = 1.0 / (psi * psi * psi + t5 + 6*t6 + 6);
        real t8 = 6*psi;
        real t9 = t7*(6 - t8);
        real t10 = 1.0 / (psi + 1);
        real t11 = 3*t6;
        real t12 = t10*(-psi - t11 - 4);
        real t13 = 2*psi;
        real t14 = t13 + 2;
        real t15 = 1.0 / (t14);
        real t16 = 11*t6 + 6;
        real t17 = psi + 2;
        real t18 = 1.0 / (t17);
        real t19 = 3*psi;
        real t20 = t19 + 12;
        real t21 = t18*(t11 + t20);
        real t22 = 1.0 / (t13 + 4);
        real t23 = 1.0 / (psi + 3);
        real t24 = t23*(-t20 - t6);
        real t25 = 1.0 / (t8 + 18);
        real t26 = 1 - psi;
        real t27 = t19 - 2*t6;
        real t28 = -t6;
        real t29 = 12*t6;

        c[0] = alpha[0]*t9 + t7*(t5 - 11);
        c[1] = psi*(alpha[0] - 11.0 / 6);
        c[2] = alpha[0]*t12 + t15*(-5*psi + t16);
        c[3] = alpha[0]*t21 + t22*(8*psi - t16);
        c[4] = alpha[0]*t24 + t25*(-9*psi + t16);
        c[5] = alpha[0]*psi + alpha[0]*t26;
        c[6] = 0;
        c[7] = alpha[1]*t9 + t7*(t13 - 2);
        c[8] = psi*(alpha[1] - 1.0 / 3);
        c[9] = alpha[1]*t12 + t15*(-t27 - 1);
        c[10] = alpha[1]*t21 + t18*(t14 + t28);
        c[11] = alpha[1]*t24 + t25*(-t27 - 3);
        c[12] = alpha[1]*psi + alpha[1]*t26;
        c[13] = 0;
        c[14] = alpha[2]*t9 + alpha[3]*t7*(24 - 24*psi) + t26*t7;
        c[15] = psi*(alpha[2] + 4*alpha[3] + 1.0 / 6);
        c[16] = alpha[2]*t12 + alpha[3]*t10*(-t19 - t29 - 15) + t15*(-t17 - t6);
        c[17] = alpha[2]*t21 + alpha[3]*t18*(8*psi + t29 + 40) + t22*(t6 + 2);
        c[18] = alpha[2]*t24 + alpha[3]*t23*(-4*t6 - t8 - 30) + t25*(t19 + t28 + 6);
        c[19] = alpha[2]*psi + alpha[2]*t26;
        c[20] = alpha[3]*psi + alpha[3]*t26;
        c[21] = 0;
        c[22] = 0;
        c[23] = 1.0 / 12;
        c[24] = -2.0 / 3;
        c[25] = 0;
        c[26] = 2.0 / 3;
        c[27] = -1.0 / 12;

        for (auto&& v : c) v /= h;
        if (right) {
            for (auto&& v : c) v *= -1;
            std::ranges::reverse(c);
        }

        return c;
    }

    std::span<const real>
    nbs_dirichlet(real h, real psi, std::span<real> c, bool right) const
    {
        c = c.subspan(0, (R - 1) * T);

        real t5 = 2*psi;
        real t6 = psi * psi;
        real t7 = 1.0 / (psi * psi * psi + 11*psi + 6*t6 + 6);
        real t8 = 6*psi;
        real t9 = t7*(6 - t8);
        real t10 = 1.0 / (psi + 1);
        real t11 = 3*t6;
        real t12 = t10*(-psi - t11 - 4);
        real t13 = t5 + 2;
        real t14 = 1.0 / (t13);
        real t15 = 3*psi;
        real t16 = t15 - 2*t6;
        real t17 = psi + 2;
        real t18 = 1.0 / (t17);
        real t19 = -t6;
        real t20 = t15 + 12;
        real t21 = t18*(t11 + t20);
        real t22 = 1.0 / (psi + 3);
        real t23 = t22*(-t20 - t6);
        real t24 = 1.0 / (t8 + 18);
        real t25 = 1 - psi;
        real t26 = 12*t6;

        c[0] = alpha[1]*t9 + t7*(t5 - 2);
        c[1] = psi*(alpha[1] - 1.0 / 3);
        c[2] = alpha[1]*t12 + t14*(-t16 - 1);
        c[3] = alpha[1]*t21 + t18*(t13 + t19);
        c[4] = alpha[1]*t23 + t24*(-t16 - 3);
        c[5] = alpha[1]*psi + alpha[1]*t25;
        c[6] = 0;
        c[7] = alpha[2]*t9 + alpha[3]*t7*(24 - 24*psi) + t25*t7;
        c[8] = psi*(alpha[2] + 4*alpha[3] + 1.0 / 6);
        c[9] = alpha[2]*t12 + alpha[3]*t10*(-t15 - t26 - 15) + t14*(-t17 - t6);
        c[10] = alpha[2]*t21 + alpha[3]*t18*(8*psi + t26 + 40) + (t6 + 2)/(t5 + 4);
        c[11] = alpha[2]*t23 + alpha[3]*t22*(-4*t6 - t8 - 30) + t24*(t15 + t19 + 6);
        c[12] = alpha[2]*psi + alpha[2]*t25;
        c[13] = alpha[3]*psi + alpha[3]*t25;
        c[14] = 0;
        c[15] = 0;
        c[16] = 1.0 / 12;
        c[17] = -2.0 / 3;
        c[18] = 0;
        c[19] = 2.0 / 3;
        c[20] = -1.0 / 12;

        for (auto&& v : c) v /= h;
        if (right) {
            for (auto&& v : c) v *= -1;
            std::ranges::reverse(c);
        }

        return c;
    }

    std::span<const real>
    nbs_neumann(real, real, std::span<real> c, std::span<real>, bool) const
    {
        return c;
    }
};

stencil make_E4_1(std::span<const real> alpha) { return E4_1{alpha}; }

} // namespace ccs::stencils

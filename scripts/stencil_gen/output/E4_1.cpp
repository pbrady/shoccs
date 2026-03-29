#include "stencil.hpp"

#include <algorithm>

#include <cmath>

namespace ccs::stencils
{
struct E4_1 {

    static constexpr int P = 2;
    static constexpr int R = 5;
    static constexpr int T = 7;
    static constexpr int X = 0;

    std::array<real, 5> alpha;

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
        real t9 = t8 - 6;
        real t10 = -t7*t9;
        real t11 = 1.0 / (psi + 1);
        real t12 = 3*t6;
        real t13 = psi + t12 + 4;
        real t14 = -t11*t13;
        real t15 = 2*psi;
        real t16 = t15 + 2;
        real t17 = 1.0 / (t16);
        real t18 = 11*t6 + 6;
        real t19 = psi + 2;
        real t20 = 1.0 / (t19);
        real t21 = 3*psi;
        real t22 = t21 + 12;
        real t23 = t12 + t22;
        real t24 = t20*t23;
        real t25 = 1.0 / (t15 + 4);
        real t26 = 8*psi;
        real t27 = psi + 3;
        real t28 = 1.0 / (t27);
        real t29 = t22 + t6;
        real t30 = -t28*t29;
        real t31 = 1.0 / (t8 + 18);
        real t32 = 9*psi;
        real t33 = alpha[0]*psi;
        real t34 = 1 - psi;
        real t35 = t7*(t15 - 2);
        real t36 = t21 - 2*t6;
        real t37 = -t6;
        real t38 = 24*psi - 24;
        real t39 = alpha[4]*t7;
        real t40 = alpha[3] + 4*alpha[4];
        real t41 = 2*t6;
        real t42 = 12*t6;
        real t43 = t21 + t42 + 15;
        real t44 = alpha[4]*t11;
        real t45 = t26 + t42 + 40;
        real t46 = alpha[4]*t20;
        real t47 = 4*t6 + t8 + 30;
        real t48 = alpha[4]*t28;
        real t49 = alpha[4]*psi + alpha[4]*t34;
        real t50 = t7*t9;
        real t51 = alpha[1] + alpha[2];
        real t52 = 4*t33;
        real t53 = t11*t13;
        real t54 = -149*t6;
        real t55 = -t20*t23;
        real t56 = t28*t29;
        real t57 = alpha[3] + t51;

        c[0] = alpha[0]*t10 + t7*(t5 - 11);
        c[1] = psi*(alpha[0] - 11.0 / 6);
        c[2] = alpha[0]*t14 + t17*(-5*psi + t18);
        c[3] = alpha[0]*t24 + t25*(-t18 + t26);
        c[4] = alpha[0]*t30 + t31*(t18 - t32);
        c[5] = alpha[0]*t34 + t33;
        c[6] = 0;
        c[7] = alpha[1]*t10 + t35;
        c[8] = psi*(alpha[1] - 1.0 / 3);
        c[9] = alpha[1]*t14 + t17*(-t36 - 1);
        c[10] = alpha[1]*t24 + t20*(t16 + t37);
        c[11] = alpha[1]*t30 + t31*(-t36 - 3);
        c[12] = alpha[1]*psi + alpha[1]*t34;
        c[13] = 0;
        c[14] = alpha[2]*t10 + t34*t7;
        c[15] = psi*(alpha[2] + 1.0 / 6);
        c[16] = alpha[2]*t14 + t17*(-t19 - t6);
        c[17] = alpha[2]*t24 + t25*(t6 + 2);
        c[18] = alpha[2]*t30 + t31*(t21 + t37 + 6);
        c[19] = alpha[2]*psi + alpha[2]*t34;
        c[20] = 0;
        c[21] = alpha[3]*t10 + t35 - t38*t39;
        c[22] = psi*(t40 - 1.0 / 3);
        c[23] = alpha[3]*t14 + t17*(t27 + t41) - t43*t44;
        c[24] = alpha[3]*t24 + t20*(-t15 - t6 - 6) + t45*t46;
        c[25] = alpha[3]*t30 + t31*(t32 + t41 + 33) - t47*t48;
        c[26] = alpha[3]*psi + alpha[3]*t34;
        c[27] = t49;
        c[28] = alpha[1]*t50 + alpha[2]*t50 + alpha[3]*t50 + t38*t39 + t7*(-149*psi - 11);
        c[29] = psi*(-alpha[0] - t40 - t51 + 149.0 / 6);
        c[30] = alpha[1]*t53 + alpha[2]*t53 + alpha[3]*t53 + t17*(t21 + t54 + 14) + t43*t44 + t52;
        c[31] = -alpha[0]*t8 + alpha[1]*t55 + alpha[2]*t55 + alpha[3]*t55 + t25*(-t26 - t54 - 38) - t45*t46;
        c[32] = alpha[1]*t56 + alpha[2]*t56 + alpha[3]*t56 + t31*(15*psi + t54 + 78) + t47*t48 + t52;
        c[33] = psi*(-alpha[0] - t57) - t34*t57;
        c[34] = -t49;

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
        real t8 = t7*(t5 - 2);
        real t9 = 6*psi;
        real t10 = t9 - 6;
        real t11 = -t10*t7;
        real t12 = 1.0 / (psi + 1);
        real t13 = 3*t6;
        real t14 = psi + t13 + 4;
        real t15 = -t12*t14;
        real t16 = t5 + 2;
        real t17 = 1.0 / (t16);
        real t18 = 3*psi;
        real t19 = t18 - 2*t6;
        real t20 = psi + 2;
        real t21 = 1.0 / (t20);
        real t22 = -t6;
        real t23 = t18 + 12;
        real t24 = t13 + t23;
        real t25 = t21*t24;
        real t26 = psi + 3;
        real t27 = 1.0 / (t26);
        real t28 = t23 + t6;
        real t29 = -t27*t28;
        real t30 = 1.0 / (t9 + 18);
        real t31 = 1 - psi;
        real t32 = 1.0 / (t5 + 4);
        real t33 = 24*psi - 24;
        real t34 = alpha[4]*t7;
        real t35 = alpha[3] + 4*alpha[4];
        real t36 = 2*t6;
        real t37 = 12*t6;
        real t38 = t18 + t37 + 15;
        real t39 = alpha[4]*t12;
        real t40 = 8*psi;
        real t41 = t37 + t40 + 40;
        real t42 = alpha[4]*t21;
        real t43 = 4*t6 + t9 + 30;
        real t44 = alpha[4]*t27;
        real t45 = alpha[4]*psi + alpha[4]*t31;
        real t46 = t10*t7;
        real t47 = alpha[1] + alpha[2];
        real t48 = 4*alpha[0]*psi;
        real t49 = t12*t14;
        real t50 = -149*t6;
        real t51 = -t21*t24;
        real t52 = t27*t28;
        real t53 = alpha[3] + t47;

        c[0] = alpha[1]*t11 + t8;
        c[1] = psi*(alpha[1] - 1.0 / 3);
        c[2] = alpha[1]*t15 + t17*(-t19 - 1);
        c[3] = alpha[1]*t25 + t21*(t16 + t22);
        c[4] = alpha[1]*t29 + t30*(-t19 - 3);
        c[5] = alpha[1]*psi + alpha[1]*t31;
        c[6] = 0;
        c[7] = alpha[2]*t11 + t31*t7;
        c[8] = psi*(alpha[2] + 1.0 / 6);
        c[9] = alpha[2]*t15 + t17*(-t20 - t6);
        c[10] = alpha[2]*t25 + t32*(t6 + 2);
        c[11] = alpha[2]*t29 + t30*(t18 + t22 + 6);
        c[12] = alpha[2]*psi + alpha[2]*t31;
        c[13] = 0;
        c[14] = alpha[3]*t11 - t33*t34 + t8;
        c[15] = psi*(t35 - 1.0 / 3);
        c[16] = alpha[3]*t15 + t17*(t26 + t36) - t38*t39;
        c[17] = alpha[3]*t25 + t21*(-t5 - t6 - 6) + t41*t42;
        c[18] = alpha[3]*t29 + t30*(9*psi + t36 + 33) - t43*t44;
        c[19] = alpha[3]*psi + alpha[3]*t31;
        c[20] = t45;
        c[21] = alpha[1]*t46 + alpha[2]*t46 + alpha[3]*t46 + t33*t34 + t7*(-149*psi - 11);
        c[22] = psi*(-alpha[0] - t35 - t47 + 149.0 / 6);
        c[23] = alpha[1]*t49 + alpha[2]*t49 + alpha[3]*t49 + t17*(t18 + t50 + 14) + t38*t39 + t48;
        c[24] = -alpha[0]*t9 + alpha[1]*t51 + alpha[2]*t51 + alpha[3]*t51 + t32*(-t40 - t50 - 38) - t41*t42;
        c[25] = alpha[1]*t52 + alpha[2]*t52 + alpha[3]*t52 + t30*(15*psi + t50 + 78) + t43*t44 + t48;
        c[26] = psi*(-alpha[0] - t53) - t31*t53;
        c[27] = -t45;

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

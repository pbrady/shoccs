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
        real t39 = alpha[3]*t7;
        real t40 = (33.0 / 2)*alpha[0]*alpha[3] - 22*alpha[1]*alpha[3] + (95.0 / 2)*alpha[2]*alpha[3];
        real t41 = -7*alpha[3] + t40;
        real t42 = alpha[0] + t41;
        real t43 = -3*alpha[1] + 3*alpha[2];
        real t44 = t42 + t43;
        real t45 = -3*alpha[3] + t40;
        real t46 = 2*t6;
        real t47 = 12*t6;
        real t48 = t21 + t47 + 15;
        real t49 = alpha[3]*t11;
        real t50 = t26 + t47 + 40;
        real t51 = alpha[3]*t20;
        real t52 = 4*t6 + t8 + 30;
        real t53 = alpha[3]*t28;
        real t54 = alpha[3]*psi + alpha[3]*t34;
        real t55 = t7*t9;
        real t56 = -2*alpha[1] + 4*alpha[2];
        real t57 = 2*alpha[0] + t56;
        real t58 = 4*t33;
        real t59 = t11*t13;
        real t60 = -149*t6;
        real t61 = -t20*t23;
        real t62 = t28*t29;

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
        c[21] = t10*t44 + t35 - t38*t39;
        c[22] = psi*(alpha[0] + t43 + t45 - 1.0 / 3);
        c[23] = t14*t44 + t17*(t27 + t46) - t48*t49;
        c[24] = t20*(-t15 - t6 - 6) + t24*t44 + t50*t51;
        c[25] = t30*t44 + t31*(t32 + t46 + 33) - t52*t53;
        c[26] = psi*t44 + t34*t44;
        c[27] = t54;
        c[28] = alpha[1]*t55 + alpha[2]*t55 + t38*t39 + t44*t55 + t7*(-149*psi - 11);
        c[29] = psi*(-t45 - t57 + 149.0 / 6);
        c[30] = alpha[1]*t59 + alpha[2]*t59 + t17*(t21 + t60 + 14) + t44*t59 + t48*t49 + t58;
        c[31] = -alpha[0]*t8 + alpha[1]*t61 + alpha[2]*t61 + t25*(-t26 - t60 - 38) + t44*t61 - t50*t51;
        c[32] = alpha[1]*t62 + alpha[2]*t62 + t31*(15*psi + t60 + 78) + t44*t62 + t52*t53 + t58;
        c[33] = psi*(-t41 - t57) + t34*(-t42 - t56);
        c[34] = -t54;

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
        real t34 = alpha[3]*t7;
        real t35 = (33.0 / 2)*alpha[0]*alpha[3] - 22*alpha[1]*alpha[3] + (95.0 / 2)*alpha[2]*alpha[3];
        real t36 = -7*alpha[3] + t35;
        real t37 = alpha[0] + t36;
        real t38 = -3*alpha[1] + 3*alpha[2];
        real t39 = t37 + t38;
        real t40 = -3*alpha[3] + t35;
        real t41 = 2*t6;
        real t42 = 12*t6;
        real t43 = t18 + t42 + 15;
        real t44 = alpha[3]*t12;
        real t45 = 8*psi;
        real t46 = t42 + t45 + 40;
        real t47 = alpha[3]*t21;
        real t48 = 4*t6 + t9 + 30;
        real t49 = alpha[3]*t27;
        real t50 = alpha[3]*psi + alpha[3]*t31;
        real t51 = t10*t7;
        real t52 = -2*alpha[1] + 4*alpha[2];
        real t53 = 2*alpha[0] + t52;
        real t54 = 4*alpha[0]*psi;
        real t55 = t12*t14;
        real t56 = -149*t6;
        real t57 = -t21*t24;
        real t58 = t27*t28;

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
        c[14] = t11*t39 - t33*t34 + t8;
        c[15] = psi*(alpha[0] + t38 + t40 - 1.0 / 3);
        c[16] = t15*t39 + t17*(t26 + t41) - t43*t44;
        c[17] = t21*(-t5 - t6 - 6) + t25*t39 + t46*t47;
        c[18] = t29*t39 + t30*(9*psi + t41 + 33) - t48*t49;
        c[19] = psi*t39 + t31*t39;
        c[20] = t50;
        c[21] = alpha[1]*t51 + alpha[2]*t51 + t33*t34 + t39*t51 + t7*(-149*psi - 11);
        c[22] = psi*(-t40 - t53 + 149.0 / 6);
        c[23] = alpha[1]*t55 + alpha[2]*t55 + t17*(t18 + t56 + 14) + t39*t55 + t43*t44 + t54;
        c[24] = -alpha[0]*t9 + alpha[1]*t57 + alpha[2]*t57 + t32*(-t45 - t56 - 38) + t39*t57 - t46*t47;
        c[25] = alpha[1]*t58 + alpha[2]*t58 + t30*(15*psi + t56 + 78) + t39*t58 + t48*t49 + t54;
        c[26] = psi*(-t36 - t53) + t31*(-t37 - t52);
        c[27] = -t50;

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

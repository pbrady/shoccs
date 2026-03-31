#include "stencil.hpp"

#include <algorithm>
#include <stdexcept>

#include <cmath>

namespace ccs::stencils
{
struct E4_1 {

    static constexpr int P = 2;
    static constexpr int R = 5;
    static constexpr int T = 7;
    static constexpr int X = 0;

    // Singularity constraints for the conservative E4_1 stencil:
    //   - psi must be in the open interval (0, 1). Coefficients have poles at
    //     psi=0 (nbs_floating, nbs_dirichlet divide by psi) and psi=1
    //     (nbs_floating, nbs_dirichlet divide by (psi - 1)).
    //   - alpha[1] must be nonzero (nbs_floating and nbs_dirichlet divide by
    //     alpha[1]).
    //   - The denominator 288*alpha[1] + 648*psi + 12*psi^3 +
    //     90*psi^2 - 197 must be nonzero for the chosen alpha[1] and psi.
    //
    // alpha[0]: boundary shape parameter (free)
    // alpha[1]: quadrature weight parameter (must be nonzero, see above)
    std::array<real, 2> alpha;

    E4_1() = default;
    E4_1(std::span<const real> a)
    {
        copy_zero_padded(a, alpha);
        if (alpha[1] == 0.0)
            throw std::invalid_argument(
                "E4_1: alpha[1] must be nonzero (used as denominator)");
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
        real t6 = psi * psi * psi;
        real t7 = psi * psi;
        real t8 = 1.0 / (t5 + t6 + 6*t7 + 6);
        real t9 = 6*psi;
        real t10 = std::pow(psi, 4);
        real t11 = alpha[1]*psi;
        real t12 = 11*t7;
        real t13 = psi - 1;
        real t14 = 1/(alpha[1]*t13);
        real t15 = t14*(t10 + 320*t11 + t12 + 6*t6 - t9 + 12);
        real t16 = (1.0 / 12)*t15;
        real t17 = t16/psi;
        real t18 = t9 - 6;
        real t19 = -t18;
        real t20 = t19*t8;
        real t21 = 2*psi;
        real t22 = t21 + 2;
        real t23 = 1.0 / (t22);
        real t24 = t12 + 6;
        real t25 = 1.0 / (psi + 1);
        real t26 = 3*t7;
        real t27 = psi + t26 + 4;
        real t28 = -t27;
        real t29 = t25*t28;
        real t30 = 1.0 / (t21 + 4);
        real t31 = 8*psi;
        real t32 = psi + 2;
        real t33 = 1.0 / (t32);
        real t34 = 3*psi;
        real t35 = t34 + 12;
        real t36 = t26 + t35;
        real t37 = t33*t36;
        real t38 = 1.0 / (t9 + 18);
        real t39 = 9*psi;
        real t40 = psi + 3;
        real t41 = 1.0 / (t40);
        real t42 = t35 + t7;
        real t43 = -t42;
        real t44 = t41*t43;
        real t45 = -t13;
        real t46 = t8*(t21 - 2);
        real t47 = alpha[0]*alpha[1];
        real t48 = 33*t7;
        real t49 = 18*t6;
        real t50 = 3*t10;
        real t51 = alpha[0]*psi;
        real t52 = alpha[1] * alpha[1];
        real t53 = 72*t52;
        real t54 = 4*t14*(-alpha[0]*t53 + alpha[1]*t48 + alpha[1]*t49 + alpha[1]*t50 - 190*alpha[1]*t51 + 57*alpha[1] + 960*psi*t52 + 18*psi + 6*t10*t47 - 999*t11 + 30*t47*t6 + 126*t47*t7 + 28*t47 - t48 - t49 - t50 + t51*t53 - 36)/(288*alpha[1] + 648*psi + 12*t6 + 90*t7 - 197);
        real t55 = t34 - 2*t7;
        real t56 = -t7;
        real t57 = alpha[0]*t8;
        real t58 = alpha[0]*t25;
        real t59 = alpha[0]*t33;
        real t60 = alpha[0]*t41;
        real t61 = 2*t7;
        real t62 = alpha[0] + t54;
        real t63 = -t17 + t62;
        real t64 = -149*t7;
        real t65 = -1.0 / 3*t15;
        real t66 = -t36;

        c[0] = -t17*t20 + t8*(t5 - 11);
        c[1] = psi*(-t17 - 11.0 / 6);
        c[2] = -t17*t29 + t23*(-5*psi + t24);
        c[3] = -t17*t37 + t30*(-t24 + t31);
        c[4] = -t17*t44 + t38*(t24 - t39);
        c[5] = -t16 - t17*t45;
        c[6] = 0;
        c[7] = t20*t54 + t46;
        c[8] = psi*(t54 - 1.0 / 3);
        c[9] = t23*(-t55 - 1) + t29*t54;
        c[10] = t33*(t22 + t56) + t37*t54;
        c[11] = t38*(-t55 - 3) + t44*t54;
        c[12] = psi*t54 + t45*t54;
        c[13] = 0;
        c[14] = t19*t57 + t45*t8;
        c[15] = psi*(alpha[0] + 1.0 / 6);
        c[16] = t23*(-t32 - t7) + t28*t58;
        c[17] = t30*(t7 + 2) + t36*t59;
        c[18] = t38*(t34 + t56 + 6) + t43*t60;
        c[19] = alpha[0]*t45 + t51;
        c[20] = 0;
        c[21] = t46;
        c[22] = -1.0 / 3*psi;
        c[23] = t23*(t40 + t61);
        c[24] = t33*(-t21 - t7 - 6);
        c[25] = t38*(t39 + t61 + 33);
        c[26] = 0;
        c[27] = 0;
        c[28] = t18*t54*t8 + t18*t57 + t8*(-149*psi - 11);
        c[29] = psi*(149.0 / 6 - t63);
        c[30] = t23*(t34 + t64 + 14) + t25*t27*t54 + t27*t58 + t65;
        c[31] = (1.0 / 2)*t15 + t30*(-t31 - t64 - 38) + t33*t54*t66 + t59*t66;
        c[32] = t38*(15*psi + t64 + 78) + t41*t42*t54 + t42*t60 + t65;
        c[33] = -psi*t63 - t45*t62;
        c[34] = 0;

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
        real t6 = psi * psi * psi;
        real t7 = psi * psi;
        real t8 = 1.0 / (11*psi + t6 + 6*t7 + 6);
        real t9 = t8*(t5 - 2);
        real t10 = 6*psi;
        real t11 = t10 - 6;
        real t12 = -t11;
        real t13 = alpha[0]*alpha[1];
        real t14 = alpha[1]*psi;
        real t15 = 33*t7;
        real t16 = 18*t6;
        real t17 = std::pow(psi, 4);
        real t18 = 3*t17;
        real t19 = alpha[0]*psi;
        real t20 = alpha[1] * alpha[1];
        real t21 = 72*t20;
        real t22 = 1.0 / (alpha[1]);
        real t23 = psi - 1;
        real t24 = 1.0 / (t23);
        real t25 = t22*t24;
        real t26 = 4*t25*(-alpha[0]*t21 + alpha[1]*t15 + alpha[1]*t16 + alpha[1]*t18 - 190*alpha[1]*t19 + 57*alpha[1] + 960*psi*t20 + 18*psi + 6*t13*t17 + 30*t13*t6 + 126*t13*t7 + 28*t13 - 999*t14 - t15 - t16 - t18 + t19*t21 - 36)/(288*alpha[1] + 648*psi + 12*t6 + 90*t7 - 197);
        real t27 = t26*t8;
        real t28 = t5 + 2;
        real t29 = 1.0 / (t28);
        real t30 = 3*psi;
        real t31 = t30 - 2*t7;
        real t32 = 3*t7;
        real t33 = psi + t32 + 4;
        real t34 = -t33;
        real t35 = 1.0 / (psi + 1);
        real t36 = t26*t35;
        real t37 = psi + 2;
        real t38 = 1.0 / (t37);
        real t39 = -t7;
        real t40 = t30 + 12;
        real t41 = t32 + t40;
        real t42 = t26*t38;
        real t43 = 1.0 / (t10 + 18);
        real t44 = t40 + t7;
        real t45 = -t44;
        real t46 = psi + 3;
        real t47 = 1.0 / (t46);
        real t48 = t26*t47;
        real t49 = -t23;
        real t50 = alpha[0]*t8;
        real t51 = alpha[0]*t35;
        real t52 = 1.0 / (t5 + 4);
        real t53 = alpha[0]*t38;
        real t54 = alpha[0]*t47;
        real t55 = 2*t7;
        real t56 = -t10 + 320*t14 + t17 + 6*t6 + 11*t7 + 12;
        real t57 = alpha[0] + t26;
        real t58 = t57 - 1.0 / 12*t22*t24*t56/psi;
        real t59 = -149*t7;
        real t60 = t25*t56;
        real t61 = -1.0 / 3*t60;
        real t62 = -t41;

        c[0] = t12*t27 + t9;
        c[1] = psi*(t26 - 1.0 / 3);
        c[2] = t29*(-t31 - 1) + t34*t36;
        c[3] = t38*(t28 + t39) + t41*t42;
        c[4] = t43*(-t31 - 3) + t45*t48;
        c[5] = psi*t26 + t26*t49;
        c[6] = 0;
        c[7] = t12*t50 + t49*t8;
        c[8] = psi*(alpha[0] + 1.0 / 6);
        c[9] = t29*(-t37 - t7) + t34*t51;
        c[10] = t41*t53 + t52*(t7 + 2);
        c[11] = t43*(t30 + t39 + 6) + t45*t54;
        c[12] = alpha[0]*t49 + t19;
        c[13] = 0;
        c[14] = t9;
        c[15] = -1.0 / 3*psi;
        c[16] = t29*(t46 + t55);
        c[17] = t38*(-t5 - t7 - 6);
        c[18] = t43*(9*psi + t55 + 33);
        c[19] = 0;
        c[20] = 0;
        c[21] = t11*t27 + t11*t50 + t8*(-149*psi - 11);
        c[22] = psi*(149.0 / 6 - t58);
        c[23] = t29*(t30 + t59 + 14) + t33*t36 + t33*t51 + t61;
        c[24] = t42*t62 + t52*(-8*psi - t59 - 38) + t53*t62 + (1.0 / 2)*t60;
        c[25] = t43*(15*psi + t59 + 78) + t44*t48 + t44*t54 + t61;
        c[26] = -psi*t58 - t49*t57;
        c[27] = 0;

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

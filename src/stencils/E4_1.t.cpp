#include "stencil.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <sol/sol.hpp>
#include <spdlog/spdlog.h>

using Catch::Matchers::Approx;
using namespace ccs;

TEST_CASE("E4_1")
{
    using T = std::vector<real>;
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math);
    lua.script(R"(
        simulation = {
            scheme = {
                order = 1,
                type = "E4",
                alpha = {0.1, 0.7}
            }
        }
    )");

    auto st_opt = stencil::from_lua(lua["simulation"]);
    REQUIRE(!!st_opt);
    const auto& st = *st_opt;

    {
        auto [p, r, t, x] = st.query(bcs::Floating);
        REQUIRE(p == 2);
        REQUIRE(r == 5);
        REQUIRE(t == 7);
        REQUIRE(x == 0);

        T c(35);
        T ex{};

        st.nbs(2.0, bcs::Floating, 0.9, false, c, ex);
        REQUIRE_THAT(c,
                     Approx(T{4.076543037523072,
                              131.40125000000003,
                              -565.4246564327486,
                              867.1651005747128,
                              -584.1362927350428,
                              146.91805555555558,
                              0.0,
                              0.2335202168689599,
                              7.527173910445624,
                              -33.182271791559316,
                              50.90248623982128,
                              -34.01110180940503,
                              8.530193233828472,
                              0.0,
                              0.0037228349388058993,
                              0.12,
                              -0.6810526315789474,
                              0.5375862068965518,
                              -0.03025641025641035,
                              0.05,
                              0.0,
                              -0.0046535436735073744,
                              -0.15,
                              0.7263157894736841,
                              -1.48448275862069,
                              0.9128205128205128,
                              0.0,
                              0.0,
                              -3.6157157587741207,
                              -128.77342391044564,
                              548.323587581033,
                              -837.5320552053386,
                              562.4040505273538,
                              -140.8064432338285,
                              0.0})
                         .margin(1.0e-8));
    }

    {
        auto [p, r, t, x] = st.query(bcs::Floating);
        REQUIRE(p == 2);
        REQUIRE(r == 5);
        REQUIRE(t == 7);
        REQUIRE(x == 0);

        T c(35);
        T ex{};

        st.nbs(1.0, bcs::Floating, 0.3, false, c, ex);
        REQUIRE_THAT(c,
                     Approx(T{18.17653361131622,
                              12.810561224489796,
                              -154.4468328100471,
                              254.01462511091393,
                              -175.0900912183055,
                              44.535204081632656,
                              0.0,
                              0.9312484107568816,
                              0.6563305763527251,
                              -9.52418136905629,
                              15.52735317473245,
                              -10.111852713961513,
                              2.52110192117575,
                              0.0,
                              0.11350967872707003,
                              0.08,
                              -1.2707692307692307,
                              1.0269565217391305,
                              -0.04969696969696974,
                              0.09999999999999999,
                              0.0,
                              -0.14188709840883754,
                              -0.09999999999999999,
                              1.3384615384615384,
                              -2.908695652173913,
                              1.8121212121212122,
                              0.0,
                              0.0,
                              -6.760781196811406,
                              -6.696891800842523,
                              63.2295031900924,
                              -101.03941617384514,
                              67.24924912707222,
                              -15.981663145665546,
                              0.0})
                         .margin(1.0e-8));
    }

    {
        auto [p, r, t, x] = st.query(bcs::Dirichlet);
        REQUIRE(p == 2);
        REQUIRE(r == 4);
        REQUIRE(t == 7);
        REQUIRE(x == 0);

        T c(28);
        T ex{};

        st.nbs(0.5, bcs::Dirichlet, 0.7, false, c, ex);
        REQUIRE_THAT(c,
                     Approx(T{1.1565022757309227,
                              7.638119280064874,
                              -43.26935234565848,
                              68.92355406910586,
                              -46.02708891743107,
                              11.578265638187913,
                              0.0,
                              0.056527115350644794,
                              0.3733333333333333,
                              -2.6023529411764708,
                              2.0755555555555554,
                              -0.10306306306306301,
                              0.2,
                              0.0,
                              -0.07065889418830598,
                              -0.4666666666666666,
                              2.7529411764705882,
                              -5.844444444444444,
                              3.6288288288288286,
                              0.0,
                              0.0,
                              -14.826643004695182,
                              -110.21470658165214,
                              556.218054493184,
                              -877.4482630638148,
                              594.7864110967531,
                              -148.51485293977515,
                              0.0})
                         .margin(1.0e-8));
    }

    SECTION("Floating near psi=1 produces finite values")
    {
        auto [p, r, t, x] = st.query(bcs::Floating);
        T c(r * t);
        T ex{};

        st.nbs(1.0, bcs::Floating, 1.0 - 1e-12, false, c, ex);
        for (std::size_t i = 0; i < c.size(); ++i) {
            REQUIRE(std::isfinite(c[i]));
        }
    }

    SECTION("Dirichlet near psi=1 produces finite values")
    {
        auto [p, r, t, x] = st.query(bcs::Dirichlet);
        T c(r * t);
        T ex{};

        st.nbs(1.0, bcs::Dirichlet, 1.0 - 1e-12, false, c, ex);
        for (std::size_t i = 0; i < c.size(); ++i) {
            REQUIRE(std::isfinite(c[i]));
        }
    }

    SECTION("Floating near psi=snap_tol produces finite values")
    {
        auto [p, r, t, x] = st.query(bcs::Floating);
        T c(r * t);
        T ex{};

        st.nbs(1.0, bcs::Floating, 1e-12, false, c, ex);
        for (std::size_t i = 0; i < c.size(); ++i) {
            REQUIRE(std::isfinite(c[i]));
        }
    }

    SECTION("Dirichlet near psi=snap_tol produces finite values")
    {
        auto [p, r, t, x] = st.query(bcs::Dirichlet);
        T c(r * t);
        T ex{};

        st.nbs(1.0, bcs::Dirichlet, 1e-12, false, c, ex);
        for (std::size_t i = 0; i < c.size(); ++i) {
            REQUIRE(std::isfinite(c[i]));
        }
    }

    SECTION("Floating near psi=1: magnitude within safe bound")
    {
        // With psi_eps=1e-4 clamp in nbs_floating, coefficients remain O(1/psi_eps)
        // = O(1e4), well within numerical stability.
        auto [p, r, t, x] = st.query(bcs::Floating);
        T c(r * t);
        T ex{};

        st.nbs(1.0, bcs::Floating, 1.0 - 1e-12, false, c, ex);
        real max_abs = 0.0;
        for (std::size_t i = 0; i < c.size(); ++i) {
            max_abs = std::max(max_abs, std::abs(c[i]));
        }
        REQUIRE(max_abs < 1e8);
    }

    SECTION("Dirichlet near psi=1: magnitude within safe bound")
    {
        // With psi_eps=1e-4 clamp in nbs_dirichlet, coefficients remain bounded.
        auto [p, r, t, x] = st.query(bcs::Dirichlet);
        T c(r * t);
        T ex{};

        st.nbs(1.0, bcs::Dirichlet, 1.0 - 1e-12, false, c, ex);
        real max_abs = 0.0;
        for (std::size_t i = 0; i < c.size(); ++i) {
            max_abs = std::max(max_abs, std::abs(c[i]));
        }
        REQUIRE(max_abs < 1e8);
    }

    SECTION("Floating near psi=0: magnitude within safe bound")
    {
        // With psi_eps=1e-4 clamp in nbs_floating, coefficients remain bounded.
        auto [p, r, t, x] = st.query(bcs::Floating);
        T c(r * t);
        T ex{};

        st.nbs(1.0, bcs::Floating, 1e-12, false, c, ex);
        real max_abs = 0.0;
        for (std::size_t i = 0; i < c.size(); ++i) {
            max_abs = std::max(max_abs, std::abs(c[i]));
        }
        REQUIRE(max_abs < 1e8);
    }

    SECTION("Dirichlet near psi=0: magnitude within safe bound")
    {
        // Unlike Floating, Dirichlet's 1/psi term is canceled by psi factors
        // in the numerator, so coefficients remain O(1) near psi=0.
        auto [p, r, t, x] = st.query(bcs::Dirichlet);
        T c(r * t);
        T ex{};

        st.nbs(1.0, bcs::Dirichlet, 1e-12, false, c, ex);
        for (std::size_t i = 0; i < c.size(); ++i) {
            REQUIRE(std::abs(c[i]) < 1e8);
        }
    }

    SECTION("No interior polynomial denominator singularity with alpha[1] >= 197/288")
    {
        // The denominator D(psi) = 288*alpha[1] + 648*psi + 12*psi^3 + 90*psi^2 - 197
        // With alpha[1]=0.7 >= 197/288 ≈ 0.684, D(0) = 288*0.7 - 197 = 4.6 > 0.
        // Since D'(psi) = 36*psi^2 + 180*psi + 648 > 0, D is strictly increasing,
        // so D(psi) > 0 for all psi in (0, 1). No interior singularity exists.
        constexpr real alpha1 = 0.7;
        auto D = [](real psi) {
            return 288 * alpha1 + 648 * psi + 12 * psi * psi * psi +
                   90 * psi * psi - 197;
        };

        // Verify D(psi) > 0 at several sample points across (0, 1)
        for (real psi : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
            REQUIRE(D(psi) > 0.0);
        }

        // Verify D(0) > 0 — the critical condition
        REQUIRE(D(0.0) > 0.0);

        // Evaluate Floating stencil across the range — all coefficients bounded
        {
            auto [p, r, t, x] = st.query(bcs::Floating);
            T c(r * t);
            T ex{};

            for (real psi : {0.1, 0.3, 0.5, 0.7, 0.9}) {
                st.nbs(1.0, bcs::Floating, psi, false, c, ex);
                real max_abs = 0.0;
                for (std::size_t i = 0; i < c.size(); ++i) {
                    max_abs = std::max(max_abs, std::abs(c[i]));
                }
                REQUIRE(max_abs < 1e8);
            }
        }

        // Evaluate Dirichlet stencil across the range — all coefficients bounded
        {
            auto [p, r, t, x] = st.query(bcs::Dirichlet);
            T c(r * t);
            T ex{};

            for (real psi : {0.1, 0.3, 0.5, 0.7, 0.9}) {
                st.nbs(1.0, bcs::Dirichlet, psi, false, c, ex);
                real max_abs = 0.0;
                for (std::size_t i = 0; i < c.size(); ++i) {
                    max_abs = std::max(max_abs, std::abs(c[i]));
                }
                REQUIRE(max_abs < 1e8);
            }
        }
    }

    SECTION("alpha[1] < 197/288 throws")
    {
        // Single alpha (zero-padded to alpha[1]=0) should throw
        std::array<real, 1> short_alpha{0.1};
        REQUIRE_THROWS_AS(stencils::make_E4_1(short_alpha), std::invalid_argument);

        // Explicit alpha[1]=0 should throw
        std::array<real, 2> zero_alpha{0.1, 0.0};
        REQUIRE_THROWS_AS(stencils::make_E4_1(zero_alpha), std::invalid_argument);

        // alpha[1] just below the bound should throw
        std::array<real, 2> below_bound{0.1, 197.0 / 288.0 - 0.001};
        REQUIRE_THROWS_AS(stencils::make_E4_1(below_bound), std::invalid_argument);

        // alpha[1] at the bound should NOT throw
        std::array<real, 2> at_bound{0.1, 197.0 / 288.0};
        REQUIRE_NOTHROW(stencils::make_E4_1(at_bound));

        // alpha[1]=0.7 (test default, above bound) should NOT throw
        std::array<real, 2> above_bound{0.1, 0.7};
        REQUIRE_NOTHROW(stencils::make_E4_1(above_bound));
    }
}

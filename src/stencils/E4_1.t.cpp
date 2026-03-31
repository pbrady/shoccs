#include "stencil.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <cmath>
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
                alpha = {0.1, -0.05}
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
                     Approx(T{-1.6129957962988815,
                              -51.99250000000003,
                              220.70176900584806,
                              -336.48813218390825,
                              226.24463675213696,
                              -56.852777777777796,
                              0.0,
                              -0.48180038034232325,
                              -15.530112559764287,
                              65.65393278542234,
                              -100.42752036351044,
                              67.87451447348838,
                              -17.08901395529365,
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
                              -2.9003951615628374,
                              77.67761255976431,
                              -284.08761699594874,
                              414.16045139799337,
                              -273.05656575553974,
                              68.20651395529369,
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
                     Approx(T{-22.94213033343469,
                              -16.16928571428572,
                              185.13752747252752,
                              -299.12245341614926,
                              205.1606277056278,
                              -52.06428571428574,
                              0.0,
                              120.83978859124875,
                              85.16615671641794,
                              -999.8059902411026,
                              1628.5627303698905,
                              -1118.9832078245145,
                              284.2205223880598,
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
                              -126.66932137730328,
                              -62.226871002132214,
                              937.5919243070366,
                              -1540.1957117363502,
                              1060.201216482523,
                              -268.7012366737741,
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
                     Approx(T{-2.7314655167925315,
                              -18.039964005656262,
                              89.86827275761836,
                              -142.61494157040636,
                              98.62280881950765,
                              -25.104710484270853,
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
                              -10.938675212171729,
                              99.8455195612118,
                              -314.44814203866406,
                              440.38308971855446,
                              -287.39205806875685,
                              72.5502660398264,
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
}

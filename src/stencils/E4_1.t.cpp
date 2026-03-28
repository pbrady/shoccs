#include "stencil.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

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
                alpha = {0.1, -0.05, 0.02, 0.01}
            }
        }
    )");

    auto st_opt = stencil::from_lua(lua["simulation"]);
    REQUIRE(!!st_opt);
    const auto& st = *st_opt;

    {
        auto [p, r, t, x] = st.query(bcs::Floating);
        REQUIRE(p == 2);
        REQUIRE(r == 4);
        REQUIRE(t == 7);
        REQUIRE(x == 0);

        T c(28);
        T ex{};

        st.nbs(2.0, bcs::Floating, 1.0, false, c, ex);
        REQUIRE_THAT(c,
                     Approx(T{0.0,
                              -0.8666666666666666,
                              1.3,
                              -0.45,
                              -0.033333333333333354,
                              0.05,
                              0.0,
                              0.0,
                              -0.19166666666666665,
                              -0.15,
                              0.35,
                              0.016666666666666677,
                              -0.025,
                              0.0,
                              0.0,
                              0.11333333333333333,
                              -0.615,
                              0.41,
                              0.07666666666666665,
                              0.01,
                              0.005,
                              0.0,
                              0.0,
                              0.041666666666666664,
                              -0.3333333333333333,
                              0.0,
                              0.3333333333333333,
                              -0.041666666666666664})
                         .margin(1.0e-8));
    }

    {
        auto [p, r, t, x] = st.query(bcs::Floating);
        REQUIRE(p == 2);
        REQUIRE(r == 4);
        REQUIRE(t == 7);
        REQUIRE(x == 0);

        T c(28);
        T ex{};

        st.nbs(1.0, bcs::Floating, 0.3, false, c, ex);
        REQUIRE_THAT(c,
                     Approx(T{-0.7378129117259552,
                              -0.5199999999999999,
                              1.76,
                              -0.4252173913043479,
                              -0.176969696969697,
                              0.09999999999999999,
                              0.0,
                              -0.16317016317016317,
                              -0.11499999999999999,
                              -0.48576923076923073,
                              0.8049999999999999,
                              0.00893939393939397,
                              -0.049999999999999996,
                              0.0,
                              0.09648322691800953,
                              0.06799999999999999,
                              -1.1201538461538458,
                              0.757913043478261,
                              0.16775757575757577,
                              0.019999999999999997,
                              0.009999999999999998,
                              0.0,
                              0.0,
                              0.08333333333333333,
                              -0.6666666666666666,
                              0.0,
                              0.6666666666666666,
                              -0.08333333333333333})
                         .margin(1.0e-8));
    }

    {
        auto [p, r, t, x] = st.query(bcs::Dirichlet);
        REQUIRE(p == 2);
        REQUIRE(r == 3);
        REQUIRE(t == 7);
        REQUIRE(x == 0);

        T c(21);
        T ex{};

        st.nbs(0.5, bcs::Dirichlet, 0.7, false, c, ex);
        REQUIRE_THAT(c,
                     Approx(T{-0.08125772831655188,
                              -0.5366666666666666,
                              -0.8841176470588232,
                              1.5788888888888888,
                              0.023153153153153083,
                              -0.1,
                              0.0,
                              0.04804804804804807,
                              0.3173333333333333,
                              -2.292,
                              1.534222222222222,
                              0.33239639639639645,
                              0.04,
                              0.02,
                              0.0,
                              0.0,
                              0.16666666666666666,
                              -1.3333333333333333,
                              0.0,
                              1.3333333333333333,
                              -0.16666666666666666})
                         .margin(1.0e-8));
    }
}

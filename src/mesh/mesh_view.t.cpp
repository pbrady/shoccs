#include "mesh_view.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("location-2D")
{
    using namespace ccs;

    auto m = cartesian{int3{2, 3, 1}, real3{-1, 1}, real3{0, 2}};

    SECTION("X")
    {
        std::vector<real3> c{
            {-1, 1, 0}, {0, 1, 0}, {-1, 1.5, 0}, {0, 1.5, 0}, {-1, 2, 0}, {0, 2, 0}};
        auto g = mesh::location_view<0>(m);
        REQUIRE(c.size() == g.size());
        for (std::size_t i = 0; i < c.size(); ++i) REQUIRE(c[i] == g[i]);
    }

    SECTION("Y")
    {
        std::vector<real3> c{
            {-1, 1, 0}, {-1, 1.5, 0}, {-1, 2, 0}, {0, 1, 0}, {0, 1.5, 0}, {0, 2, 0}};
        auto g = mesh::location_view<1>(m);
        REQUIRE(c.size() == g.size());
        for (std::size_t i = 0; i < c.size(); ++i) REQUIRE(c[i] == g[i]);
    }
}

TEST_CASE("location-3D")
{
    using namespace ccs;

    auto m = cartesian{int3{2, 2, 3}, real3{-1, 1, 3}, real3{0, 2, 4}};

    SECTION("X")
    {
        std::vector<real3> c{{-1, 1, 3},
                             {0, 1, 3},
                             {-1, 1, 3.5},
                             {0, 1, 3.5},
                             {-1, 1, 4},
                             {0, 1, 4},
                             {-1, 2, 3},
                             {0, 2, 3},
                             {-1, 2, 3.5},
                             {0, 2, 3.5},
                             {-1, 2, 4},
                             {0, 2, 4}};
        auto g = mesh::location_view<0>(m);
        REQUIRE(c.size() == g.size());
        for (std::size_t i = 0; i < c.size(); ++i) REQUIRE(c[i] == g[i]);
    }

    SECTION("Y")
    {
        std::vector<real3> c{{-1, 1, 3},
                             {-1, 2, 3},
                             {-1, 1, 3.5},
                             {-1, 2, 3.5},
                             {-1, 1, 4},
                             {-1, 2, 4},
                             {0, 1, 3},
                             {0, 2, 3},
                             {0, 1, 3.5},
                             {0, 2, 3.5},
                             {0, 1, 4},
                             {0, 2, 4}};
        auto g = mesh::location_view<1>(m);
        REQUIRE(c.size() == g.size());
        for (std::size_t i = 0; i < c.size(); ++i) REQUIRE(c[i] == g[i]);
    }

    SECTION("Z")
    {
        std::vector<real3> c{{-1, 1, 3},
                             {-1, 1, 3.5},
                             {-1, 1, 4},
                             {-1, 2, 3},
                             {-1, 2, 3.5},
                             {-1, 2, 4},
                             {0, 1, 3},
                             {0, 1, 3.5},
                             {0, 1, 4},
                             {0, 2, 3},
                             {0, 2, 3.5},
                             {0, 2, 4}};
        auto r = mesh::location_view<2>(m);
        REQUIRE(c == r);
    }
}

TEST_CASE("plane")
{
    using namespace ccs;

    auto m = cartesian{int3{2, 2, 3}, real3{-1, 1, 3}, real3{0, 2, 4}};

    SECTION("X")
    {
        auto x = mesh::location_view<0>(m, 0);

        REQUIRE(x == std::vector<real3>{{-1, 1, 3},
                                        {-1, 1, 3.5},
                                        {-1, 1, 4},
                                        {-1, 2, 3},
                                        {-1, 2, 3.5},
                                        {-1, 2, 4}});
        x = mesh::location_view<0>(m, -1);
        REQUIRE(
            x ==
            std::vector<real3>{
                {0, 1, 3}, {0, 1, 3.5}, {0, 1, 4}, {0, 2, 3}, {0, 2, 3.5}, {0, 2, 4}});
    }

    SECTION("Y")
    {
        auto y = mesh::location_view<1>(m, 0);
        REQUIRE(
            y ==
            std::vector<real3>{
                {-1, 1, 3}, {-1, 1, 3.5}, {-1, 1, 4}, {0, 1, 3}, {0, 1, 3.5}, {0, 1, 4}});
        y = mesh::location_view<1>(m, -1);
        REQUIRE(
            y ==
            std::vector<real3>{
                {-1, 2, 3}, {-1, 2, 3.5}, {-1, 2, 4}, {0, 2, 3}, {0, 2, 3.5}, {0, 2, 4}});
    }

    SECTION("Z")
    {
        auto z = mesh::location_view<2>(m, 0);
        REQUIRE(z == std::vector<real3>{{-1, 1, 3}, {-1, 2, 3}, {0, 1, 3}, {0, 2, 3}});
        z = mesh::location_view<2>(m, -1);
        REQUIRE(z == std::vector<real3>{{-1, 1, 4}, {-1, 2, 4}, {0, 1, 4}, {0, 2, 4}});
    }
}

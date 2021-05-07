#include "fields/selector.hpp"
#include "mesh.hpp"
#include "random/random.hpp"
#include "selections.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "real3_operators.hpp"

#include <range/v3/algorithm/count.hpp>

using namespace ccs;

constexpr auto g = []() { return pick(); };

TEST_CASE("lines with no cut-cells")
{
    const auto db = domain_extents{.min = {-1, -1, 0}, .max = {1, 2, 2.2}};
    const auto extents = int3{21, 22, 23};

    auto m = mesh{index_extents{extents}, db};

    REQUIRE(extents[1] * extents[2] == (integer)m.lines(0).size());
    REQUIRE(extents[0] * extents[2] == (integer)m.lines(1).size());
    REQUIRE(extents[0] * extents[1] == (integer)m.lines(2).size());
}

TEST_CASE("lines")
{
    const auto db = domain_extents{.min = {-1, -1, 0}, .max = {1, 2, 2.2}};
    const auto extents = int3{21, 22, 23};

    auto m = mesh{index_extents{extents},
                  db,
                  std::vector<shape>{make_sphere(0, real3{0.01, -0.01, 0.5}, 0.25)}};

    // intersections in x from Mathematica
    SECTION("X")
    {
        constexpr integer n_intersections = 26;
        const auto& line = m.lines(0);

        REQUIRE(extents[1] * extents[2] + n_intersections / 2 == (integer)line.size());

        // check first two lines
        REQUIRE(line[0].stride == extents[1] * extents[2]);
        REQUIRE(line[0].start.mesh_coordinate == int3{0, 0, 0});
        REQUIRE(line[0].end.mesh_coordinate == int3{20, 0, 0});
        REQUIRE(line[1].stride == extents[1] * extents[2]);
        REQUIRE(line[1].start.mesh_coordinate == int3{0, 0, 1});
        REQUIRE(line[1].end.mesh_coordinate == int3{20, 0, 1});

        // check last line
        REQUIRE(line.back().stride == extents[1] * extents[2]);
        REQUIRE(line.back().start.mesh_coordinate == int3{0, 21, 22});
        REQUIRE(line.back().end.mesh_coordinate == int3{20, 21, 22});

        // check first/second intersection point at 10,6,3
        {
            const auto& l = line[6 * extents[2] + 3];
            REQUIRE(l.start.mesh_coordinate == int3{0, 6, 3});
            REQUIRE(!l.start.object);
            REQUIRE(l.end.mesh_coordinate == int3{10, 6, 3});
            REQUIRE(l.end.object);
            REQUIRE(l.end.object->object_coordinate == 0);
            REQUIRE(l.end.object->objectID == 0);
            REQUIRE(l.end.object->psi == Catch::Approx(0.40365385103120377));
        }

        {
            const auto& l = line[6 * extents[2] + 3 + 1];
            REQUIRE(l.start.mesh_coordinate == int3{10, 6, 3});
            REQUIRE(l.start.object);
            REQUIRE(l.start.object->object_coordinate == 1);
            REQUIRE(l.start.object->objectID == 0);
            REQUIRE(l.start.object->psi == Catch::Approx(0.2036538510312047));
            REQUIRE(l.end.mesh_coordinate == int3{20, 6, 3});
            REQUIRE(!l.end.object);
        }
    }

    SECTION("Y")
    {
        constexpr integer n_intersections = 42;
        const auto& line = m.lines(1);

        REQUIRE(extents[0] * extents[2] + n_intersections / 2 == (integer)line.size());

        // check first two lines
        REQUIRE(line[0].stride == extents[2]);
        REQUIRE(line[0].start.mesh_coordinate == int3{0, 0, 0});
        REQUIRE(line[0].end.mesh_coordinate == int3{0, 21, 0});
        REQUIRE(line[1].stride == extents[2]);
        REQUIRE(line[1].start.mesh_coordinate == int3{0, 0, 1});
        REQUIRE(line[1].end.mesh_coordinate == int3{0, 21, 1});

        // check last line
        REQUIRE(line.back().stride == extents[2]);
        REQUIRE(line.back().start.mesh_coordinate == int3{20, 0, 22});
        REQUIRE(line.back().end.mesh_coordinate == int3{20, 21, 22});

        // check first/second intersection point at 8,7,4
        {
            const auto& l = line[8 * extents[2] + 4];
            REQUIRE(l.start.mesh_coordinate == int3{8, 0, 4});
            REQUIRE(!l.start.object);
            REQUIRE(l.end.mesh_coordinate == int3{8, 7, 4});
            REQUIRE(l.end.object);
            REQUIRE(l.end.object->object_coordinate == 0);
            REQUIRE(l.end.object->objectID == 0);
            REQUIRE(l.end.object->psi == Catch::Approx(0.2884394027061823));
        }

        {
            const auto& l = line[8 * extents[2] + 4 + 1];
            REQUIRE(l.start.mesh_coordinate == int3{8, 7, 4});
            REQUIRE(l.start.object);
            REQUIRE(l.start.object->object_coordinate == 1);
            REQUIRE(l.start.object->objectID == 0);
            REQUIRE(l.start.object->psi == Catch::Approx(0.42843940270618086));
            REQUIRE(l.end.mesh_coordinate == int3{8, 21, 4});
            REQUIRE(!l.end.object);
        }
    }
    SECTION("Z")
    {
        constexpr integer n_intersections = 28;
        const auto& line = m.lines(2);

        REQUIRE(extents[0] * extents[1] + n_intersections / 2 == (integer)line.size());

        // check first two lines
        REQUIRE(line[0].stride == 1);
        REQUIRE(line[0].start.mesh_coordinate == int3{0, 0, 0});
        REQUIRE(line[0].end.mesh_coordinate == int3{0, 0, 22});
        REQUIRE(line[1].stride == 1);
        REQUIRE(line[1].start.mesh_coordinate == int3{0, 1, 0});
        REQUIRE(line[1].end.mesh_coordinate == int3{0, 1, 22});

        // check last line
        REQUIRE(line.back().stride == 1);
        REQUIRE(line.back().start.mesh_coordinate == int3{20, 21, 0});
        REQUIRE(line.back().end.mesh_coordinate == int3{20, 21, 22});

        // check last intersection points at 12,8,5
        {
            const auto& l = line[12 * extents[1] + 8 + (n_intersections / 2) - 1];
            REQUIRE(l.start.mesh_coordinate == int3{12, 8, 0});
            REQUIRE(!l.start.object);
            REQUIRE(l.end.mesh_coordinate == int3{12, 8, 5});
            REQUIRE(l.end.object);
            REQUIRE(l.end.object->object_coordinate == n_intersections - 2);
            REQUIRE(l.end.object->objectID == 0);
            REQUIRE(l.end.object->psi == Catch::Approx(0.4491194432954615));
        }

        {
            const auto& l = line[12 * extents[1] + 8 + (n_intersections / 2)];
            REQUIRE(l.start.mesh_coordinate == int3{12, 8, 5});
            REQUIRE(l.start.object);
            REQUIRE(l.start.object->object_coordinate == n_intersections - 1);
            REQUIRE(l.start.object->objectID == 0);
            REQUIRE(l.start.object->psi == Catch::Approx(0.4491194432954615));
            REQUIRE(l.end.mesh_coordinate == int3{12, 8, 22});
            REQUIRE(!l.end.object);
        }
    }
}

TEST_CASE("selections")
{
    using T = std::vector<real>;

    const auto db = domain_extents{.min = {-1, -1, 0}, .max = {1, 2, 2.2}};
    const auto extents = int3{21, 22, 23};

    auto m = mesh{index_extents{extents}, db};

    randomize();
    scalar<T> u{};
    u | sel::D = vs::generate_n(g, m.size());
    // test whole field comparison
    REQUIRE(rs::equal(u | sel::D, u | sel::D | m.F()));
    REQUIRE(rs::size(u | sel::D) == rs::size(u | sel::D | m.F()));

    auto view = u | sel::D | m.F();
    auto d = u | sel::D;

    // test next/prev going over a line
    {
        auto it = rs::begin(view);
        for (int i = 0; i < extents[2]; i++) REQUIRE(*it++ == d[i]);
        for (int i = 0; i < extents[2]; i++) REQUIRE(*it-- == d[extents[2] - i]);
    }

    // test advance a
    {
        auto it = rs::begin(view);
        rs::advance(it, 1);
        REQUIRE(*it == d[1]);
        rs::advance(it, -1);
        REQUIRE(*it == d[0]);
        rs::advance(it, extents[2]);
        REQUIRE(*it == d[extents[2]]);
        rs::advance(it, extents[2] + 1);
        REQUIRE(*it == d[2 * extents[2] + 1]);
        rs::advance(it, -2 * extents[2] - 1);
        REQUIRE(*it == d[0]);
    }
}

TEST_CASE("selections with object")
{
    using T = std::vector<int>;

    const auto db = domain_extents{.min = {-1, -1, 0}, .max = {1, 2, 2.2}};
    const auto extents = int3{21, 22, 23};

    auto m = mesh{index_extents{extents},
                  db,
                  std::vector<shape>{make_sphere(0, real3{0.01, -0.01, 0.5}, 0.25)}};

    scalar<T> u{};
    u | sel::D = vs::repeat_n(-1, m.size());
    u | sel::D | m.F() = 1;

    {
        using F = decltype(u | sel::D | m.F());

        REQUIRE(rs::bidirectional_range<F>);
        REQUIRE(!rs::contiguous_range<F>);
        REQUIRE(rs::random_access_range<F>);
        REQUIRE(rs::sized_range<F>);
    }

    auto nsolid = rs::count(u | sel::D, -1);
    REQUIRE(nsolid > 0);
    auto nfluid = rs::count(u | sel::D, 1);
    REQUIRE(nfluid > 0);

    REQUIRE(nfluid + nsolid == m.size());
    REQUIRE(nfluid == (integer)rs::size(u | sel::D | m.F()));

    scalar<T> v{u};

    v | sel::D =
        m.location() | vs::transform([center = real3{0.01, -0.01, 0.5}](auto&& loc) {
            return length(loc - center) > 0.25 ? 1 : -1;
        });

    REQUIRE(u == v);

    // test assignment
    scalar<T> w{};
    w | sel::D = vs::repeat_n(-1, m.size());

    w | sel::D | m.F() = u | sel::D | m.F();
    REQUIRE(w == u);
}
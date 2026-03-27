#include "field_registry.hpp"
#include "field.hpp"
#include "handle.hpp"
#include "scalar.hpp"
#include "selector_fwd.hpp"

#include <functional>

#include <Kokkos_Core.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace ccs;

// ---------------------------------------------------------------------------
// Custom main: Kokkos must be initialized before any test allocates Views.
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard kokkos(argc, argv);
    return Catch::Session().run(argc, argv);
}

// ---------------------------------------------------------------------------
// field_ref type properties
// ---------------------------------------------------------------------------

TEST_CASE("field_ref is trivially copyable")
{
    STATIC_REQUIRE(std::is_trivially_copyable_v<field_ref>);
}

TEST_CASE("field_ref size is 12 bytes")
{
    STATIC_REQUIRE(sizeof(field_ref) == 12);
}

// ---------------------------------------------------------------------------
// field_registry construction
// ---------------------------------------------------------------------------

TEST_CASE("field_registry construction and buffers_per_slot")
{
    // field_layout<2,1>::total_buffers = 2*4 + 1*12 = 20
    using reg_type = field_registry<4, 2, 1>;
    STATIC_REQUIRE(reg_type::buffers_per_slot == 20);

    // Verify it compiles and constructs.
    reg_type reg;
    (void)reg;
}

// ---------------------------------------------------------------------------
// allocate_scalar
// ---------------------------------------------------------------------------

TEST_CASE("allocate_scalar basic")
{
    field_registry<4, 2, 1> reg;
    constexpr auto layout = field_layout<2, 1>{};
    auto sh = scalar_handle{0 * layout.scalar_stride};

    auto ref = reg.allocate_scalar(0, 0, 100, 5, 3, 2);

    SECTION("returned field_ref is correct")
    {
        REQUIRE(ref.slot == 0);
        REQUIRE(ref.n_scalars == 1);
        REQUIRE(ref.n_vectors == 0);
    }

    SECTION("sizes match allocation parameters")
    {
        REQUIRE(reg.size(ref, sh.D()) == 100);
        REQUIRE(reg.size(ref, sh.Rx()) == 5);
        REQUIRE(reg.size(ref, sh.Ry()) == 3);
        REQUIRE(reg.size(ref, sh.Rz()) == 2);
    }

    SECTION("data pointers are non-null")
    {
        REQUIRE(reg.data(ref, sh.D()) != nullptr);
        REQUIRE(reg.data(ref, sh.Rx()) != nullptr);
        REQUIRE(reg.data(ref, sh.Ry()) != nullptr);
        REQUIRE(reg.data(ref, sh.Rz()) != nullptr);
    }
}

// ---------------------------------------------------------------------------
// allocate_vector
// ---------------------------------------------------------------------------

TEST_CASE("allocate_vector basic")
{
    field_registry<4, 2, 1> reg;
    constexpr auto layout = field_layout<2, 1>{};
    auto vh = vector_handle{layout.vector_base + 0 * layout.vector_stride};

    auto ref = reg.allocate_vector(0, 0, 100, 5, 3, 2);

    SECTION("returned field_ref is correct")
    {
        REQUIRE(ref.slot == 0);
        REQUIRE(ref.n_scalars == 0);
        REQUIRE(ref.n_vectors == 1);
    }

    SECTION("12 Views allocated with correct sizes")
    {
        // x component: D=100, Rx=5, Ry=3, Rz=2
        REQUIRE(reg.size(ref, vh.x().D()) == 100);
        REQUIRE(reg.size(ref, vh.x().Rx()) == 5);
        REQUIRE(reg.size(ref, vh.x().Ry()) == 3);
        REQUIRE(reg.size(ref, vh.x().Rz()) == 2);

        // y component: same sizes
        REQUIRE(reg.size(ref, vh.y().D()) == 100);
        REQUIRE(reg.size(ref, vh.y().Rx()) == 5);
        REQUIRE(reg.size(ref, vh.y().Ry()) == 3);
        REQUIRE(reg.size(ref, vh.y().Rz()) == 2);

        // z component: same sizes
        REQUIRE(reg.size(ref, vh.z().D()) == 100);
        REQUIRE(reg.size(ref, vh.z().Rx()) == 5);
        REQUIRE(reg.size(ref, vh.z().Ry()) == 3);
        REQUIRE(reg.size(ref, vh.z().Rz()) == 2);
    }

    SECTION("data pointers are non-null for all 12 buffers")
    {
        auto bufs = vh.all();
        for (auto bh : bufs) {
            REQUIRE(reg.data(ref, bh) != nullptr);
        }
    }
}

// ---------------------------------------------------------------------------
// view / data / size access
// ---------------------------------------------------------------------------

TEST_CASE("view returns writable Kokkos::View")
{
    field_registry<4, 2, 1> reg;
    auto sh = scalar_handle{0};

    auto ref = reg.allocate_scalar(0, 0, 100, 5, 3, 2);

    // Write through data pointer, read through view.
    reg.data(ref, sh.D())[0] = 42.0;
    REQUIRE(reg.view(ref, sh.D())(0) == 42.0);
}

// ---------------------------------------------------------------------------
// Unallocated slots
// ---------------------------------------------------------------------------

TEST_CASE("unallocated slots have size 0 and null data")
{
    field_registry<4, 2, 1> reg;
    field_ref ref{.slot = 0, .n_scalars = 0, .n_vectors = 0};
    auto sh = scalar_handle{0};

    REQUIRE(reg.size(ref, sh.D()) == 0);
    REQUIRE(reg.data(ref, sh.D()) == nullptr);
}

// ---------------------------------------------------------------------------
// Sequential scalar allocation
// ---------------------------------------------------------------------------

TEST_CASE("sequential scalar allocation in same slot")
{
    field_registry<4, 2, 1> reg;
    constexpr auto layout = field_layout<2, 1>{};
    auto sh0 = scalar_handle{0 * layout.scalar_stride};
    auto sh1 = scalar_handle{1 * layout.scalar_stride};

    auto ref1 = reg.allocate_scalar(0, 0, 100, 5, 3, 2);
    REQUIRE(ref1.n_scalars == 1);

    auto ref2 = reg.allocate_scalar(0, 1, 200, 10, 6, 4);
    REQUIRE(ref2.n_scalars == 2);

    SECTION("both scalars are accessible")
    {
        REQUIRE(reg.size(ref2, sh0.D()) == 100);
        REQUIRE(reg.size(ref2, sh1.D()) == 200);
    }

    SECTION("both scalars point to different memory")
    {
        REQUIRE(reg.data(ref2, sh0.D()) != reg.data(ref2, sh1.D()));
    }
}

// ---------------------------------------------------------------------------
// Mixed scalar + vector allocation
// ---------------------------------------------------------------------------

TEST_CASE("mixed scalar and vector allocation in same slot")
{
    field_registry<4, 2, 1> reg;
    constexpr auto layout = field_layout<2, 1>{};
    auto sh = scalar_handle{0 * layout.scalar_stride};
    auto vh = vector_handle{layout.vector_base + 0 * layout.vector_stride};

    reg.allocate_scalar(0, 0, 100, 5, 3, 2);
    auto ref = reg.allocate_vector(0, 0, 200, 10, 6, 4);

    SECTION("field_ref reflects both allocations")
    {
        REQUIRE(ref.n_scalars == 1);
        REQUIRE(ref.n_vectors == 1);
    }

    SECTION("scalar D buffer (index 0) and vector x.D buffer (index 8) are both accessible")
    {
        REQUIRE(reg.size(ref, sh.D()) == 100);
        REQUIRE(reg.size(ref, vh.x().D()) == 200);
    }

    SECTION("scalar and vector buffers point to different memory")
    {
        REQUIRE(reg.data(ref, sh.D()) != reg.data(ref, vh.x().D()));
    }
}

// ---------------------------------------------------------------------------
// deep_copy_slot
// ---------------------------------------------------------------------------

TEST_CASE("deep_copy_slot copies data between slots")
{
    field_registry<4, 2, 1> reg;
    auto sh = scalar_handle{0};

    // Allocate scalar 0 in both slots with matching sizes.
    auto ref0 = reg.allocate_scalar(0, 0, 100, 5, 3, 2);
    auto ref1 = reg.allocate_scalar(1, 0, 100, 5, 3, 2);

    // Fill source (slot 0) D buffer with 42.0 and Rx buffer with 7.0.
    auto& src_d = reg.view(ref0, sh.D());
    for (int i = 0; i < 100; ++i) {
        src_d(i) = 42.0;
    }
    auto& src_rx = reg.view(ref0, sh.Rx());
    for (int i = 0; i < 5; ++i) {
        src_rx(i) = 7.0;
    }

    reg.deep_copy_slot(1, 0);

    SECTION("D buffer is copied")
    {
        auto& dst_d = reg.view(ref1, sh.D());
        for (int i = 0; i < 100; ++i) {
            REQUIRE(dst_d(i) == 42.0);
        }
    }

    SECTION("Rx buffer is copied")
    {
        auto& dst_rx = reg.view(ref1, sh.Rx());
        for (int i = 0; i < 5; ++i) {
            REQUIRE(dst_rx(i) == 7.0);
        }
    }
}

// ---------------------------------------------------------------------------
// swap_slots
// ---------------------------------------------------------------------------

TEST_CASE("swap_slots swaps extents and data")
{
    field_registry<4, 2, 1> reg;
    auto sh = scalar_handle{0};

    // Allocate scalar 0 in slot 0 with size 100 and in slot 1 with size 200.
    auto ref0 = reg.allocate_scalar(0, 0, 100, 5, 3, 2);
    auto ref1 = reg.allocate_scalar(1, 0, 200, 10, 6, 4);

    // Fill slot 0 D buffer with 1.0, slot 1 D buffer with 2.0.
    for (int i = 0; i < 100; ++i) {
        reg.view(ref0, sh.D())(i) = 1.0;
    }
    for (int i = 0; i < 200; ++i) {
        reg.view(ref1, sh.D())(i) = 2.0;
    }

    reg.swap_slots(0, 1);

    SECTION("extents are swapped")
    {
        // After swap, slot 0 should have extent 200, slot 1 should have extent 100.
        // We need updated refs since metadata was swapped.
        field_ref swapped_ref0{.slot = 0, .n_scalars = 1, .n_vectors = 0};
        field_ref swapped_ref1{.slot = 1, .n_scalars = 1, .n_vectors = 0};

        REQUIRE(reg.size(swapped_ref0, sh.D()) == 200);
        REQUIRE(reg.size(swapped_ref1, sh.D()) == 100);
    }

    SECTION("data values are swapped")
    {
        field_ref swapped_ref0{.slot = 0, .n_scalars = 1, .n_vectors = 0};
        field_ref swapped_ref1{.slot = 1, .n_scalars = 1, .n_vectors = 0};

        // Slot 0 should now contain 2.0 (was in slot 1).
        REQUIRE(reg.view(swapped_ref0, sh.D())(0) == 2.0);
        // Slot 1 should now contain 1.0 (was in slot 0).
        REQUIRE(reg.view(swapped_ref1, sh.D())(0) == 1.0);
    }

    SECTION("boundary buffer extents are swapped")
    {
        field_ref swapped_ref0{.slot = 0, .n_scalars = 1, .n_vectors = 0};
        field_ref swapped_ref1{.slot = 1, .n_scalars = 1, .n_vectors = 0};

        // Slot 0 was allocated with Rx=5, slot 1 with Rx=10.
        // After swap, slot 0 should have Rx extent 10, slot 1 should have 5.
        REQUIRE(reg.size(swapped_ref0, sh.Rx()) == 10);
        REQUIRE(reg.size(swapped_ref1, sh.Rx()) == 5);
    }
}

// ---------------------------------------------------------------------------
// Span bridge: extract_scalar_span
// ---------------------------------------------------------------------------

TEST_CASE("extract_scalar_span returns scalar_span with correct spans")
{
    field_registry<4, 2, 1> reg;
    constexpr auto layout = field_layout<2, 1>{};
    auto sh = scalar_handle{0 * layout.scalar_stride};

    auto ref = reg.allocate_scalar(0, 0, 100, 5, 3, 2);

    auto span = extract_scalar_span(reg, ref, sh);

    SECTION("D span matches registry data and size")
    {
        auto d = get<si::D>(span);
        REQUIRE(d.data() == reg.data(ref, sh.D()));
        REQUIRE(static_cast<int>(d.size()) == reg.size(ref, sh.D()));
    }

    SECTION("Rx span matches registry data and size")
    {
        auto rx = get<si::Rx>(span);
        REQUIRE(rx.data() == reg.data(ref, sh.Rx()));
        REQUIRE(static_cast<int>(rx.size()) == reg.size(ref, sh.Rx()));
    }

    SECTION("Ry span matches registry data and size")
    {
        auto ry = get<si::Ry>(span);
        REQUIRE(ry.data() == reg.data(ref, sh.Ry()));
        REQUIRE(static_cast<int>(ry.size()) == reg.size(ref, sh.Ry()));
    }

    SECTION("Rz span matches registry data and size")
    {
        auto rz = get<si::Rz>(span);
        REQUIRE(rz.data() == reg.data(ref, sh.Rz()));
        REQUIRE(static_cast<int>(rz.size()) == reg.size(ref, sh.Rz()));
    }
}

// ---------------------------------------------------------------------------
// Span bridge: extract_scalar_view (const)
// ---------------------------------------------------------------------------

TEST_CASE("extract_scalar_view returns const spans from const registry")
{
    field_registry<4, 2, 1> reg;
    constexpr auto layout = field_layout<2, 1>{};
    auto sh = scalar_handle{0 * layout.scalar_stride};

    auto ref = reg.allocate_scalar(0, 0, 100, 5, 3, 2);

    const auto& creg = reg;
    auto sv = extract_scalar_view(creg, ref, sh);

    SECTION("D span type is std::span<const real>")
    {
        auto d = get<si::D>(sv);
        STATIC_REQUIRE(std::is_same_v<decltype(d), std::span<const real>>);
    }

    SECTION("D span matches registry data and size")
    {
        auto d = get<si::D>(sv);
        REQUIRE(d.data() == creg.data(ref, sh.D()));
        REQUIRE(static_cast<int>(d.size()) == creg.size(ref, sh.D()));
    }
}

// ---------------------------------------------------------------------------
// Span bridge: write-through
// ---------------------------------------------------------------------------

TEST_CASE("writing through scalar_span modifies registry storage")
{
    field_registry<4, 2, 1> reg;
    auto sh = scalar_handle{0};

    auto ref = reg.allocate_scalar(0, 0, 100, 5, 3, 2);

    auto span = extract_scalar_span(reg, ref, sh);
    get<si::D>(span)[0] = 42.0;

    REQUIRE(reg.view(ref, sh.D())(0) == 42.0);
}

// ---------------------------------------------------------------------------
// Span bridge: different slots yield different data
// ---------------------------------------------------------------------------

TEST_CASE("extract_scalar_span from different slots points to different data")
{
    field_registry<4, 2, 1> reg;
    auto sh = scalar_handle{0};

    auto ref0 = reg.allocate_scalar(0, 0, 100, 5, 3, 2);
    auto ref1 = reg.allocate_scalar(1, 0, 100, 5, 3, 2);

    auto span0 = extract_scalar_span(reg, ref0, sh);
    auto span1 = extract_scalar_span(reg, ref1, sh);

    REQUIRE(get<si::D>(span0).data() != get<si::D>(span1).data());
}

// ---------------------------------------------------------------------------
// Integration test: registry-backed spans work with existing tuple operators
// ---------------------------------------------------------------------------

TEST_CASE("integration: registry span bridge with tuple_math operator+=")
{
    // 1. Create registry with 2 slots (input + output), 1 scalar, 0 vectors.
    field_registry<2, 1, 0> reg;
    constexpr auto layout = field_layout<1, 0>{};
    auto sh = scalar_handle{0 * layout.scalar_stride};

    // 2. Allocate scalar in both slots: D=512 (8x8x8), Rx=5, Ry=3, Rz=2.
    auto in_ref  = reg.allocate_scalar(0, 0, 512, 5, 3, 2);
    auto out_ref = reg.allocate_scalar(1, 0, 512, 5, 3, 2);

    // 3. Fill input slot's D buffer with known values (i * 0.5).
    auto& in_d = reg.view(in_ref, sh.D());
    for (int i = 0; i < 512; ++i) {
        in_d(i) = i * 0.5;
    }

    // Also fill output slot's D buffer with 1.0 so we can verify +=.
    auto& out_d = reg.view(out_ref, sh.D());
    for (int i = 0; i < 512; ++i) {
        out_d(i) = 1.0;
    }

    // 4. Extract scalar_view from input slot and scalar_span from output slot.
    const auto& creg = reg;
    auto input  = extract_scalar_view(creg, in_ref, sh);
    auto output = extract_scalar_span(reg, out_ref, sh);

    // 5. Perform element-wise += through the span bridge.
    //    Use manual loop over get<si::D>() — the tuple_math operator+= between
    //    scalar_span and scalar_view may not compile due to the OutputTuple concept
    //    requiring matching non-const range types.
    auto d_in  = get<si::D>(input);   // std::span<const real>
    auto d_out = get<si::D>(output);  // std::span<real>
    for (int i = 0; i < static_cast<int>(d_in.size()); ++i) {
        d_out[i] += d_in[i];
    }

    // Also exercise boundary buffers to prove the full span bridge works.
    auto rx_in  = get<si::Rx>(input);
    auto rx_out = get<si::Rx>(output);
    for (int i = 0; i < static_cast<int>(rx_in.size()); ++i) {
        rx_out[i] += rx_in[i];
    }

    // 6. Verify the result is written into the registry's output slot storage.
    SECTION("D buffer: output = 1.0 + i*0.5")
    {
        for (int i = 0; i < 512; ++i) {
            REQUIRE(reg.view(out_ref, sh.D())(i) == 1.0 + i * 0.5);
        }
    }

    SECTION("Rx buffer is accessible through span bridge")
    {
        // Rx was default-initialized to 0.0 in output, 0.0 in input (both unwritten).
        // After +=, still 0.0 + 0.0 = 0.0. Verify the span bridge didn't corrupt anything.
        for (int i = 0; i < 5; ++i) {
            REQUIRE(reg.view(out_ref, sh.Rx())(i) == 0.0);
        }
    }

    SECTION("input slot is unmodified (read-only view)")
    {
        for (int i = 0; i < 512; ++i) {
            REQUIRE(reg.view(in_ref, sh.D())(i) == i * 0.5);
        }
    }
}

// ---------------------------------------------------------------------------
// field_ref SBO fitness
// ---------------------------------------------------------------------------

TEST_CASE("field_ref fits in std::function SBO")
{
    SECTION("field_ref is small and trivially copyable")
    {
        STATIC_REQUIRE(sizeof(field_ref) == 12);
        STATIC_REQUIRE(std::is_trivially_copyable_v<field_ref>);
        REQUIRE(sizeof(field_ref) <= 24);
    }

    SECTION("field_span is too large for SBO")
    {
        // field_span contains std::vectors; it's at least 48 bytes — well beyond
        // the typical 16–32 byte SBO threshold.
        REQUIRE(sizeof(field_span) >= 48);
    }

    SECTION("std::function capturing field_ref by value works")
    {
        field_ref ref{.slot = 2, .n_scalars = 1, .n_vectors = 3};
        std::function<int(int)> fn = [ref](int x) { return ref.slot + x; };
        REQUIRE(fn(10) == 12);
    }
}

TEST_CASE("integration: tuple_math operator+= with registry-backed spans")
{
    // Same setup as above, but exercise the tuple_math operator+= on the full
    // scalar_span/scalar_view types rather than individual span elements.
    field_registry<2, 1, 0> reg;
    auto sh = scalar_handle{0};

    auto in_ref  = reg.allocate_scalar(0, 0, 512, 5, 3, 2);
    auto out_ref = reg.allocate_scalar(1, 0, 512, 5, 3, 2);

    // Fill input D with i * 0.5, Rx with 3.0.
    for (int i = 0; i < 512; ++i) reg.view(in_ref, sh.D())(i) = i * 0.5;
    for (int i = 0; i < 5; ++i) reg.view(in_ref, sh.Rx())(i) = 3.0;

    // Fill output D with 1.0, Rx with 10.0.
    for (int i = 0; i < 512; ++i) reg.view(out_ref, sh.D())(i) = 1.0;
    for (int i = 0; i < 5; ++i) reg.view(out_ref, sh.Rx())(i) = 10.0;

    auto input  = extract_scalar_view(std::as_const(reg), in_ref, sh);
    auto output = extract_scalar_span(reg, out_ref, sh);

    // Use tuple_math operator+= on the full nested scalar tuple.
    output += input;

    SECTION("D buffer after tuple_math +=")
    {
        for (int i = 0; i < 512; ++i) {
            REQUIRE(reg.view(out_ref, sh.D())(i) == 1.0 + i * 0.5);
        }
    }

    SECTION("Rx buffer after tuple_math +=")
    {
        for (int i = 0; i < 5; ++i) {
            REQUIRE(reg.view(out_ref, sh.Rx())(i) == 13.0);
        }
    }

    SECTION("Ry and Rz buffers unchanged (both zero)")
    {
        for (int i = 0; i < 3; ++i) {
            REQUIRE(reg.view(out_ref, sh.Ry())(i) == 0.0);
        }
        for (int i = 0; i < 2; ++i) {
            REQUIRE(reg.view(out_ref, sh.Rz())(i) == 0.0);
        }
    }
}

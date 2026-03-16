#pragma once

//
// Compile-time handle arithmetic for flat field storage.
//
// Layout (max-capacity):
//   Index 0..3:                         scalar[0] buffers (D, Rx, Ry, Rz)
//   Index 4..7:                         scalar[1] buffers (D, Rx, Ry, Rz)
//   ...
//   Index 4*MaxS .. 4*MaxS+11:          vector[0] buffers (x.D, x.Rx, x.Ry, x.Rz,
//                                                          y.D, y.Rx, y.Ry, y.Rz,
//                                                          z.D, z.Rx, z.Ry, z.Rz)
//   ...
//   Total capacity: 4*MaxS + 12*MaxV buffers
//
// Design decisions:
//   - MaxScalars/MaxVectors are template parameters of the layout, NOT globals.
//     Different systems (heat: {1,0}, scalar_wave: {1,1}) instantiate different
//     layouts. The layout is a value type, usable as an NTTP.
//   - Handles are pure-value, trivially-copyable structs with defaulted operator==,
//     satisfying C++20 structural type requirements for use as NTTPs.
//   - Factory functions are consteval: index violations are compile-time errors.
//   - The handle stores only a buffer index, not a length. Lengths are queried
//     from the registry at kernel-launch time (see proposal section 10.7).
//

#include <array>
#include <type_traits>

namespace ccs
{

// ---------------------------------------------------------------------------
// Layout descriptor: template parameters fix the maximum capacity.
// ---------------------------------------------------------------------------

template <int MaxS, int MaxV>
struct field_layout {
    static_assert(MaxS >= 0 && MaxV >= 0);

    static constexpr int max_scalars      = MaxS;
    static constexpr int max_vectors      = MaxV;
    static constexpr int scalar_stride    = 4;   // D, Rx, Ry, Rz
    static constexpr int vector_stride    = 12;  // 3 components x 4 buffers
    static constexpr int vector_base      = MaxS * scalar_stride;
    static constexpr int total_buffers    = vector_base + MaxV * vector_stride;

    // Active counts (runtime state embedded in the layout value).
    // These are set once at construction and then frozen.
    int n_scalars = 0;
    int n_vectors = 0;

    constexpr bool operator==(const field_layout&) const = default;
};

// ---------------------------------------------------------------------------
// Buffer handle: a single buffer index into flat storage.
// Structural type => usable as NTTP.
// ---------------------------------------------------------------------------

struct buf_handle {
    int id = -1;

    constexpr bool operator==(const buf_handle&) const = default;
    constexpr explicit operator bool() const { return id >= 0; }
};

// ---------------------------------------------------------------------------
// Scalar handle: named access to the 4 buffers of one scalar field.
//
//   scalar[i] occupies indices [i*4 .. i*4+3]:
//     +0 = D,  +1 = Rx,  +2 = Ry,  +3 = Rz
// ---------------------------------------------------------------------------

struct scalar_handle {
    int base = -1;  // = scalar_index * 4

    constexpr buf_handle D()  const { return {base + 0}; }
    constexpr buf_handle Rx() const { return {base + 1}; }
    constexpr buf_handle Ry() const { return {base + 2}; }
    constexpr buf_handle Rz() const { return {base + 3}; }

    // All 4 buffer indices, for iteration.
    constexpr std::array<buf_handle, 4> all() const
    {
        return {D(), Rx(), Ry(), Rz()};
    }

    // R-component handles (Rx, Ry, Rz) as a group.
    constexpr std::array<buf_handle, 3> R() const
    {
        return {Rx(), Ry(), Rz()};
    }

    constexpr bool operator==(const scalar_handle&) const = default;
    constexpr explicit operator bool() const { return base >= 0; }
};

// ---------------------------------------------------------------------------
// Vector handle: named access to the 12 buffers of one vector field.
//
//   vector[i] occupies indices [vector_base + i*12 .. vector_base + i*12 + 11]:
//     x: +0..+3   (D, Rx, Ry, Rz)
//     y: +4..+7   (D, Rx, Ry, Rz)
//     z: +8..+11  (D, Rx, Ry, Rz)
// ---------------------------------------------------------------------------

struct vector_handle {
    int base = -1;  // = vector_base + vector_index * 12

    // Component access as scalar_handles.
    constexpr scalar_handle x() const { return {base + 0}; }
    constexpr scalar_handle y() const { return {base + 4}; }
    constexpr scalar_handle z() const { return {base + 8}; }

    // Direct buffer access with full names matching current codebase selectors.
    // Domain components.
    constexpr buf_handle Dx()  const { return x().D(); }
    constexpr buf_handle Dy()  const { return y().D(); }
    constexpr buf_handle Dz()  const { return z().D(); }

    // x-component boundary.
    constexpr buf_handle xRx() const { return x().Rx(); }
    constexpr buf_handle xRy() const { return x().Ry(); }
    constexpr buf_handle xRz() const { return x().Rz(); }

    // y-component boundary.
    constexpr buf_handle yRx() const { return y().Rx(); }
    constexpr buf_handle yRy() const { return y().Ry(); }
    constexpr buf_handle yRz() const { return y().Rz(); }

    // z-component boundary.
    constexpr buf_handle zRx() const { return z().Rx(); }
    constexpr buf_handle zRy() const { return z().Ry(); }
    constexpr buf_handle zRz() const { return z().Rz(); }

    // All 12 buffer indices.
    constexpr std::array<buf_handle, 12> all() const
    {
        auto xa = x().all();
        auto ya = y().all();
        auto za = z().all();
        return {xa[0], xa[1], xa[2], xa[3],
                ya[0], ya[1], ya[2], ya[3],
                za[0], za[1], za[2], za[3]};
    }

    // All 3 component scalar_handles.
    constexpr std::array<scalar_handle, 3> components() const
    {
        return {x(), y(), z()};
    }

    constexpr bool operator==(const vector_handle&) const = default;
    constexpr explicit operator bool() const { return base >= 0; }
};

// ---------------------------------------------------------------------------
// Consteval factory functions.
//
// These guarantee compile-time bounds checking. If you write
//   constexpr auto h = make_scalar_handle<layout>(2);
// and the layout only has 1 active scalar, the program is ill-formed.
//
// Use consteval so that even in non-constexpr contexts the call is
// evaluated at compile time and the assertion fires as a compile error.
// ---------------------------------------------------------------------------

template <int MaxS, int MaxV>
consteval scalar_handle make_scalar_handle(field_layout<MaxS, MaxV> layout, int i)
{
    // Bounds check against active count.
    if (i < 0 || i >= layout.n_scalars) {
        // Trigger a compile-time error: calling a non-constexpr function
        // inside consteval makes the program ill-formed.
        throw "scalar index out of bounds";
    }
    return {i * field_layout<MaxS, MaxV>::scalar_stride};
}

template <int MaxS, int MaxV>
consteval vector_handle make_vector_handle(field_layout<MaxS, MaxV> layout, int i)
{
    if (i < 0 || i >= layout.n_vectors) {
        throw "vector index out of bounds";
    }
    return {field_layout<MaxS, MaxV>::vector_base +
            i * field_layout<MaxS, MaxV>::vector_stride};
}

// Convenience: make a scalar handle with only capacity checking (no active count).
// Useful during layout construction before n_scalars is finalized.
template <int MaxS, int MaxV>
consteval scalar_handle make_scalar_handle_unchecked(field_layout<MaxS, MaxV>, int i)
{
    if (i < 0 || i >= MaxS) {
        throw "scalar index exceeds layout capacity";
    }
    return {i * field_layout<MaxS, MaxV>::scalar_stride};
}

template <int MaxS, int MaxV>
consteval vector_handle make_vector_handle_unchecked(field_layout<MaxS, MaxV>, int i)
{
    if (i < 0 || i >= MaxV) {
        throw "vector index exceeds layout capacity";
    }
    return {field_layout<MaxS, MaxV>::vector_base +
            i * field_layout<MaxS, MaxV>::vector_stride};
}

// ---------------------------------------------------------------------------
// Free-function index computation (matches the specification exactly).
// These are constexpr (not consteval) so they can be used in both
// compile-time and runtime contexts.
// ---------------------------------------------------------------------------

template <int MaxS, int MaxV>
constexpr int scalar_D(field_layout<MaxS, MaxV>, int i)
{
    return i * field_layout<MaxS, MaxV>::scalar_stride + 0;
}

template <int MaxS, int MaxV>
constexpr int scalar_Rx(field_layout<MaxS, MaxV>, int i)
{
    return i * field_layout<MaxS, MaxV>::scalar_stride + 1;
}

template <int MaxS, int MaxV>
constexpr int scalar_Ry(field_layout<MaxS, MaxV>, int i)
{
    return i * field_layout<MaxS, MaxV>::scalar_stride + 2;
}

template <int MaxS, int MaxV>
constexpr int scalar_Rz(field_layout<MaxS, MaxV>, int i)
{
    return i * field_layout<MaxS, MaxV>::scalar_stride + 3;
}

template <int MaxS, int MaxV>
constexpr int vector_xD(field_layout<MaxS, MaxV>, int i)
{
    return field_layout<MaxS, MaxV>::vector_base + i * 12 + 0;
}

template <int MaxS, int MaxV>
constexpr int vector_yD(field_layout<MaxS, MaxV>, int i)
{
    return field_layout<MaxS, MaxV>::vector_base + i * 12 + 4;
}

template <int MaxS, int MaxV>
constexpr int vector_zD(field_layout<MaxS, MaxV>, int i)
{
    return field_layout<MaxS, MaxV>::vector_base + i * 12 + 8;
}

// ---------------------------------------------------------------------------
// Selector dispatch: compile-time multi-selection patterns.
//
// These replicate the current sel::D / sel::R semantics but operate on
// handle types instead of nested tuples.  The key insight: applying a
// selector to a scalar returns a single buf_handle; applying the same
// selector to a vector returns an array of buf_handles.
// ---------------------------------------------------------------------------

namespace handle_sel
{

// --- sel::D ---
// scalar: returns 1 domain buffer.
// vector: returns 3 domain buffers (x.D, y.D, z.D).

struct D_selector {
    constexpr buf_handle operator()(scalar_handle h) const
    {
        return h.D();
    }

    constexpr std::array<buf_handle, 3> operator()(vector_handle h) const
    {
        return {h.Dx(), h.Dy(), h.Dz()};
    }
};

// --- sel::Rx / sel::Ry / sel::Rz ---
// scalar: returns 1 boundary buffer.
// vector: returns 3 boundary buffers (x.R_, y.R_, z.R_).

struct Rx_selector {
    constexpr buf_handle operator()(scalar_handle h) const
    {
        return h.Rx();
    }

    constexpr std::array<buf_handle, 3> operator()(vector_handle h) const
    {
        return {h.xRx(), h.yRx(), h.zRx()};
    }
};

struct Ry_selector {
    constexpr buf_handle operator()(scalar_handle h) const
    {
        return h.Ry();
    }

    constexpr std::array<buf_handle, 3> operator()(vector_handle h) const
    {
        return {h.xRy(), h.yRy(), h.zRy()};
    }
};

struct Rz_selector {
    constexpr buf_handle operator()(scalar_handle h) const
    {
        return h.Rz();
    }

    constexpr std::array<buf_handle, 3> operator()(vector_handle h) const
    {
        return {h.xRz(), h.yRz(), h.zRz()};
    }
};

// --- sel::R ---
// scalar: returns 3 boundary buffers (Rx, Ry, Rz).
// vector: returns 9 boundary buffers (xRx, yRx, zRx, xRy, yRy, zRy, xRz, yRz, zRz).
// Note: the vector ordering matches the current mp_list ordering in selector.hpp.

struct R_selector {
    constexpr std::array<buf_handle, 3> operator()(scalar_handle h) const
    {
        return h.R();
    }

    constexpr std::array<buf_handle, 9> operator()(vector_handle h) const
    {
        return {h.xRx(), h.yRx(), h.zRx(),
                h.xRy(), h.yRy(), h.zRy(),
                h.xRz(), h.yRz(), h.zRz()};
    }
};

// --- Component selectors for vectors only (sel::xR, sel::yR, sel::zR) ---

struct xR_selector {
    constexpr std::array<buf_handle, 3> operator()(vector_handle h) const
    {
        return {h.xRx(), h.xRy(), h.xRz()};
    }
};

struct yR_selector {
    constexpr std::array<buf_handle, 3> operator()(vector_handle h) const
    {
        return {h.yRx(), h.yRy(), h.yRz()};
    }
};

struct zR_selector {
    constexpr std::array<buf_handle, 3> operator()(vector_handle h) const
    {
        return {h.zRx(), h.zRy(), h.zRz()};
    }
};

// --- Individual vector component selectors (sel::Dx, sel::Dy, sel::Dz) ---

struct Dx_selector {
    constexpr buf_handle operator()(vector_handle h) const { return h.Dx(); }
};

struct Dy_selector {
    constexpr buf_handle operator()(vector_handle h) const { return h.Dy(); }
};

struct Dz_selector {
    constexpr buf_handle operator()(vector_handle h) const { return h.Dz(); }
};

// Selector instances.
inline constexpr D_selector  D{};
inline constexpr Rx_selector Rx{};
inline constexpr Ry_selector Ry{};
inline constexpr Rz_selector Rz{};
inline constexpr R_selector  R{};
inline constexpr xR_selector xR{};
inline constexpr yR_selector yR{};
inline constexpr zR_selector zR{};
inline constexpr Dx_selector Dx{};
inline constexpr Dy_selector Dy{};
inline constexpr Dz_selector Dz{};

} // namespace handle_sel

// ---------------------------------------------------------------------------
// handle_for_each: visit every buf_handle in a scalar/vector handle.
// Replaces the recursive for_each/transform over NestedTuples.
// ---------------------------------------------------------------------------

template <typename F>
constexpr void handle_for_each(F&& f, scalar_handle h)
{
    f(h.D());
    f(h.Rx());
    f(h.Ry());
    f(h.Rz());
}

template <typename F>
constexpr void handle_for_each(F&& f, vector_handle h)
{
    handle_for_each(f, h.x());
    handle_for_each(f, h.y());
    handle_for_each(f, h.z());
}

// Paired iteration (e.g., copy from one handle's buffers to another's).
template <typename F>
constexpr void handle_for_each(F&& f, scalar_handle a, scalar_handle b)
{
    f(a.D(),  b.D());
    f(a.Rx(), b.Rx());
    f(a.Ry(), b.Ry());
    f(a.Rz(), b.Rz());
}

template <typename F>
constexpr void handle_for_each(F&& f, vector_handle a, vector_handle b)
{
    handle_for_each(f, a.x(), b.x());
    handle_for_each(f, a.y(), b.y());
    handle_for_each(f, a.z(), b.z());
}

// ---------------------------------------------------------------------------
// NTTP usage examples (for documentation / static_assert verification).
//
// Because all handle types are structural (all public members, literal type,
// defaulted operator==), they can be used directly as template parameters:
//
//   template <scalar_handle H>
//   void kernel(field_storage& storage) {
//       auto* d_ptr = storage.data(H.D().id);
//       // ...
//   }
//
// This enables zero-overhead template dispatch: the buffer index is baked
// into the generated code as an immediate constant.
// ---------------------------------------------------------------------------

// Compile-time proof that handle types satisfy structural type requirements.
static_assert(std::is_trivially_copyable_v<buf_handle>);
static_assert(std::is_trivially_copyable_v<scalar_handle>);
static_assert(std::is_trivially_copyable_v<vector_handle>);

// Structural type requirements: literal type + all members public.
// (The defaulted operator== is already declared above.)
static_assert(std::is_aggregate_v<buf_handle>);
static_assert(std::is_aggregate_v<scalar_handle>);
static_assert(std::is_aggregate_v<vector_handle>);

// ---------------------------------------------------------------------------
// NTTP accessor template: demonstrates zero-overhead dispatch.
//
// The handle is a template parameter, so buffer indices become compile-time
// constants in the generated code.
// ---------------------------------------------------------------------------

template <scalar_handle H>
struct scalar_accessor {
    static constexpr int d_index  = H.D().id;
    static constexpr int rx_index = H.Rx().id;
    static constexpr int ry_index = H.Ry().id;
    static constexpr int rz_index = H.Rz().id;
};

template <vector_handle H>
struct vector_accessor {
    static constexpr scalar_accessor<H.x()> x{};
    static constexpr scalar_accessor<H.y()> y{};
    static constexpr scalar_accessor<H.z()> z{};
};

// ---------------------------------------------------------------------------
// System-specific layout aliases.
//
// Each system defines its own layout. This keeps MaxS/MaxV as template
// parameters of the layout (not global constants), while providing
// convenient aliases for each system's configuration.
// ---------------------------------------------------------------------------

// Heat system: 1 scalar, 0 vectors.
using heat_layout = field_layout<1, 0>;

// Scalar wave system: 1 scalar, 1 vector (grad_G, du stored separately).
// (The main field has 1 scalar, 0 vectors, but auxiliary fields need vectors.)
using scalar_wave_layout = field_layout<1, 1>;

// A general-purpose layout for systems with moderate field counts.
using general_layout = field_layout<8, 4>;

// ---------------------------------------------------------------------------
// Compile-time layout and handle construction example.
//
// constexpr auto layout = heat_layout{.n_scalars = 1, .n_vectors = 0};
// constexpr auto u = make_scalar_handle(layout, 0);
// // u.D().id == 0, u.Rx().id == 1, u.Ry().id == 2, u.Rz().id == 3
//
// constexpr auto layout2 = scalar_wave_layout{.n_scalars = 1, .n_vectors = 1};
// constexpr auto u2 = make_scalar_handle(layout2, 0);
// constexpr auto v  = make_vector_handle(layout2, 0);
// // v.x().D().id == 4 (= 1*4 + 0), v.x().Rx().id == 5, ...
// // v.z().Rz().id == 15 (= 4 + 11)
// ---------------------------------------------------------------------------

namespace detail
{

// Verify the arithmetic at compile time.
constexpr auto test_layout_ = field_layout<2, 1>{.n_scalars = 2, .n_vectors = 1};

// Scalar handles.
constexpr auto s0_ = make_scalar_handle(test_layout_, 0);
constexpr auto s1_ = make_scalar_handle(test_layout_, 1);

static_assert(s0_.D().id  == 0);
static_assert(s0_.Rx().id == 1);
static_assert(s0_.Ry().id == 2);
static_assert(s0_.Rz().id == 3);

static_assert(s1_.D().id  == 4);
static_assert(s1_.Rx().id == 5);
static_assert(s1_.Ry().id == 6);
static_assert(s1_.Rz().id == 7);

// Vector handle.
constexpr auto v0_ = make_vector_handle(test_layout_, 0);

// vector_base = 2 * 4 = 8.
static_assert(v0_.base == 8);

// x component: [8, 9, 10, 11]
static_assert(v0_.x().D().id  == 8);
static_assert(v0_.x().Rx().id == 9);
static_assert(v0_.x().Ry().id == 10);
static_assert(v0_.x().Rz().id == 11);

// y component: [12, 13, 14, 15]
static_assert(v0_.y().D().id  == 12);
static_assert(v0_.y().Rx().id == 13);
static_assert(v0_.y().Ry().id == 14);
static_assert(v0_.y().Rz().id == 15);

// z component: [16, 17, 18, 19]
static_assert(v0_.z().D().id  == 16);
static_assert(v0_.z().Rx().id == 17);
static_assert(v0_.z().Ry().id == 18);
static_assert(v0_.z().Rz().id == 19);

// Total capacity.
static_assert(test_layout_.total_buffers == 2 * 4 + 1 * 12);  // 20

// Selector dispatch verification.
static_assert(handle_sel::D(s0_).id == 0);
static_assert(handle_sel::Rx(s0_).id == 1);

constexpr auto v0_domain_ = handle_sel::D(v0_);
static_assert(v0_domain_[0].id == 8);   // x.D
static_assert(v0_domain_[1].id == 12);  // y.D
static_assert(v0_domain_[2].id == 16);  // z.D

constexpr auto v0_R_ = handle_sel::R(v0_);
static_assert(v0_R_[0].id == 9);   // xRx
static_assert(v0_R_[1].id == 13);  // yRx
static_assert(v0_R_[2].id == 17);  // zRx
static_assert(v0_R_[3].id == 10);  // xRy
static_assert(v0_R_[4].id == 14);  // yRy
static_assert(v0_R_[5].id == 18);  // zRy
static_assert(v0_R_[6].id == 11);  // xRz
static_assert(v0_R_[7].id == 15);  // yRz
static_assert(v0_R_[8].id == 19);  // zRz

constexpr auto s0_R_ = handle_sel::R(s0_);
static_assert(s0_R_[0].id == 1);  // Rx
static_assert(s0_R_[1].id == 2);  // Ry
static_assert(s0_R_[2].id == 3);  // Rz

// Free-function index computation.
static_assert(scalar_D(test_layout_, 0) == 0);
static_assert(scalar_D(test_layout_, 1) == 4);
static_assert(scalar_Rx(test_layout_, 0) == 1);
static_assert(vector_xD(test_layout_, 0) == 8);
static_assert(vector_yD(test_layout_, 0) == 12);
static_assert(vector_zD(test_layout_, 0) == 16);

// NTTP usage: the handle is a compile-time constant.
constexpr scalar_accessor<s0_> s0_acc_{};
static_assert(s0_acc_.d_index == 0);
static_assert(s0_acc_.rx_index == 1);

constexpr vector_accessor<v0_> v0_acc_{};
static_assert(v0_acc_.x.d_index == 8);
static_assert(v0_acc_.y.d_index == 12);
static_assert(v0_acc_.z.d_index == 16);

} // namespace detail

} // namespace ccs

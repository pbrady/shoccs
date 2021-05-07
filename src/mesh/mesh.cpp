#include "mesh.hpp"

namespace ccs
{

namespace
{
// We need to return lines of boundary_info so the caller
// can build up the appropriate operators
// the lines will be of a few different types:
// [domain, domain]
// [domain, object]
// [object, domain]
// [onbject, object]
//
// As with the solid point identification algorithm, we do not
// properly handle the case of fully solid lines
template <int J, int K>
constexpr bool same_plane(const int3& x, const int3& y)
{
    return x[J] == y[J] && x[K] == y[K];
}

// constexpr auto offset = [](int3 n) {
//     return [n](int3 ijk) { return ijk[0] * n[1] * n[2] + ijk[1] * n[2] + ijk[2]; };
// };

template <auto I>
void init_line(std::vector<line>& v, int3 extents, std::span<const mesh_object_info> r)
{
    // early exit if we are building operators in this direction
    if (extents[I] == 1) return;

    constexpr auto S = index::dir<I>::slow;
    constexpr auto F = index::dir<I>::fast;
    // auto off = offset(extents);

    integer ns = extents[S];
    integer nf = extents[F];

    v.reserve(ns * nf + r.size());
    auto first = rs::begin(r);
    auto last = rs::end(r);

    int3 left{};
    int3 right{};
    for (integer s = 0; s < ns; s++) {
        left[S] = s;
        right[S] = s;
        for (integer f = 0; f < nf; f++) {
            left[F] = f;
            right[F] = f;

            left[I] = 0;
            right[I] = extents[I] - 1;

            std::optional<boundary> left_boundary = boundary{left, std::nullopt};

            while (first != last && same_plane<S, F>(left, first->solid_coord)) {
                if (first->ray_outside) {
                    // set the `right` point and add both to line
                    v.emplace_back(
                        // off(left_boundary->mesh_coordinate),
                        index::stride<I>(extents),
                        *left_boundary,
                        boundary{.mesh_coordinate = first->solid_coord,
                                 .object = object_boundary{
                                     first - rs::begin(r), first->shape_id, first->psi}});
                    // invalidate the boundary point to indicate it was consumed
                    left_boundary.reset();
                } else {
                    // set the left_boundary and allow the next loop to process
                    left_boundary =
                        boundary{.mesh_coordinate = first->solid_coord,
                                 .object = object_boundary{
                                     first - rs::begin(r), first->shape_id, first->psi}};
                }
                ++first;
            }

            // consume the left boundary
            if (left_boundary) {
                v.emplace_back(
                    // off(left_boundary->mesh_coordinate),
                    index::stride<I>(extents),
                    *left_boundary,
                    boundary{.mesh_coordinate = right, .object = std::nullopt});
            }
        }
    }
}
} // namespace

mesh::mesh(const index_extents& extents, const domain_extents& bounds)
    : mesh{extents, bounds, std::vector<shape>{}}
{
}

mesh::mesh(const index_extents& extents,
           const domain_extents& bounds,
           const std::vector<shape>& shapes)
    : cart{extents.extents, bounds.min, bounds.max}, geometry{shapes, cart}
{
    init_line<0>(lines_[0], cart.extents(), geometry.R(0));
    init_line<1>(lines_[1], cart.extents(), geometry.R(1));
    init_line<2>(lines_[2], cart.extents(), geometry.R(2));
}

bool mesh::dirichlet_line(const int3& start, int dir, const bcs::Grid& cart_bcs) const
{
    bool result = false;

    auto f = [&](int i) {
        return (cart_bcs[i].left == bcs::Dirichlet &&
                cart.on_boundary(i, false, start)) ||
               (cart_bcs[i].right == bcs::Dirichlet && cart.on_boundary(i, true, start));
    };

    for (int i = 0; i < dir; i++) { result = result || f(i); }
    for (int i = dir + 1; i < 3; i++) { result = result || f(i); }

    return result;
}

} // namespace ccs
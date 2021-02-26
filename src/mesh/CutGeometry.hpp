#pragma once

#include "Cartesian.hpp"
#include "Shapes.hpp"
#include "types.hpp"
#include <span>
#include <vector>

namespace ccs::mesh
{

// only difference between this and hit_info is the solid_coord.
struct mesh_object_info {
    real psi; // 1D cutcell distance
    real3 position;
    bool ray_outside;
    int3 solid_coord;
    int shape_id;
};

class CutGeometry
{
    // mesh / object intersection info for all rays
    std::vector<mesh_object_info> rx_;
    std::vector<mesh_object_info> ry_;
    std::vector<mesh_object_info> rz_;
    // mesh / object intersection info rays organized by shape_id
    std::vector<std::vector<mesh_object_info>> rx_m_;
    std::vector<std::vector<mesh_object_info>> ry_m_;
    std::vector<std::vector<mesh_object_info>> rz_m_;
    // solid points not associated with mesh / object intersections
    std::vector<int3> sx_;
    std::vector<int3> sy_;
    std::vector<int3> sz_;

public:
    CutGeometry() = default;

    // constructor for uniform meshes.
    CutGeometry(std::span<const shape>, const Cartesian& m, bool check_domain = true);

    // Intersection of rays in x and object `shape_id`
    std::span<const mesh_object_info> Rx(int shape_id) const;
    // Intersection of rays in x and all objects
    std::span<const mesh_object_info> Rx() const;
    // Intersection of rays in y and object `shape_id`
    std::span<const mesh_object_info> Ry(int shape_id) const;
    // Intersection of rays in y and all objects
    std::span<const mesh_object_info> Ry() const;
    // Intersection of rays in z and object `shape_id`
    std::span<const mesh_object_info> Rz(int shape_id) const;
    // Intersection of rays in z and all objects
    std::span<const mesh_object_info> Rz() const;

    std::span<const mesh_object_info> R(int dir) const
    {
        switch (dir) {
        case 0:
            return Rx();
        case 1:
            return Ry();
        default:
            return Rz();
        }
    }

    // auto Rxyz() const { return vector_range{Rx(), Ry(), Rz()}; }

    // details about points in solid
    std::span<const int3> Sx() const;
    std::span<const int3> Sy() const;
    std::span<const int3> Sz() const;

    std::span<const int3> S(int dir) const
    {
        switch (dir) {
        case 0:
            return Sx();
        case 1:
            return Sy();
        default:
            return Sz();
        }
    }

    // auto Sxyz() const { return vector_range{Sx(), Sy(), Sz()}; }
};

} // namespace ccs::mesh
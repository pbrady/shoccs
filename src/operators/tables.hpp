#pragma once

// Operator tables over one packed vector V = [D | Rx | Ry | Rz], read out of the
// assembled derivative matrices. The data contract is
// docs/reference/pyshoccs-export.md (Sections 1, 2.3 and 4).

#include "laplacian.hpp"
#include "mesh/mesh.hpp"
#include "types.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace ccs
{

// Offsets of the four buffers of a scalar field inside V (Section 1).
struct packed_layout {
    int3 extents{};
    integer n_D{};
    std::array<integer, 3> n_R{};
    std::array<integer, 4> offsets{}; // D, Rx, Ry, Rz
    integer n_V{};

    packed_layout() = default;
    explicit packed_layout(const mesh&);

    integer off_R(int r) const { return offsets[r + 1]; }

    // distance in D between neighbours along dir (z fastest)
    integer stride(int dir) const;
};

// Part codes of Section 2.3. N has no code there because it is written to its
// own arrays; 4 is used for it internally.
namespace coo_part
{
constexpr uint8_t O_left = 0;
constexpr uint8_t O_interior = 1;
constexpr uint8_t O_right = 2;
constexpr uint8_t B = 3;
constexpr uint8_t N = 4;
constexpr uint8_t Bf(int r) { return 5 + 2 * r; }
constexpr uint8_t Br(int r) { return 6 + 2 * r; }
} // namespace coo_part

struct coo_entry {
    integer row;
    integer col;
    real val;
    uint8_t part;
    int32_t line; // index into mesh::lines(dir) for O entries, -1 otherwise
};

struct derivative_coo {
    std::vector<coo_entry> A; // O, B, Bf_r and Br_r entries, rows and columns in V
    std::vector<coo_entry> N; // rows in D (packed), columns in the Neumann data buffer
};

// The lines of mesh::lines(dir) that domain_discretization kept, in block order:
// block i of derivative{dir}'s O matrix was built from lines(dir)[result[i]].
std::vector<int32_t> block_line_map(const mesh&, int dir, const bcs::Grid&);

// Every stored entry of one derivative's matrices (explicit zeros included), in
// storage order: O line by line (left, interior, right), then B, Bfx, Brx, Bfy,
// Bry, Bfz, Brz. block_lines may be empty, in which case every line is -1.
derivative_coo
extract_coo(const derivative&, const packed_layout&, std::span<const int32_t> block_lines);

// Role bits of Section 4.
namespace role_bit
{
constexpr uint8_t interior(int dir) { return uint8_t(1u << dir); }
constexpr uint8_t fluid = 1u << 3;
constexpr uint8_t dirichlet_face = 1u << 4;
constexpr uint8_t any_interior = 0x7;
} // namespace role_bit

// The structured form of a Laplacian: per-direction axis taps on the rows flagged
// interior for that direction, plus one exception CSR over V holding every other
// entry (duplicates summed, rows unique and ascending).
struct compact_tables {
    packed_layout layout;
    std::vector<uint8_t> role;             // [n_D]
    int width{};                           // 2p + 1
    std::array<std::vector<real>, 3> taps; // [3][width]; zero when a direction has no lines
    std::vector<int32_t> exc_rows;         // unique, ascending V rows
    std::vector<integer> exc_rowptr;       // [exc_rows.size() + 1]
    std::vector<int32_t> exc_col;          // V columns
    std::vector<real> exc_val;
};

compact_tables build_compact_tables(const laplacian&,
                                    const packed_layout&,
                                    std::span<const int> fluid_idx,
                                    std::span<const int> dirichlet_face_idx);

// out_V = the Section 4 total: axis taps on every D point (0 when no bit is set),
// then the exception rows added. Serial.
void apply_structured(const compact_tables&, std::span<const real> u_V, std::span<real> out_V);

} // namespace ccs

#include "../src/stencils/stencil.hpp"
#include <fmt/format.h>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>

using namespace ccs;

// Helper to write JSON array
template<typename T>
std::string to_json_array(const std::vector<T>& arr) {
    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "[";
    for (size_t i = 0; i < arr.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << arr[i];
    }
    oss << "]";
    return oss.str();
}

// Helper to write JSON object
struct TestCase {
    std::string name;
    real h;
    real psi;
    bool ray_outside;
    std::string bc_type;
    int r;
    int t;
    std::vector<real> coefficients;

    std::string to_json(int indent = 2) const {
        std::string ind(indent, ' ');
        std::string ind2(indent + 2, ' ');
        std::ostringstream oss;
        oss << std::setprecision(17);
        oss << ind << "{\n";
        oss << ind2 << "\"name\": \"" << name << "\",\n";
        oss << ind2 << "\"h\": " << h << ",\n";
        oss << ind2 << "\"psi\": " << psi << ",\n";
        oss << ind2 << "\"ray_outside\": " << (ray_outside ? "true" : "false") << ",\n";
        oss << ind2 << "\"bc_type\": \"" << bc_type << "\",\n";
        oss << ind2 << "\"r\": " << r << ",\n";
        oss << ind2 << "\"t\": " << t << ",\n";
        oss << ind2 << "\"coefficients\": " << to_json_array(coefficients) << "\n";
        oss << ind << "}";
        return oss.str();
    }
};

struct InteriorTestCase {
    std::string name;
    real h;
    std::vector<real> coefficients;

    std::string to_json(int indent = 2) const {
        std::string ind(indent, ' ');
        std::string ind2(indent + 2, ' ');
        std::ostringstream oss;
        oss << std::setprecision(17);
        oss << ind << "{\n";
        oss << ind2 << "\"name\": \"" << name << "\",\n";
        oss << ind2 << "\"h\": " << h << ",\n";
        oss << ind2 << "\"coefficients\": " << to_json_array(coefficients) << "\n";
        oss << ind << "}";
        return oss.str();
    }
};

int main() {
    std::ofstream outfile("/home/user/shoccs/tools/stencil_reference_data.json");
    outfile << std::setprecision(17);

    outfile << "{\n";
    outfile << "  \"metadata\": {\n";
    outfile << "    \"description\": \"Reference data from C++ SHOCCS implementation for Python validation\",\n";
    outfile << "    \"generator\": \"generate_stencil_reference.cpp\",\n";
    outfile << "    \"precision\": \"double (IEEE 754)\"\n";
    outfile << "  },\n";

    // ============================================================================
    // E2_1 STENCIL
    // ============================================================================
    outfile << "  \"E2_1\": {\n";
    outfile << "    \"alpha\": [1.0, 2.0, 3.0, -1.0],\n";

    std::array<real, 4> alpha{1.0, 2.0, 3.0, -1.0};
    auto st = stencils::make_E2_1(alpha);

    // Interior stencil
    outfile << "    \"interior\": [\n";
    std::vector<real> h_values{0.1, 0.5, 1.0, 2.0};
    std::vector<InteriorTestCase> interior_cases;

    for (auto h : h_values) {
        std::vector<real> c(3);  // 2p+1 = 3 for p=1
        auto result = st.interior(h, c);
        interior_cases.push_back({
            fmt::format("h={}", h),
            h,
            std::vector<real>(result.begin(), result.end())
        });
    }

    for (size_t i = 0; i < interior_cases.size(); ++i) {
        outfile << interior_cases[i].to_json(6);
        if (i < interior_cases.size() - 1) outfile << ",";
        outfile << "\n";
    }
    outfile << "    ],\n";

    // Floating boundary conditions
    outfile << "    \"floating\": [\n";
    std::vector<TestCase> floating_cases;

    std::vector<real> psi_values{0.1, 0.25, 0.5, 0.75, 0.9, 1.0};
    std::vector<real> h_boundary{0.5, 1.0, 2.0};

    for (auto h : h_boundary) {
        for (auto psi : psi_values) {
            for (bool ray_outside : {false, true}) {
                auto [p, r, t, x] = st.query(bcs::Floating);
                std::vector<real> c(r * t);
                std::vector<real> ex;

                auto result = st.nbs(h, bcs::Floating, psi, ray_outside, c, ex);

                floating_cases.push_back({
                    fmt::format("h={}_psi={}_ray_{}", h, psi, ray_outside ? "outside" : "inside"),
                    h,
                    psi,
                    ray_outside,
                    "floating",
                    r,
                    t,
                    std::vector<real>(result.begin(), result.end())
                });
            }
        }
    }

    for (size_t i = 0; i < floating_cases.size(); ++i) {
        outfile << floating_cases[i].to_json(6);
        if (i < floating_cases.size() - 1) outfile << ",";
        outfile << "\n";
    }
    outfile << "    ],\n";

    // Dirichlet boundary conditions
    outfile << "    \"dirichlet\": [\n";
    std::vector<TestCase> dirichlet_cases;

    std::vector<real> psi_dirichlet{0.0, 0.1, 0.5, 0.9};

    for (auto h : h_boundary) {
        for (auto psi : psi_dirichlet) {
            for (bool ray_outside : {false, true}) {
                auto [p, r, t, x] = st.query(bcs::Dirichlet);
                std::vector<real> c(r * t);
                std::vector<real> ex;

                auto result = st.nbs(h, bcs::Dirichlet, psi, ray_outside, c, ex);

                dirichlet_cases.push_back({
                    fmt::format("h={}_psi={}_ray_{}", h, psi, ray_outside ? "outside" : "inside"),
                    h,
                    psi,
                    ray_outside,
                    "dirichlet",
                    r,
                    t,
                    std::vector<real>(result.begin(), result.end())
                });
            }
        }
    }

    for (size_t i = 0; i < dirichlet_cases.size(); ++i) {
        outfile << dirichlet_cases[i].to_json(6);
        if (i < dirichlet_cases.size() - 1) outfile << ",";
        outfile << "\n";
    }
    outfile << "    ]\n";
    outfile << "  },\n";

    // ============================================================================
    // E2_2 STENCIL
    // ============================================================================
    outfile << "  \"E2_2\": {\n";

    auto st2 = stencils::make_E2_2();

    // Interior stencil
    outfile << "    \"interior\": [\n";
    interior_cases.clear();

    for (auto h : h_values) {
        std::vector<real> c(5);  // 2p+1 = 5 for p=2
        auto result = st2.interior(h, c);
        interior_cases.push_back({
            fmt::format("h={}", h),
            h,
            std::vector<real>(result.begin(), result.end())
        });
    }

    for (size_t i = 0; i < interior_cases.size(); ++i) {
        outfile << interior_cases[i].to_json(6);
        if (i < interior_cases.size() - 1) outfile << ",";
        outfile << "\n";
    }
    outfile << "    ],\n";

    // Floating boundary conditions (limited set)
    outfile << "    \"floating\": [\n";
    floating_cases.clear();

    std::vector<real> psi_e2_2{0.5, 1.0};

    for (auto psi : psi_e2_2) {
        for (bool ray_outside : {false, true}) {
            real h = 1.0;
            auto [p, r, t, x] = st2.query(bcs::Floating);
            std::vector<real> c(r * t);
            std::vector<real> ex;

            auto result = st2.nbs(h, bcs::Floating, psi, ray_outside, c, ex);

            floating_cases.push_back({
                fmt::format("h={}_psi={}_ray_{}", h, psi, ray_outside ? "outside" : "inside"),
                h,
                psi,
                ray_outside,
                "floating",
                r,
                t,
                std::vector<real>(result.begin(), result.end())
            });
        }
    }

    for (size_t i = 0; i < floating_cases.size(); ++i) {
        outfile << floating_cases[i].to_json(6);
        if (i < floating_cases.size() - 1) outfile << ",";
        outfile << "\n";
    }
    outfile << "    ]\n";
    outfile << "  },\n";

    // ============================================================================
    // polyE2_1 STENCIL
    // ============================================================================
    outfile << "  \"polyE2_1\": {\n";
    outfile << "    \"floating_alpha\": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],\n";
    outfile << "    \"dirichlet_alpha\": [-0.1, -0.2, -0.3],\n";
    outfile << "    \"interpolant_alpha\": [0.15, 0.25, 0.35, 0.45],\n";

    std::array<real, 6> fa{0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    std::array<real, 3> da{-0.1, -0.2, -0.3};
    std::array<real, 4> ia{0.15, 0.25, 0.35, 0.45};
    auto stp = stencils::make_polyE2_1(fa, da, ia);

    // Interior stencil
    outfile << "    \"interior\": [\n";
    interior_cases.clear();

    for (auto h : h_values) {
        std::vector<real> c(3);  // 2p+1 = 3 for p=1
        auto result = stp.interior(h, c);
        interior_cases.push_back({
            fmt::format("h={}", h),
            h,
            std::vector<real>(result.begin(), result.end())
        });
    }

    for (size_t i = 0; i < interior_cases.size(); ++i) {
        outfile << interior_cases[i].to_json(6);
        if (i < interior_cases.size() - 1) outfile << ",";
        outfile << "\n";
    }
    outfile << "    ],\n";

    // Floating boundary conditions
    outfile << "    \"floating\": [\n";
    floating_cases.clear();

    psi_values = {0.2, 0.5, 0.8};

    for (auto psi : psi_values) {
        for (bool ray_outside : {false, true}) {
            real h = 1.0;
            auto [p, r, t, x] = stp.query(bcs::Floating);
            std::vector<real> c(r * t);
            std::vector<real> ex;

            auto result = stp.nbs(h, bcs::Floating, psi, ray_outside, c, ex);

            floating_cases.push_back({
                fmt::format("h={}_psi={}_ray_{}", h, psi, ray_outside ? "outside" : "inside"),
                h,
                psi,
                ray_outside,
                "floating",
                r,
                t,
                std::vector<real>(result.begin(), result.end())
            });
        }
    }

    for (size_t i = 0; i < floating_cases.size(); ++i) {
        outfile << floating_cases[i].to_json(6);
        if (i < floating_cases.size() - 1) outfile << ",";
        outfile << "\n";
    }
    outfile << "    ],\n";

    // Dirichlet boundary conditions
    outfile << "    \"dirichlet\": [\n";
    dirichlet_cases.clear();

    psi_dirichlet = {0.0, 0.001, 0.5};

    for (auto psi : psi_dirichlet) {
        for (bool ray_outside : {false, true}) {
            real h = 1.0;
            auto [p, r, t, x] = stp.query(bcs::Dirichlet);
            std::vector<real> c(r * t);
            std::vector<real> ex;

            auto result = stp.nbs(h, bcs::Dirichlet, psi, ray_outside, c, ex);

            dirichlet_cases.push_back({
                fmt::format("h={}_psi={}_ray_{}", h, psi, ray_outside ? "outside" : "inside"),
                h,
                psi,
                ray_outside,
                "dirichlet",
                r,
                t,
                std::vector<real>(result.begin(), result.end())
            });
        }
    }

    for (size_t i = 0; i < dirichlet_cases.size(); ++i) {
        outfile << dirichlet_cases[i].to_json(6);
        if (i < dirichlet_cases.size() - 1) outfile << ",";
        outfile << "\n";
    }
    outfile << "    ]\n";
    outfile << "  }\n";

    outfile << "}\n";
    outfile.close();

    fmt::print("Reference data written to: /home/user/shoccs/tools/stencil_reference_data.json\n");
    fmt::print("Total test cases generated:\n");
    fmt::print("  E2_1: {} interior, {} floating, {} dirichlet\n",
               h_values.size(), 36, 24);
    fmt::print("  E2_2: {} interior, {} floating\n",
               h_values.size(), 4);
    fmt::print("  polyE2_1: {} interior, {} floating, {} dirichlet\n",
               h_values.size(), 6, 6);

    return 0;
}

#include "field_data.hpp"
#include <fstream>


namespace ccs
{

void field_data::write_geom(std::span<const std::string> filenames,
                            tuple<std::span<const mesh_object_info>,
                                  std::span<const mesh_object_info>,
                                  std::span<const mesh_object_info>> t) const
{
    auto f = [&]<int I>() {
        auto rng = get<I>(t);
        std::ofstream o(filenames[I]);
        for (auto&& info : rng) {
            auto&& pos = info.position;
            if (ix[2] == 1) {
                real3 tmp{pos[2], pos[1], pos[0]};
                const real* d = tmp.data();
                o.write(reinterpret_cast<const char*>(d),
                        tmp.size() * sizeof(real));
            } else {
                const real* d = pos.data();
                o.write(reinterpret_cast<const char*>(d),
                        pos.size() * sizeof(real));
            }
        }
    };

    f.template operator()<0>();
    f.template operator()<1>();
    f.template operator()<2>();
}

void field_data::write(field_view f, std::span<const std::string> filenames) const
{
    unsigned long sz = ix[0] * ix[1] * ix[2] * sizeof(real);

    auto& scalars = f.scalars();
    for (size_t idx = 0; idx < filenames.size(); ++idx) {
        auto& fname = filenames[idx];
        auto& sc = scalars[idx];
        std::ofstream o(fname);

        const real* d = get<si::D>(sc).data();
        o.write(reinterpret_cast<const char*>(d), sz);

        auto g = [&]<int I>(auto&& r) {
            if (auto&& rng = get<I>(r); rng.size() > 0) {
                d = rng.data();
                o.write(reinterpret_cast<const char*>(d), rng.size() * sizeof(*d));
            }
        };
        g.template operator()<0>(sc | sel::R);
        g.template operator()<1>(sc | sel::R);
        g.template operator()<2>(sc | sel::R);
    }
}

} // namespace ccs

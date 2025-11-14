add_executable(generate_stencil_reference generate_stencil_reference.cpp)
target_link_libraries(generate_stencil_reference
    PRIVATE
    stencils
    fmt::fmt
)

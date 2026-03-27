add_library(shoccs-matrices 
    dense.cpp
    circulant.cpp
    inner_block.cpp 
    csr.cpp 
    unit_stride_visitor.cpp 
    coefficient_visitor.cpp)

target_include_directories(shoccs-matrices PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)
target_link_libraries(shoccs-matrices PUBLIC fields)


add_unit_test(dense "matrices" shoccs-matrices shoccs-random)
add_unit_test(csr "matrices" shoccs-matrices shoccs-random)
add_unit_test(unit_stride_visitor "matrices" shoccs-matrices)
add_unit_test(coefficient_visitor "matrices" shoccs-matrices)

if (BUILD_TESTING)
  add_executable(t-block block.t.cpp)
  target_link_libraries(t-block Catch2::Catch2 shoccs-matrices shoccs-random Kokkos::kokkos)
  add_test(NAME t-block COMMAND t-block)
  set_tests_properties(t-block PROPERTIES LABELS "matrices")

  add_executable(t-circulant circulant.t.cpp)
  target_link_libraries(t-circulant Catch2::Catch2 shoccs-matrices shoccs-random Kokkos::kokkos)
  add_test(NAME t-circulant COMMAND t-circulant)
  set_tests_properties(t-circulant PROPERTIES LABELS "matrices")

  add_executable(t-inner_block inner_block.t.cpp)
  target_link_libraries(t-inner_block Catch2::Catch2 shoccs-matrices shoccs-random Kokkos::kokkos)
  add_test(NAME t-inner_block COMMAND t-inner_block)
  set_tests_properties(t-inner_block PROPERTIES LABELS "matrices")
endif()


cmake_minimum_required(VERSION 3.20)
project(scipio LANGUAGES C CXX)

cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0079 NEW)

find_package(GSL 2.7 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED fftw3)

find_package(BLAS)

add_library(scipio scipio.cc)

target_include_directories(scipio PRIVATE ${FFTW3_INCLUDE_DIRS} ${GSL_INCLUDE_DIRS})
set_target_properties(scipio PROPERTIES OUTPUT_NAME scipio)

target_include_directories(scipio INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(scipio INTERFACE ${FFTW3_LIBRARIES} ${GSL_LIBRARY} scipio)

target_link_libraries(scipio INTERFACE $<$<BOOL:${BLAS_FOUND}>:${BLAS_LIBRARIES}>)
target_link_options(scipio INTERFACE $<$<BOOL:${BLAS_FOUND}>:${BLAS_LINKER_FLAGS}>)
target_link_libraries(scipio INTERFACE $<$<NOT:$<BOOL:${BLAS_FOUND}>>:${GSL_CBLAS_LIBRARY}>)

# vim: set ft=cmake:

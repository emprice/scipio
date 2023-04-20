cmake_minimum_required(VERSION 3.20)
project(scipio LANGUAGES CXX)

cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0079 NEW)

find_package(GSL 2.7 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED fftw3)

find_package(BLAS)

add_library(scipio INTERFACE)

target_sources(scipio INTERFACE scipio.cc)
target_include_directories(scipio INTERFACE ${FFTW3_INCLUDE_DIRS} ${GSL_INCLUDE_DIRS})

target_include_directories(scipio INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(scipio INTERFACE ${FFTW3_LIBRARIES} ${GSL_LIBRARY})

target_link_libraries(scipio INTERFACE $<$<BOOL:${BLAS_FOUND}>:${BLAS_LIBRARIES}>)
target_link_options(scipio INTERFACE $<$<BOOL:${BLAS_FOUND}>:${BLAS_LINKER_FLAGS}>)
target_link_libraries(scipio INTERFACE $<$<NOT:$<BOOL:${BLAS_FOUND}>>:${GSL_CBLAS_LIBRARY}>)

add_executable(demo demo.cc)
target_link_libraries(demo scipio)

# vim: set ft=cmake:

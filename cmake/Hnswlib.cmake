# Routing uses a checked-in IVF variant and a fetched, unmodified comparison. Upstream is never
# patched or configured as a subproject.
include_guard(GLOBAL)
include(FetchContent)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()
FetchContent_Declare(
    ivf_hnswlib
    URL https://codeload.github.com/nmslib/hnswlib/tar.gz/refs/tags/v0.8.0
    URL_HASH SHA256=cf61d7dd8dc9bfba7f4abe0ed26698e90ac9f49a0badb2e1b0f3ba89b72cf3bb)
FetchContent_GetProperties(ivf_hnswlib)
if(NOT ivf_hnswlib_POPULATED)
    FetchContent_Populate(ivf_hnswlib)
endif()
# Header consumers opt in explicitly; no generation or Python configure step.
add_library(ivf_hnswlib_headers INTERFACE)
target_include_directories(
    ivf_hnswlib_headers SYSTEM INTERFACE "${IVF_ROOT}/dependencies/hnswlib/include"
                                         "${ivf_hnswlib_SOURCE_DIR}")
set_property(TARGET ivf_hnswlib_headers PROPERTY IVF_UPSTREAM_SOURCE_DIR
                                                 "${ivf_hnswlib_SOURCE_DIR}")
option(IVF_ROUTER_SANITIZER_SCALAR "Disable manual HNSW SIMD in sanitizer builds" ON)
if(CMAKE_CXX_FLAGS MATCHES "sanitize" AND IVF_ROUTER_SANITIZER_SCALAR)
    target_compile_definitions(ivf_hnswlib_headers INTERFACE NO_MANUAL_VECTORIZATION)
endif()

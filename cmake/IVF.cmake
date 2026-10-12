# Dependency-light IVF libraries, shared by native and isolated builds.
get_filename_component(IVF_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
# Imported Threads targets are directory-scoped. Discover them in each caller, even when the library
# definitions below were already included by a sibling.
find_package(Threads REQUIRED)
include_guard(GLOBAL)

# Finite-coordinate validation must survive the native parent's -Ofast flags. Public usage
# requirements keep all IVF consumers on the same numeric policy.
add_library(ivf_numeric_policy INTERFACE)
if(NOT MSVC)
    target_compile_options(ivf_numeric_policy INTERFACE -fno-fast-math)
endif()

add_library(
    ivf_contracts STATIC
    "${IVF_ROOT}/src/bufann/ivf_config.cpp" "${IVF_ROOT}/src/bufann/ivf_artifacts.cpp"
    "${IVF_ROOT}/src/bufann/ivf_io.cpp")
target_include_directories(ivf_contracts PUBLIC "${IVF_ROOT}/include")
target_compile_features(ivf_contracts PUBLIC cxx_std_20)
target_link_libraries(ivf_contracts PUBLIC ivf_numeric_policy)

add_library(ivf_centroids STATIC "${IVF_ROOT}/src/bufann/ivf_centroids.cpp")
target_link_libraries(ivf_centroids PUBLIC ivf_contracts Threads::Threads)

# This core uses only the standard library: no lists, router, or fetch required.
add_library(ivf_cost_model_core STATIC "${IVF_ROOT}/src/bufann/ivf_cost_model.cpp")
target_include_directories(ivf_cost_model_core PUBLIC "${IVF_ROOT}/include")
target_compile_features(ivf_cost_model_core PUBLIC cxx_std_20)
target_link_libraries(ivf_cost_model_core PUBLIC ivf_numeric_policy)

option(IVF_BUILD_ROUTING "Build centroid HNSW router (fetches pinned hnswlib)" OFF)
if(IVF_BUILD_ROUTING)
    include("${CMAKE_CURRENT_LIST_DIR}/Hnswlib.cmake")
    add_library(
        ivf_router STATIC
        "${IVF_ROOT}/src/bufann/ivf_router.cpp" "${IVF_ROOT}/src/bufann/ivf_patched_graph.cpp"
        "${IVF_ROOT}/src/bufann/ivf_pca.cpp")
    target_link_libraries(
        ivf_router
        PUBLIC ivf_centroids
        PRIVATE ivf_hnswlib_headers)

    add_library(ivf_lists STATIC "${IVF_ROOT}/src/bufann/ivf_lists.cpp"
                                 "${IVF_ROOT}/src/bufann/ivf_posting_lists.cpp")
    target_link_libraries(ivf_lists PUBLIC ivf_router Threads::Threads)
endif()

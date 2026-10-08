include("${CMAKE_CURRENT_LIST_DIR}/IVFCentroids.cmake")
if(TARGET ivf_router AND NOT TARGET ivf_lists)
    add_library(ivf_lists STATIC "${IVF_ROOT}/src/bufann/ivf_lists.cpp"
        "${IVF_ROOT}/src/bufann/ivf_posting_lists.cpp")
    target_link_libraries(ivf_lists PUBLIC ivf_router)
    if(NOT MSVC)
        target_compile_options(ivf_lists PRIVATE -fno-fast-math)
    endif()
endif()

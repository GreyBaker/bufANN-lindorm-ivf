# Shared by the normal BufANN build and the dependency-light contract tests.
get_filename_component(IVF_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
if(NOT TARGET ivf_contracts)
    add_library(ivf_contracts STATIC
        "${IVF_ROOT}/src/bufann/ivf_config.cpp"
        "${IVF_ROOT}/src/bufann/ivf_artifacts.cpp"
        "${IVF_ROOT}/src/bufann/ivf_io.cpp")
    target_include_directories(ivf_contracts PUBLIC "${IVF_ROOT}/include")
    target_compile_features(ivf_contracts PUBLIC cxx_std_20)
    # Input validation must reject NaN/Inf even in BufANN's -Ofast builds.
    if(NOT MSVC)
        target_compile_options(ivf_contracts PRIVATE -fno-fast-math)
    endif()
endif()

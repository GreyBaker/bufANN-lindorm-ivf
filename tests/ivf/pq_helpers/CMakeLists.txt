cmake_minimum_required(VERSION 3.16)
project(IVFStandalonePQHelpers LANGUAGES CXX)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "This installed-library standalone-helper wrapper targets Linux/WSL")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(IVF_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(DISKANN "${IVF_ROOT}/baselines/DiskANN")
set(MKL_ROOT
    "$ENV{MKLROOT}"
    CACHE PATH "Installed MKL prefix")
set(MKL_LIBRARY_DIR
    "${MKL_ROOT}/lib"
    CACHE PATH "Installed MKL library directory")
set(INTEL_OMP_LIBRARY_DIR
    ""
    CACHE PATH "Installed Intel OpenMP library directory")
if(NOT EXISTS "${MKL_ROOT}/include/mkl.h")
    message(FATAL_ERROR "Set MKL_ROOT to the administrator-installed MKL prefix")
endif()
foreach(name mkl_intel_ilp64 mkl_intel_thread mkl_core)
    find_library(
        ${name}_LIBRARY
        NAMES ${name}
        HINTS "${MKL_LIBRARY_DIR}")
    if(NOT ${name}_LIBRARY)
        message(FATAL_ERROR "Missing installed ${name}; set MKL_LIBRARY_DIR")
    endif()
endforeach()
find_library(
    IOMP_LIBRARY
    NAMES iomp5
    HINTS "${INTEL_OMP_LIBRARY_DIR}")
find_library(AIO_LIBRARY NAMES aio)
find_library(TCMALLOC_LIBRARY NAMES tcmalloc)
if(NOT IOMP_LIBRARY
   OR NOT AIO_LIBRARY
   OR NOT TCMALLOC_LIBRARY)
    message(FATAL_ERROR "Installed iomp5, libaio and tcmalloc are required")
endif()
find_package(OpenMP REQUIRED)
# Compile the existing baseline helper/quantizer sources without editing them. This does not
# configure or qualify the baseline's canonical-index build.
add_library(
    standalone_pq STATIC
    "${DISKANN}/src/partition_and_pq.cpp"
    "${DISKANN}/src/math_utils.cpp"
    "${DISKANN}/src/utils.cpp"
    "${DISKANN}/src/distance.cpp"
    "${DISKANN}/src/logger.cpp"
    "${DISKANN}/src/ann_exception.cpp"
    "${DISKANN}/src/linux_aligned_file_reader.cpp")
target_include_directories(standalone_pq PUBLIC "${DISKANN}/include" "${MKL_ROOT}/include")
target_compile_definitions(standalone_pq PUBLIC MKL_ILP64 USE_AVX2 USE_ACCELERATED_PQ)
target_compile_options(standalone_pq PUBLIC -march=native -fopenmp)
target_compile_options(standalone_pq PUBLIC $<$<CONFIG:Release>:-Ofast>
                                            $<$<CONFIG:Release>:-mtune=native>)
target_link_libraries(
    standalone_pq
    PUBLIC ${mkl_intel_ilp64_LIBRARY}
           ${mkl_intel_thread_LIBRARY}
           ${mkl_core_LIBRARY}
           ${IOMP_LIBRARY}
           pthread
           m
           dl
           ${AIO_LIBRARY}
           ${TCMALLOC_LIBRARY})
foreach(tool build_pq_pivot_standalone build_pq_codes_standalone)
    add_executable(${tool} "${DISKANN}/tests/${tool}.cpp")
    target_link_libraries(${tool} PRIVATE standalone_pq)
endforeach()

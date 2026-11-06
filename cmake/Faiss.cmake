# Unmodified, checksum-pinned CPU FAISS. No GPU, Python, demos, or upstream tests.
include_guard(GLOBAL)
include(FetchContent)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()
FetchContent_Declare(ivf_faiss
    URL https://codeload.github.com/facebookresearch/faiss/tar.gz/refs/tags/v1.9.0
    URL_HASH SHA256=f6721f1a479b0bdd8547e5bb0dbfa872643e7ea37686c6bf6fc99fb933b3fe4f)
FetchContent_GetProperties(ivf_faiss)
if(NOT ivf_faiss_POPULATED)
    FetchContent_Populate(ivf_faiss)
endif()
find_package(OpenMP REQUIRED COMPONENTS CXX)
# Consume exactly the pinned upstream CPU source manifest, without configuring
# its unrelated build targets. Source files themselves are never patched.
file(READ "${ivf_faiss_SOURCE_DIR}/faiss/CMakeLists.txt" _faiss_manifest)
string(REGEX MATCH "set\\(FAISS_SRC([^)]*)\\)" _faiss_sources "${_faiss_manifest}")
if(NOT _faiss_sources)
    message(FATAL_ERROR "Pinned FAISS CPU source manifest changed")
endif()
string(REGEX REPLACE "^set\\(FAISS_SRC|\\)$" "" _faiss_sources "${_faiss_sources}")
string(REGEX REPLACE "[ \n\r\t]+" ";" _faiss_sources "${_faiss_sources}")
list(FILTER _faiss_sources EXCLUDE REGEX "^$")
list(REMOVE_DUPLICATES _faiss_sources)
if(NOT WIN32)
    list(APPEND _faiss_sources invlists/OnDiskInvertedLists.cpp)
endif()
list(TRANSFORM _faiss_sources PREPEND "${ivf_faiss_SOURCE_DIR}/faiss/")
add_library(ivf_faiss_cpu STATIC ${_faiss_sources})
set_property(TARGET ivf_faiss_cpu PROPERTY LINK_LIBRARIES "")
set_property(TARGET ivf_faiss_cpu PROPERTY IVF_UPSTREAM_SOURCE_DIR "${ivf_faiss_SOURCE_DIR}")
target_include_directories(ivf_faiss_cpu SYSTEM PUBLIC "${ivf_faiss_SOURCE_DIR}")
target_link_libraries(ivf_faiss_cpu PUBLIC OpenMP::OpenMP_CXX)
target_compile_features(ivf_faiss_cpu PUBLIC cxx_std_17)
target_link_libraries(ivf_faiss_cpu PRIVATE ivf_numeric_policy)
set_target_properties(ivf_faiss_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MKL_ROOT)
    # The native backend was compiled with MKL_ILP64. FAISS's Fortran declarations
    # must use the same 64-bit integers; never introduce LP64 symbols beside it.
    foreach(component mkl_intel_ilp64 mkl_sequential mkl_core)
        find_library(IVF_${component} NAMES ${component} PATHS "${MKL_LIBRARY_DIR}" REQUIRED NO_DEFAULT_PATH)
        target_link_libraries(ivf_faiss_cpu PUBLIC "${IVF_${component}}")
    endforeach()
    target_compile_definitions(ivf_faiss_cpu PRIVATE FINTEGER=int64_t)
else()
    # Isolated routing builds use the system LP64 BLAS/LAPACK, not native MKL.
    # Debian/Ubuntu alternatives can point even BLA_VENDOR=Generic's libblas.so
    # at mkl_rt, which dynamically loads Intel OpenMP beside GNU OpenMP. Prefer
    # the installed reference-library directories, including their runtime paths.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        find_library(IVF_REFERENCE_BLAS NAMES blas
            PATHS "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/blas" "/usr/lib/blas"
            NO_DEFAULT_PATH)
        find_library(IVF_REFERENCE_LAPACK NAMES lapack
            PATHS "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/lapack" "/usr/lib/lapack"
            NO_DEFAULT_PATH)
        if(IVF_REFERENCE_BLAS AND IVF_REFERENCE_LAPACK)
            # FindLAPACK invokes FindBLAS again. Cache the selected paths so
            # both discovery passes see the same libraries on a fresh tree.
            set(BLAS_blas_LIBRARY "${IVF_REFERENCE_BLAS}" CACHE FILEPATH
                "IVF isolated reference BLAS" FORCE)
            set(LAPACK_lapack_LIBRARY "${IVF_REFERENCE_LAPACK}" CACHE FILEPATH
                "IVF isolated reference LAPACK" FORCE)
        endif()
    endif()
    set(BLA_SIZEOF_INTEGER 4)
    set(BLA_VENDOR Generic)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    foreach(library IN LISTS BLAS_LIBRARIES LAPACK_LIBRARIES)
        if(IS_ABSOLUTE "${library}")
            get_filename_component(real_library "${library}" REALPATH)
            if(real_library MATCHES "mkl|iomp5")
                message(FATAL_ERROR "Isolated FAISS requires non-MKL LP64 BLAS/LAPACK; ${library} resolves to ${real_library}. Install reference libblas-dev/liblapack-dev or select their explicit library paths.")
            endif()
        endif()
    endforeach()
    target_compile_definitions(ivf_faiss_cpu PRIVATE FINTEGER=int)
    target_link_libraries(ivf_faiss_cpu PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

# Root native CMake supplies Intel-threaded MKL/iomp5 as directory defaults.
# IVF opts out of that interface, using sequential MKL and compiler OpenMP only.
# Unrelated baseline targets retain their original link policy.
function(ivf_native_link_policy target)
    if(MKL_ROOT)
        foreach(property LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
            get_target_property(libraries ${target} ${property})
            if(libraries)
                list(FILTER libraries EXCLUDE REGEX "^(mkl_intel_thread|iomp5)$|^\\$<LINK_ONLY:(mkl_intel_thread|iomp5)>$")
                set_property(TARGET ${target} PROPERTY ${property} "${libraries}")
            endif()
        endforeach()
        target_link_libraries(${target} PRIVATE "${IVF_mkl_sequential}")
    endif()
endfunction()
ivf_native_link_policy(ivf_faiss_cpu)
